#include <iostream>
#include <cassert>
#include <cmath>
#include <tl/tensor.h>
#include <tl/ops.h>
#include <tl/activation.h>
#include <tl/factory.h>

static bool close(float a, float b, float e = 1e-4f) {
  return std::abs(a - b) < e;
}

void test_autograd() {
  std::cout << "Running autograd tests...\n";

  // AddBackward: gradient passes through unchanged to both inputs
  {
    tl::Tensor a({3});
    tl::Tensor b({3});
    for (int i = 0; i < 3; ++i) { a.data()[i] = i + 1.0f; b.data()[i] = i + 10.0f; }
    a.set_requires_grad(true);
    b.set_requires_grad(true);

    tl::Tensor c = tl::add(a, b);
    c.backward();  // seeds grad = [1, 1, 1]

    for (int i = 0; i < 3; ++i) {
      assert(close(a.grad().data()[i], 1.0f));
      assert(close(b.grad().data()[i], 1.0f));
    }
    std::cout << "  AddBackward ok\n";
  }

  // SubBackward: +grad for a, -grad for b
  {
    tl::Tensor a({2});
    tl::Tensor b({2});
    a.data()[0] = 5.0f; a.data()[1] = 6.0f;
    b.data()[0] = 1.0f; b.data()[1] = 2.0f;
    a.set_requires_grad(true);
    b.set_requires_grad(true);

    tl::Tensor c = tl::sub(a, b);
    c.backward();

    assert(close(a.grad().data()[0], 1.0f));
    assert(close(a.grad().data()[1], 1.0f));
    assert(close(b.grad().data()[0], -1.0f));
    assert(close(b.grad().data()[1], -1.0f));
    std::cout << "  SubBackward ok\n";
  }

  // MulBackward: d/da = b, d/db = a
  {
    tl::Tensor a({2});
    tl::Tensor b({2});
    a.data()[0] = 2.0f; a.data()[1] = 3.0f;
    b.data()[0] = 4.0f; b.data()[1] = 5.0f;
    a.set_requires_grad(true);
    b.set_requires_grad(true);

    tl::Tensor c = tl::mul(a, b);
    c.backward();

    assert(close(a.grad().data()[0], 4.0f));  // = b[0]
    assert(close(a.grad().data()[1], 5.0f));  // = b[1]
    assert(close(b.grad().data()[0], 2.0f));  // = a[0]
    assert(close(b.grad().data()[1], 3.0f));  // = a[1]
    std::cout << "  MulBackward ok\n";
  }

  // ReluBackward: grad passes where input > 0, else 0
  {
    tl::Tensor x({4});
    x.data()[0] = -1.0f; x.data()[1] = 0.0f;
    x.data()[2] = 2.0f;  x.data()[3] = 3.0f;
    x.set_requires_grad(true);

    tl::Tensor y = tl::relu(x);
    y.backward();

    assert(close(x.grad().data()[0], 0.0f));
    assert(close(x.grad().data()[1], 0.0f));
    assert(close(x.grad().data()[2], 1.0f));
    assert(close(x.grad().data()[3], 1.0f));
    std::cout << "  ReluBackward ok\n";
  }

  // SigmoidBackward: d/dx = sig(x) * (1 - sig(x))
  {
    tl::Tensor x({1});
    x.data()[0] = 0.0f;  // sig(0) = 0.5, derivative = 0.25
    x.set_requires_grad(true);

    tl::Tensor y = tl::sigmoid(x);
    y.backward();

    assert(close(x.grad().data()[0], 0.25f));
    std::cout << "  SigmoidBackward ok\n";
  }

  // Chain rule: y = (a * b) + c, d/da = b, d/db = a, d/dc = 1
  {
    tl::Tensor a({2}); tl::Tensor b({2}); tl::Tensor c({2});
    a.data()[0] = 2.0f; a.data()[1] = 3.0f;
    b.data()[0] = 4.0f; b.data()[1] = 5.0f;
    c.data()[0] = 1.0f; c.data()[1] = 1.0f;
    a.set_requires_grad(true);
    b.set_requires_grad(true);
    c.set_requires_grad(true);

    tl::Tensor y = tl::add(tl::mul(a, b), c);
    // intermediates version
    // tl::Tensor prod = tl::mul(a, b);
    // tl::Tensor y = tl::add(prod, c);
    y.backward();

    assert(close(a.grad().data()[0], 4.0f));  // b[0]
    assert(close(a.grad().data()[1], 5.0f));  // b[1]
    assert(close(b.grad().data()[0], 2.0f));  // a[0]
    assert(close(b.grad().data()[1], 3.0f));  // a[1]
    assert(close(c.grad().data()[0], 1.0f));
    assert(close(c.grad().data()[1], 1.0f));
    std::cout << "  Chain rule (mul+add) ok\n";
  }

  // Gradient accumulation: same tensor used twice
  // y = a * a, dy/da = 2a (d/da from first factor + d/da from second factor)
  {
    tl::Tensor a({2});
    a.data()[0] = 3.0f; a.data()[1] = 4.0f;
    a.set_requires_grad(true);

    tl::Tensor y = tl::mul(a, a);
    y.backward();

    // each branch contributes a to the grad, accumulated => 2a
    assert(close(a.grad().data()[0], 6.0f));  // 2 * 3
    assert(close(a.grad().data()[1], 8.0f));  // 2 * 4
    std::cout << "  Gradient accumulation (a * a) ok\n";
  }

  // MatmulBackward (Level 1 — pure 2D): dA = grad @ B^T, dB = A^T @ grad
  // A = [[1,2,3],[4,5,6]]          shape (2,3)
  // B = [[1,2],[3,4],[5,6]]        shape (3,2)
  // Seed grad = ones(2,2).
  // dA[i,k] = sum_n(grad[i,n] * B[k,n]) = sum_n B[k,n] = row sum of B.
  //   B row sums: [1+2, 3+4, 5+6] = [3, 7, 11]
  //   dA = [[3,7,11], [3,7,11]]
  // dB[k,n] = sum_i(A[i,k] * grad[i,n]) = sum_i A[i,k] = col sum of A.
  //   A col sums: [1+4, 2+5, 3+6] = [5, 7, 9]
  //   dB = [[5,5], [7,7], [9,9]]
  {
    tl::Tensor A({2, 3});
    tl::Tensor B({3, 2});
    for (int i = 0; i < 6; ++i) { A.data()[i] = i + 1.0f; B.data()[i] = i + 1.0f; }
    A.set_requires_grad(true);
    B.set_requires_grad(true);

    tl::Tensor C = tl::matmul(A, B);
    C.backward();

    assert(!A.grad().empty());  // sanity: tracking actually ran
    assert(!B.grad().empty());

    // check dA
    assert(close(A.grad().data()[0], 3.0f));
    assert(close(A.grad().data()[1], 7.0f));
    assert(close(A.grad().data()[2], 11.0f));
    assert(close(A.grad().data()[3], 3.0f));
    assert(close(A.grad().data()[4], 7.0f));
    assert(close(A.grad().data()[5], 11.0f));

    // check dB
    assert(close(B.grad().data()[0], 5.0f));
    assert(close(B.grad().data()[1], 5.0f));
    assert(close(B.grad().data()[2], 7.0f));
    assert(close(B.grad().data()[3], 7.0f));
    assert(close(B.grad().data()[4], 9.0f));
    assert(close(B.grad().data()[5], 9.0f));

    std::cout << "  MatmulBackward (2D) ok\n";

    // MatmulBackward Level 2 — batched (B, M, K) @ (B, K, N)
    // A = ones(2, 2, 3), B = ones(2, 3, 2), seed grad = ones(2, 2, 2)
    // dA[b,i,k] = sum_n grad[b,i,n] * B[b,k,n] = sum_n 1 = N = 2
    // dB[b,k,n] = sum_i A[b,i,k] * grad[b,i,n] = sum_i 1 = M = 2
    {
      tl::Tensor A({2, 2, 3});
      tl::Tensor B({2, 3, 2});
      for (int i = 0; i < 12; ++i) { A.data()[i] = 1.0f; B.data()[i] = 1.0f; }
      A.set_requires_grad(true);
      B.set_requires_grad(true);

      tl::Tensor C = tl::matmul(A, B);
      C.backward();

      for (int i = 0; i < 12; ++i) assert(close(A.grad().data()[i], 2.0f));
      for (int i = 0; i < 12; ++i) assert(close(B.grad().data()[i], 2.0f));
      std::cout << "  MatmulBackward (batched 3D) ok\n";
    }

    // MatmulBackward Level 3 — broadcast batch: (B, M, K) @ (K, N)
    // A = ones(2, 3, 4), B = ones(4, 2), seed grad = ones(2, 3, 2)
    // dA: grad @ B^T -> (2,3,2) @ (2,4) = (2,3,4), each entry = sum of B row = 2
    // dB: A^T @ grad -> (4,3) @ (3,2) summed over batch -> (4,2), each entry = M*B = 3*2 = 6
    {
      tl::Tensor A({2, 3, 4});
      tl::Tensor B({4, 2});
      for (int i = 0; i < 24; ++i) A.data()[i] = 1.0f;
      for (int i = 0; i < 8;  ++i) B.data()[i] = 1.0f;
      A.set_requires_grad(true);
      B.set_requires_grad(true);

      tl::Tensor C = tl::matmul(A, B);  // (2, 3, 2)
      C.backward();

      for (int i = 0; i < 24; ++i) assert(close(A.grad().data()[i], 2.0f));
      for (int i = 0; i < 8;  ++i) assert(close(B.grad().data()[i], 6.0f));
      std::cout << "  MatmulBackward (broadcast batch) ok\n";
    }

     // MatmulBackward Level 4a — (K,) @ (K, N): vector-matrix
    {
      tl::Tensor a({4});
      tl::Tensor B({4, 3});
      for (int i = 0; i < 4; ++i) a.data()[i] = 1.0f;
      for (int i = 0; i < 12; ++i) B.data()[i] = 1.0f;
      a.set_requires_grad(true);
      B.set_requires_grad(true);

      tl::Tensor c = tl::matmul(a, B);  // (3,)
      c.backward();

      // da = grad @ B^T, grad=ones(3), B^T=(3,4) -> da=ones(4)*3=3
      for (int i = 0; i < 4; ++i) assert(close(a.grad().data()[i], 3.0f));
      // dB = a^T @ grad -> (4,1)@(1,3) = (4,3), each entry = 1
      for (int i = 0; i < 12; ++i) assert(close(B.grad().data()[i], 1.0f));
      std::cout << "  MatmulBackward (vec @ mat) ok\n";
    }

    // MatmulBackward Level 4b — (M, K) @ (K,): matrix-vector
    {
      tl::Tensor A({3, 4});
      tl::Tensor b({4});
      for (int i = 0; i < 12; ++i) A.data()[i] = 1.0f;
      for (int i = 0; i < 4; ++i) b.data()[i] = 1.0f;
      A.set_requires_grad(true);
      b.set_requires_grad(true);

      tl::Tensor c = tl::matmul(A, b);  // (3,)
      c.backward();

      // dA = grad @ b^T, grad=ones(3), b^T=(1,4) -> dA=(3,4) all ones
      for (int i = 0; i < 12; ++i) assert(close(A.grad().data()[i], 1.0f));
      // db = A^T @ grad -> (4,3)@(3,) -> (4,) each = sum of col of A = 3
      for (int i = 0; i < 4; ++i) assert(close(b.grad().data()[i], 3.0f));
      std::cout << "  MatmulBackward (mat @ vec) ok\n";
    }


  }

  std::cout << "autograd tests passed.\n";
}
