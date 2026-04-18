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

    tl::Tensor prod = tl::mul(a, b);
    tl::Tensor y = tl::add(prod, c);
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

  std::cout << "Autograd tests passed.\n\n";
}
