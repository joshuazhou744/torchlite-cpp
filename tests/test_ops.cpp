#include <iostream>
#include <tl/tensor.h>
#include <tl/ops.h>
#include <cassert>
#include <cmath>

// helper function for float comparison
bool is_close(float a, float b, float e = 1e-5) {
  return std::abs( a - b ) < e;
}

void test_ops() {
  // test element wise addition
  tl::Tensor a({2, 2});
  tl::Tensor b({2, 2});
  a.data()[0] = 1.0f; a.data()[1] = 2.0f; // [[1.0, 2.0], [0.0, 0.0]]
  b.data()[0] = 3.0f; b.data()[1] = 4.0f; // [[3.0, 4.0], [0.0, 0.0]]

  tl::Tensor res_add = tl::add(a, b);
  assert(res_add.data()[0] == 4.0f);
  assert(res_add.data()[1] == 6.0f);

  // test ReLU edge cases (0 and negative)
  tl::Tensor c({1, 3});
  c.data()[0] = -5.0f; c.data()[1] = 0.0f; c.data()[2] = 5.0f;

  tl::Tensor res_relu = tl::relu(c);
  assert(res_relu.data()[0] == 0.0f);
  assert(res_relu.data()[1] == 0.0f);
  assert(res_relu.data()[2] == 5.0f);

  // test matmul (indexing logic check)
  tl::Tensor A({2, 3}); // [[1, 2, 3], [4, 5, 6]]
  tl::Tensor B({3, 2}); // [[7, 8], [9, 10], [11, 12]]

  for(int i = 0; i < A.numel(); ++i) A.data()[i] = i + 1.0f;
  for(int i = 0; i < B.numel(); ++i) B.data()[i] = i + 7.0f;

  tl::Tensor res_matmul = tl::matmul(A, B);
  // Row 0, Col 0: (1*7) + (2*9) + (3*11) = 58
  assert(res_matmul.data()[0] == 58.0f);
  // Row 1, Col 1: (4*8) + (5*10) + (6*12) = 154
  assert(res_matmul.data()[3] == 154.0f);

  // test sigmoid
  tl::Tensor d({1, 1});
  d.data()[0] = 0.0f;
  tl::Tensor res_sigmoid = tl::sigmoid(d);
  // sigmoid(0) = 0.5
  assert(is_close(res_sigmoid.data()[0], 0.5f));

  // test transpose
  tl::Tensor transpose_in({2, 3});
  for (int i = 0; i < 6; ++i) transpose_in.data()[i] = (float) i;
  tl::Tensor transpose_out = tl::transpose(transpose_in, 0, 1);

  assert(transpose_out.sizes()[0] == 3);
  assert(transpose_out.sizes()[1] == 2);
  assert(transpose_out.data()[1] == 3.0f);


  // test broadcasting on a (2, 3) matrix + (1, 3) vector
  tl::Tensor mat({2, 3});
  tl::Tensor row({1, 3});
  for (int i = 0; i < 6; ++i) mat.data()[i] = 10.0f;
  for (int i = 0; i < 3; ++i) row.data()[i] = (float) i;

  tl::Tensor broadcast_res = tl::add(mat, row);
  assert(broadcast_res.sizes()[0] == 2 && broadcast_res.sizes()[1] == 3);
  assert(broadcast_res.data()[0] == 10.0f); // 10 + 0
  assert(broadcast_res.data()[2] == 12.0f); // 10 + 2
  assert(broadcast_res.data()[5] == 12.0f); // 10 + 2

  // test batched matmul: (2, 3, 2, 2) @ (2, 4) -> (2, 3, 2, 4)
  // simulate a real multi-head attention batched matmul
  tl::Tensor input({2, 3, 2, 2}); // 4D input matrix
  for (int i = 0; i < input.numel(); ++i) input.data()[i] = 1.0f;

  tl::Tensor weights({2, 4}); // 2D weight matrix
  for (int i = 0; i < weights.numel(); ++i) weights.data()[i] = 0.5f;

  // execute matmul and assert tests
  tl::Tensor output = tl::matmul(input, weights);

  assert(output.sizes().size() == 4);
  assert(output.sizes()[0] == 2); // batch dim
  assert(output.sizes()[1] == 3); // heads
  assert(output.sizes()[2] == 2); // seq
  assert(output.sizes()[3] == 4); // out dimension

  // value check
  for (int i = 0; i < output.numel(); ++i) {
    assert(is_close(output.data()[i], 1.0f));
  }

  // test softmax normalization and order
  tl::Tensor softmax_tensor({3});
  softmax_tensor.data()[0] = 1.0f;
  softmax_tensor.data()[1] = 2.0f;
  softmax_tensor.data()[2] = 3.0f;

  tl::Tensor softmax_out = tl::softmax(softmax_tensor);

  float sum = softmax_out.data()[0] + softmax_out.data()[1] + softmax_out.data()[2];
  assert(is_close(sum, 1.0f));
  assert(softmax_out.data()[2] > softmax_out.data()[1] && softmax_out.data()[1] > softmax_out.data()[0]);

  // test scale
  tl::Tensor scale_tensor({2});
  scale_tensor.data()[0] = 2.0f;
  scale_tensor.data()[1] = 4.0f;

  tl::Tensor scale_out = tl::scale(scale_tensor, 0.5f);
  assert(is_close(scale_out.data()[0], 1.0f));
  assert(is_close(scale_out.data()[1], 2.0f));


  std::cout << "ops tests passed" << std::endl;
}
