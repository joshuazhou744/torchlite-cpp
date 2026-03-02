#include <iostream>
#include <tl/tensor.h>
#include <tl/activation.h>
#include <cassert>
#include <cmath>

// helper function for float comparison
bool is_close_act(float a, float b, float e = 1e-5) {
  return std::abs( a - b ) < e;
}

void test_activation() {
  // test ReLU edge cases (0 and negative)
  tl::Tensor c({1, 3});
  c.data()[0] = -5.0f; c.data()[1] = 0.0f; c.data()[2] = 5.0f;

  tl::Tensor res_relu = tl::relu(c);
  assert(res_relu.data()[0] == 0.0f);
  assert(res_relu.data()[1] == 0.0f);
  assert(res_relu.data()[2] == 5.0f);

  // test sigmoid(0) = 0.5
  tl::Tensor d({1, 1});
  d.data()[0] = 0.0f;
  tl::Tensor res_sigmoid = tl::sigmoid(d);
  assert(is_close_act(res_sigmoid.data()[0], 0.5f));

  // test gelu: gelu(0) = 0, gelu positive > 0, gelu large negative ~ 0
  tl::Tensor g({3});
  g.data()[0] = 0.0f;
  g.data()[1] = 1.0f;
  g.data()[2] = -3.0f;

  tl::Tensor res_gelu = tl::gelu(g);
  assert(is_close_act(res_gelu.data()[0], 0.0f));
  assert(is_close_act(res_gelu.data()[1], 0.8412f, 1e-4));
  assert(is_close_act(res_gelu.data()[2], -0.0036f, 1e-3));

  std::cout << "activation tests passed" << std::endl;
}
