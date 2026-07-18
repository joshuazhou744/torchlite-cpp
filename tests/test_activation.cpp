#include <iostream>
#include <tl/tensor.h>
#include <tl/activation.h>
#include <cassert>
#include "test_utils.h"

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
  assert(is_close(res_sigmoid.data()[0], 0.5f));

  // test gelu: gelu(0) = 0, gelu positive > 0, gelu large negative ~ 0
  tl::Tensor g({3});
  g.data()[0] = 0.0f;
  g.data()[1] = 1.0f;
  g.data()[2] = -3.0f;

  tl::Tensor res_gelu = tl::gelu(g);
  assert(is_close(res_gelu.data()[0], 0.0f));
  assert(is_close(res_gelu.data()[1], 0.8412f, 1e-4));
  assert(is_close(res_gelu.data()[2], -0.0036f, 1e-3));

  // test gelu_exact: x * 0.5 * (1 + erf(x / sqrt(2)))
  // tolerances are tight (1e-5) on purpose: exact vs tanh differ by ~1.5e-4 at x=1
  // and ~4e-4 at x=-3, so this fails if gelu_exact accidentally uses the approximation
  tl::Tensor ge({5});
  ge.data()[0] = 0.0f; ge.data()[1] = 1.0f; ge.data()[2] = -1.0f;
  ge.data()[3] = 2.0f; ge.data()[4] = -3.0f;

  tl::Tensor res_ge = tl::gelu_exact(ge);
  assert(is_close(res_ge.data()[0], 0.0f));
  assert(is_close(res_ge.data()[1], 0.8413447f, 1e-5));
  assert(is_close(res_ge.data()[2], -0.1586553f, 1e-5));
  assert(is_close(res_ge.data()[3], 1.9544997f, 1e-5));
  assert(is_close(res_ge.data()[4], -0.0040497f, 1e-5));

  // test SiLU: silu(x) = x * sigmoid(x)
  // silu(0) = 0 * 0.5 = 0
  // silu(1) = 1 * sigmoid(1) = 1 * 0.7311 = 0.7311
  // silu(-5) ~ -5 * 0.0067 ~ -0.0335 (nearly zero for large negative)
  tl::Tensor s({3});
  s.data()[0] = 0.0f; s.data()[1] = 1.0f; s.data()[2] = -5.0f;
  tl::Tensor res_silu = tl::silu(s);
  assert(is_close(res_silu.data()[0], 0.0f));
  assert(is_close(res_silu.data()[1], 0.7311f, 1e-4));
  assert(is_close(res_silu.data()[2], -0.0335f, 1e-3));

  // test tanh: tanh(0) = 0, tanh(1) = 0.7616, tanh(-1) = -0.7616, tanh(large) ~ 1
  tl::Tensor t({4});
  t.data()[0] = 0.0f; t.data()[1] = 1.0f; t.data()[2] = -1.0f; t.data()[3] = 20.0f;
  tl::Tensor res_tanh = tl::tanh(t);
  assert(is_close(res_tanh.data()[0], 0.0f));
  assert(is_close(res_tanh.data()[1], 0.7616f, 1e-4));
  assert(is_close(res_tanh.data()[2], -0.7616f, 1e-4));
  assert(is_close(res_tanh.data()[3], 1.0f, 1e-4));

  std::cout << "activation tests passed" << std::endl;
}
