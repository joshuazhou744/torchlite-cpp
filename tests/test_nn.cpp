#include <iostream>
#include <tl/tensor.h>
#include <tl/nn.h>
#include <tl/ops.h>
#include <tl/factory.h>
#include <cassert>
#include <cmath>

// helper function for float comparison
bool is_close_nn(float a, float b, float e = 1e-5) {
  return std::abs( a - b ) < e;
}

void test_nn() {
  // test Linear: shape check
  tl::nn::Linear linear(4, 3);
  tl::Tensor lin_in = tl::ones({2, 4});
  tl::Tensor lin_out = linear.forward(lin_in);
  assert(lin_out.sizes().size() == 2);
  assert(lin_out.sizes()[0] == 2);
  assert(lin_out.sizes()[1] == 3);

  // test Linear without bias
  tl::nn::Linear linear_nb(4, 3, false);
  tl::Tensor lin_nb_out = linear_nb.forward(lin_in);
  assert(lin_nb_out.sizes()[0] == 2);
  assert(lin_nb_out.sizes()[1] == 3);

  // test Linear: batched input [2, 5, 4] -> [2, 5, 3]
  tl::Tensor lin_batch = tl::ones({2, 5, 4});
  tl::Tensor lin_batch_out = linear.forward(lin_batch);
  assert(lin_batch_out.sizes().size() == 3);
  assert(lin_batch_out.sizes()[0] == 2);
  assert(lin_batch_out.sizes()[1] == 5);
  assert(lin_batch_out.sizes()[2] == 3);

  // test LayerNorm: output should have mean ~0 and variance ~1
  tl::nn::LayerNorm ln(4);
  tl::Tensor ln_in({2, 4});
  ln_in.data()[0] = 1.0f; ln_in.data()[1] = 2.0f;
  ln_in.data()[2] = 3.0f; ln_in.data()[3] = 4.0f;
  ln_in.data()[4] = 10.0f; ln_in.data()[5] = 20.0f;
  ln_in.data()[6] = 30.0f; ln_in.data()[7] = 40.0f;

  tl::Tensor ln_out = ln.forward(ln_in);
  assert(ln_out.sizes()[0] == 2);
  assert(ln_out.sizes()[1] == 4);

  // each row should have mean ~0 after normalization
  float row0_mean = (ln_out.data()[0] + ln_out.data()[1] + ln_out.data()[2] + ln_out.data()[3]) / 4.0f;
  float row1_mean = (ln_out.data()[4] + ln_out.data()[5] + ln_out.data()[6] + ln_out.data()[7]) / 4.0f;
  assert(is_close_nn(row0_mean, 0.0f, 1e-4));
  assert(is_close_nn(row1_mean, 0.0f, 1e-4));

  // each row should have variance ~1 after normalization
  float row0_var = 0.0f;
  for (int i = 0; i < 4; ++i) {
    row0_var += (ln_out.data()[i] - row0_mean) * (ln_out.data()[i] - row0_mean);
  }
  row0_var /= 4.0f;
  assert(is_close_nn(row0_var, 1.0f, 1e-4));

  std::cout << "nn tests passed" << std::endl;
}
