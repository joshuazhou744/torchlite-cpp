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

  // test MultiHeadAttention: shape check
  tl::nn::MultiHeadAttention msa(16, 4); // d_model=16, 4 heads of dim 4
  tl::Tensor msa_in = tl::randn({2, 5, 16}); // [batch=2, seq=5, d_model=16]
  tl::Tensor msa_out = msa.forward(msa_in);
  assert(msa_out.sizes().size() == 3);
  assert(msa_out.sizes()[0] == 2);  // batch preserved
  assert(msa_out.sizes()[1] == 5);  // seq preserved
  assert(msa_out.sizes()[2] == 16); // d_model preserved

  // test MSA with different config
  tl::nn::MultiHeadAttention msa2(32, 8); // d_model=32, 8 heads of dim 4
  tl::Tensor msa2_in = tl::randn({1, 10, 32});
  tl::Tensor msa2_out = msa2.forward(msa2_in);
  assert(msa2_out.sizes()[0] == 1);
  assert(msa2_out.sizes()[1] == 10);
  assert(msa2_out.sizes()[2] == 32);

  // test MSA output values are finite
  for (int i = 0; i < msa_out.numel(); ++i) {
    assert(std::isfinite(msa_out.data()[i]));
  }

  // test TransformerEncoderLayer: shape preserved
  tl::nn::TransformerEncoderLayer enc_layer(16, 4, 64); // d_model=16, 4 heads, d_ff=64
  tl::Tensor enc_in = tl::randn({2, 5, 16}); // [batch=2, seq=5, d_model=16]
  tl::Tensor enc_out = enc_layer.forward(enc_in);
  assert(enc_out.sizes().size() == 3);
  assert(enc_out.sizes()[0] == 2);
  assert(enc_out.sizes()[1] == 5);
  assert(enc_out.sizes()[2] == 16);

  // test output values are finite
  for (int i = 0; i < enc_out.numel(); ++i) {
    assert(std::isfinite(enc_out.data()[i]));
  }

  // test TransformerEncoder: shape preserved through multiple layers
  tl::nn::TransformerEncoder encoder(16, 4, 64, 3); // d_model=16, 4 heads, d_ff=64, 3 layers
  tl::Tensor te_in = tl::randn({2, 5, 16});
  tl::Tensor te_out = encoder.forward(te_in);
  assert(te_out.sizes().size() == 3);
  assert(te_out.sizes()[0] == 2);
  assert(te_out.sizes()[1] == 5);
  assert(te_out.sizes()[2] == 16);

  // test output values are finite
  for (int i = 0; i < te_out.numel(); ++i) {
    assert(std::isfinite(te_out.data()[i]));
  }

  // test PositionalEncoding: shape preserved
  tl::nn::PositionalEncoding pe(16);
  tl::Tensor pe_in = tl::randn({2, 5, 16}); // [batch=2, seq=5, d_model=16]
  tl::Tensor pe_out = pe.forward(pe_in);
  assert(pe_out.sizes().size() == 3);
  assert(pe_out.sizes()[0] == 2);
  assert(pe_out.sizes()[1] == 5);
  assert(pe_out.sizes()[2] == 16);

  // test output differs from input (encoding was added)
  bool any_diff = false;
  for (int i = 0; i < pe_in.numel(); ++i) {
    if (!is_close_nn(pe_in.data()[i], pe_out.data()[i])) {
      any_diff = true;
      break;
    }
  }
  assert(any_diff);

  // test same position gets same encoding regardless of input content
  tl::Tensor pe_in2 = tl::randn({1, 5, 16}); // different content, same seq length
  tl::Tensor pe_out2 = pe.forward(pe_in2);
  tl::Tensor pe_zeros = tl::zeros({1, 5, 16});
  tl::Tensor pe_out3 = pe.forward(pe_zeros);
  // pe_out3 should equal pure positional encoding since input was zeros
  // pe_out2 - pe_in2 should equal pe_out3 - zeros = pe_out3
  for (int i = 0; i < pe_out3.numel(); ++i) {
    float added_to_in2 = pe_out2.data()[i] - pe_in2.data()[i];
    assert(is_close_nn(added_to_in2, pe_out3.data()[i]));
  }

  std::cout << "nn tests passed" << std::endl;
}
