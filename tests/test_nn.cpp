#include <iostream>
#include <tl/tensor.h>
#include <tl/nn.h>
#include <tl/ops.h>
#include <tl/factory.h>
#include <cassert>
#include <cmath>
#include <vector>
#include "test_utils.h" // CHECK(): NDEBUG-proof, unlike assert()


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
  assert(is_close(row0_mean, 0.0f, 1e-4));
  assert(is_close(row1_mean, 0.0f, 1e-4));

  // each row should have variance ~1 after normalization
  float row0_var = 0.0f;
  for (int i = 0; i < 4; ++i) {
    row0_var += (ln_out.data()[i] - row0_mean) * (ln_out.data()[i] - row0_mean);
  }
  row0_var /= 4.0f;
  assert(is_close(row0_var, 1.0f, 1e-4));

  // test BatchNorm2d: each channel output should have mean ~0 and variance ~1
  // input (N=2, C=3, H=2, W=2): fill each channel with a distinct scale to ensure
  // channels normalize independently (channel 0 in [1..8], channel 1 in [10..80],
  // channel 2 in [100..800]). post-norm, all channels collapse to mean 0, var 1.
  tl::nn::BatchNorm2d bn(3);
  tl::Tensor bn_in({2, 3, 2, 2});
  for (int64_t n = 0; n < 2; ++n) {
    for (int64_t c = 0; c < 3; ++c) {
      for (int64_t h = 0; h < 2; ++h) {
        for (int64_t w = 0; w < 2; ++w) {
          int64_t idx = n*12 + c*4 + h*2 + w;     // flat (N,C,H,W) index
          int64_t pos = n*4 + h*2 + w;            // 0..7 position within this channel
          float scale = (c == 0) ? 1.0f : (c == 1 ? 10.0f : 100.0f);
          bn_in.data()[idx] = scale * (pos + 1);
        }
      }
    }
  }

  tl::Tensor bn_out = bn.forward(bn_in);
  assert(bn_out.sizes().size() == 4);
  assert(bn_out.sizes()[0] == 2);
  assert(bn_out.sizes()[1] == 3);
  assert(bn_out.sizes()[2] == 2);
  assert(bn_out.sizes()[3] == 2);

  // per-channel statistics check (reduce over N, H, W; keep channel separate)
  for (int64_t c = 0; c < 3; ++c) {
    float ch_vals[8];
    float ch_mean = 0.0f;
    int64_t k = 0;
    for (int64_t n = 0; n < 2; ++n) {
      for (int64_t h = 0; h < 2; ++h) {
        for (int64_t w = 0; w < 2; ++w) {
          int64_t idx = n*12 + c*4 + h*2 + w;
          ch_vals[k] = bn_out.data()[idx];
          ch_mean += ch_vals[k];
          ++k;
        }
      }
    }
    ch_mean /= 8.0f;
    assert(is_close(ch_mean, 0.0f, 1e-4));

    float ch_var = 0.0f;
    for (int i = 0; i < 8; ++i) ch_var += (ch_vals[i] - ch_mean) * (ch_vals[i] - ch_mean);
    ch_var /= 8.0f;
    assert(is_close(ch_var, 1.0f, 1e-3)); // small slack for the eps term inside BN
  }

  // parameter count: gamma and beta, each shape [C]
  auto bn_params = bn.parameters();
  assert(bn_params.size() == 2);
  assert(bn_params[0]->numel() == 3);
  assert(bn_params[1]->numel() == 3);

  // test GroupNorm: 4 channels, 2 groups -> normalize over pairs of channels
  // input [N=2, C=4, H=2, W=2]: fill group 0 (ch 0-1) with scale 1, group 1 (ch 2-3) with scale 100
  // post-norm each group should have mean ~0 and variance ~1
  {
    tl::nn::GroupNorm gn(2, 4); // 2 groups, 4 channels
    tl::Tensor gn_in({2, 4, 2, 2});
    for (int64_t n = 0; n < 2; ++n) {
      for (int64_t c = 0; c < 4; ++c) {
        float scale = (c < 2) ? 1.0f : 100.0f;
        for (int64_t h = 0; h < 2; ++h) {
          for (int64_t w = 0; w < 2; ++w) {
            int64_t idx = n*16 + c*4 + h*2 + w;
            gn_in.data()[idx] = scale * (float)(h*2 + w + 1); // 1,2,3,4 scaled
          }
        }
      }
    }
    tl::Tensor gn_out = gn.forward(gn_in);
    assert(gn_out.sizes().size() == 4);
    assert(gn_out.sizes()[0] == 2);
    assert(gn_out.sizes()[1] == 4);
    assert(gn_out.sizes()[2] == 2);
    assert(gn_out.sizes()[3] == 2);

    // each group per batch item should have mean ~0
    for (int64_t n = 0; n < 2; ++n) {
      for (int64_t g = 0; g < 2; ++g) {
        float group_mean = 0.0f;
        for (int64_t c = g*2; c < g*2+2; ++c)
          for (int64_t h = 0; h < 2; ++h)
            for (int64_t w = 0; w < 2; ++w)
              group_mean += gn_out.data()[n*16 + c*4 + h*2 + w];
        group_mean /= 8.0f; // 2 channels * 2H * 2W
        assert(is_close(group_mean, 0.0f, 1e-4));
      }
    }

    // parameter count: gamma and beta each [C=4]
    auto gn_params = gn.parameters();
    assert(gn_params.size() == 2);
    assert(gn_params[0]->numel() == 4);
    assert(gn_params[1]->numel() == 4);
  }

  // test AdaptiveGroupNorm: output shape preserved, different cond -> different output
  {
    tl::nn::AdaptiveGroupNorm agn(2, 4, 8); // 2 groups, 4 channels, cond_dim=8

    tl::Tensor agn_in  = tl::randn({2, 4, 4, 4}); // [N=2, C=4, H=4, W=4]
    tl::Tensor cond1   = tl::randn({2, 8});         // conditioning vector
    tl::Tensor cond2   = tl::randn({2, 8});         // different conditioning

    tl::Tensor out1 = agn.forward(agn_in, cond1);
    tl::Tensor out2 = agn.forward(agn_in, cond2);

    // output shape matches input
    assert(out1.sizes().size() == 4);
    assert(out1.sizes()[0] == 2);
    assert(out1.sizes()[1] == 4);
    assert(out1.sizes()[2] == 4);
    assert(out1.sizes()[3] == 4);

    // all values finite
    for (int i = 0; i < out1.numel(); ++i)
      assert(std::isfinite(out1.data()[i]));

    // different cond -> different output (conditioning actually changes output)
    bool any_diff = false;
    for (int i = 0; i < out1.numel(); ++i) {
      if (!is_close(out1.data()[i], out2.data()[i])) { any_diff = true; break; }
    }
    assert(any_diff);

    // only proj_ params exposed: weight [8->8] + bias [8] = 2 tensors
    auto agn_params = agn.parameters();
    assert(agn_params.size() == 2); // proj_ weight + bias
  }

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

  // test MSA cross-attention overload: query and context have different seq lengths
  // output shape must be [batch, tgt_seq, d_model] — matches query, not context
  {
    tl::nn::MultiHeadAttention msa_cross(16, 4);
    tl::Tensor xq = tl::randn({2, 5, 16});   // [batch=2, tgt_seq=5, d_model=16]
    tl::Tensor xc = tl::randn({2, 8, 16});   // [batch=2, src_seq=8, d_model=16]
    tl::Tensor cross_out = msa_cross.forward(xq, xc, tl::Tensor());
    assert(cross_out.sizes().size() == 3);
    assert(cross_out.sizes()[0] == 2);   // batch preserved
    assert(cross_out.sizes()[1] == 5);   // tgt_seq from query, not context
    assert(cross_out.sizes()[2] == 16);  // d_model preserved
    for (int i = 0; i < cross_out.numel(); ++i)
      assert(std::isfinite(cross_out.data()[i]));
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

  // test TransformerDecoderLayer: output shape matches [batch, tgt_seq, d_model]
  // src_seq (8) differs from tgt_seq (5) to confirm cross-attention handles mismatched lengths
  {
    tl::nn::TransformerDecoderLayer dec_layer(16, 4, 64);
    tl::Tensor dec_in  = tl::randn({2, 5, 16});  // [batch=2, tgt_seq=5, d_model=16]
    tl::Tensor enc_out = tl::randn({2, 8, 16});  // [batch=2, src_seq=8, d_model=16]
    tl::Tensor dec_out = dec_layer.forward(dec_in, enc_out);
    assert(dec_out.sizes().size() == 3);
    assert(dec_out.sizes()[0] == 2);
    assert(dec_out.sizes()[1] == 5);   // tgt_seq preserved
    assert(dec_out.sizes()[2] == 16);
    for (int i = 0; i < dec_out.numel(); ++i)
      assert(std::isfinite(dec_out.data()[i]));
  }

  // test TransformerDecoder: shape preserved through multiple layers, mismatched seq lengths
  {
    tl::nn::TransformerDecoder decoder(16, 4, 64, 3); // d_model=16, 4 heads, d_ff=64, 3 layers
    tl::Tensor td_in  = tl::randn({2, 5, 16});  // [batch=2, tgt_seq=5, d_model=16]
    tl::Tensor td_enc = tl::randn({2, 8, 16});  // [batch=2, src_seq=8, d_model=16]
    tl::Tensor td_out = decoder.forward(td_in, td_enc);
    assert(td_out.sizes().size() == 3);
    assert(td_out.sizes()[0] == 2);
    assert(td_out.sizes()[1] == 5);   // tgt_seq preserved through all layers
    assert(td_out.sizes()[2] == 16);
    for (int i = 0; i < td_out.numel(); ++i)
      assert(std::isfinite(td_out.data()[i]));
  }

  // test CausalTransformerLayer: shape preserved, no mask
  {
    tl::nn::CausalTransformerLayer causal_layer(16, 4, 64);
    tl::Tensor cl_in  = tl::randn({2, 5, 16});
    tl::Tensor cl_out = causal_layer.forward(cl_in);
    assert(cl_out.sizes().size() == 3);
    assert(cl_out.sizes()[0] == 2);
    assert(cl_out.sizes()[1] == 5);
    assert(cl_out.sizes()[2] == 16);
    for (int i = 0; i < cl_out.numel(); ++i)
      assert(std::isfinite(cl_out.data()[i]));
  }

  // test CausalTransformer: shape preserved through multiple layers
  {
    tl::nn::CausalTransformer causal(16, 4, 64, 3);
    tl::Tensor ct_in  = tl::randn({2, 5, 16});
    tl::Tensor ct_out = causal.forward(ct_in);
    assert(ct_out.sizes().size() == 3);
    assert(ct_out.sizes()[0] == 2);
    assert(ct_out.sizes()[1] == 5);
    assert(ct_out.sizes()[2] == 16);
    for (int i = 0; i < ct_out.numel(); ++i)
      assert(std::isfinite(ct_out.data()[i]));
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
    if (!is_close(pe_in.data()[i], pe_out.data()[i])) {
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
    assert(is_close(added_to_in2, pe_out3.data()[i]));
  }

  // test MaxPool2d: kernel=2, stride=2 over (1,1,4,4) input filled 1..16
  tl::nn::MaxPool2d maxpool(2, 2);
  tl::Tensor pool_in({1, 1, 4, 4});
  for (int64_t i = 0; i < 16; ++i) pool_in.data()[i] = static_cast<float>(i + 1);
  tl::Tensor maxpool_out = maxpool.forward(pool_in);
  assert(maxpool_out.sizes().size() == 4);
  assert(maxpool_out.sizes()[0] == 1);
  assert(maxpool_out.sizes()[1] == 1);
  assert(maxpool_out.sizes()[2] == 2);
  assert(maxpool_out.sizes()[3] == 2);
  // each 2x2 window's max: top-left=6, top-right=8, bottom-left=14, bottom-right=16
  assert(is_close(maxpool_out.data()[0], 6.0f));
  assert(is_close(maxpool_out.data()[1], 8.0f));
  assert(is_close(maxpool_out.data()[2], 14.0f));
  assert(is_close(maxpool_out.data()[3], 16.0f));

  // test AvgPool2d: same input and config
  tl::nn::AvgPool2d avgpool(2, 2);
  tl::Tensor avgpool_out = avgpool.forward(pool_in);
  assert(avgpool_out.sizes().size() == 4);
  assert(avgpool_out.sizes()[0] == 1);
  assert(avgpool_out.sizes()[1] == 1);
  assert(avgpool_out.sizes()[2] == 2);
  assert(avgpool_out.sizes()[3] == 2);
  // each 2x2 window's average
  assert(is_close(avgpool_out.data()[0], 3.5f));
  assert(is_close(avgpool_out.data()[1], 5.5f));
  assert(is_close(avgpool_out.data()[2], 11.5f));
  assert(is_close(avgpool_out.data()[3], 13.5f));

  // test default stride (stride == kernel_size when omitted)
  tl::nn::MaxPool2d maxpool_def(2);
  tl::Tensor maxpool_def_out = maxpool_def.forward(pool_in);
  assert(maxpool_def_out.sizes()[2] == 2);
  assert(maxpool_def_out.sizes()[3] == 2);
  assert(is_close(maxpool_def_out.data()[0], 6.0f));
  assert(is_close(maxpool_def_out.data()[3], 16.0f));

  // test Upsample: scale_factor=2 on (1,1,2,2) input [[1,2],[3,4]]
  // each pixel repeated 2x in H and W -> (1,1,4,4)
  {
    tl::nn::Upsample up(2);
    tl::Tensor up_in({1, 1, 2, 2});
    up_in.data()[0] = 1.0f; up_in.data()[1] = 2.0f;
    up_in.data()[2] = 3.0f; up_in.data()[3] = 4.0f;
    tl::Tensor up_out = up.forward(up_in);
    assert(up_out.sizes().size() == 4);
    assert(up_out.sizes()[0] == 1);
    assert(up_out.sizes()[1] == 1);
    assert(up_out.sizes()[2] == 4);
    assert(up_out.sizes()[3] == 4);
    // top-left 2x2 block all == 1, top-right 2x2 block all == 2
    assert(is_close(up_out.data()[0],  1.0f));  // row0, col0
    assert(is_close(up_out.data()[1],  1.0f));  // row0, col1
    assert(is_close(up_out.data()[2],  2.0f));  // row0, col2
    assert(is_close(up_out.data()[3],  2.0f));  // row0, col3
    assert(is_close(up_out.data()[4],  1.0f));  // row1, col0
    assert(is_close(up_out.data()[8],  3.0f));  // row2, col0
    assert(is_close(up_out.data()[15], 4.0f));  // row3, col3
  }

  // test Downsample: halves H and W via strided conv
  {
    tl::nn::Downsample ds(4); // 4 channels

    tl::Tensor x = tl::randn({2, 4, 16, 16});
    tl::Tensor out = ds.forward(x);

    // spatial dims halved
    assert(out.sizes()[0] == 2);
    assert(out.sizes()[1] == 4);
    assert(out.sizes()[2] == 8);
    assert(out.sizes()[3] == 8);

    // channels unchanged
    assert(out.sizes()[1] == x.sizes()[1]);

    // has parameters (conv weight + bias)
    auto params = ds.parameters();
    assert(!params.empty());

    for (int i = 0; i < out.numel(); ++i)
      assert(std::isfinite(out.data()[i]));
  }

  // test Upsample with in_channels > 0: output shape correct, has parameters
  {
    tl::nn::Upsample up(2, 4); // scale=2, in_channels=4 -> conv applied after
    tl::Tensor up_in = tl::randn({2, 4, 8, 8});
    tl::Tensor up_out = up.forward(up_in);
    // spatial dims doubled
    assert(up_out.sizes()[0] == 2);
    assert(up_out.sizes()[1] == 4);
    assert(up_out.sizes()[2] == 16);
    assert(up_out.sizes()[3] == 16);
    // has conv parameters
    auto params = up.parameters();
    assert(!params.empty());
    for (int i = 0; i < up_out.numel(); ++i)
      assert(std::isfinite(up_out.data()[i]));
  }

  // test Checkpoint: gradient checkpointing must reproduce the plain block's
  // gradients exactly (same input-grads AND same weight-grads). The wrapped run
  // builds no graph in forward and recomputes the block in backward, so matching
  // the plain run proves the recompute is faithful.
  {
    // one shared block instance -> identical weights for both runs
    tl::nn::Linear ckpt_lin(4, 3);
    tl::nn::ReLU ckpt_relu;
    tl::nn::Sequential ckpt_block({&ckpt_lin, &ckpt_relu});
    std::vector<tl::Tensor*> ckpt_params = ckpt_block.parameters(); // weight, bias

    // two inputs with identical values, both requiring grad
    tl::Tensor ckpt_x1 = tl::randn({2, 4});
    tl::Tensor ckpt_x2({2, 4});
    for (int64_t i = 0; i < ckpt_x1.numel(); ++i) ckpt_x2.data()[i] = ckpt_x1.data()[i];
    ckpt_x1.set_requires_grad(true);
    ckpt_x2.set_requires_grad(true);

    // --- plain run (builds the full graph) ---
    tl::Tensor ckpt_y1 = ckpt_block.forward(ckpt_x1);
    ckpt_y1.backward(); // seeds dY = ones

    // capture plain grads
    std::vector<float> plain_xgrad(ckpt_x1.grad().data(),
                                   ckpt_x1.grad().data() + ckpt_x1.grad().numel());
    std::vector<std::vector<float>> plain_pgrad;
    for (tl::Tensor* p : ckpt_params)
      plain_pgrad.emplace_back(p->grad().data(), p->grad().data() + p->grad().numel());

    // zero all grads before the wrapped run
    ckpt_x1.grad() = tl::Tensor();
    for (tl::Tensor* p : ckpt_params) p->grad() = tl::Tensor();

    // --- wrapped run (silent forward + recompute in backward) ---
    tl::nn::Checkpoint ckpt(&ckpt_block);
    tl::Tensor ckpt_y2 = ckpt.forward(ckpt_x2);
    ckpt_y2.backward(); // same seed: dY = ones

    // forward outputs must match (silent forward == normal forward)
    assert(ckpt_y2.numel() == ckpt_y1.numel());
    for (int64_t i = 0; i < ckpt_y1.numel(); ++i)
      assert(is_close(ckpt_y2.data()[i], ckpt_y1.data()[i]));

    // input-grads must match
    assert(ckpt_x2.grad().numel() == (int64_t)plain_xgrad.size());
    for (int64_t i = 0; i < ckpt_x2.grad().numel(); ++i)
      assert(is_close(ckpt_x2.grad().data()[i], plain_xgrad[i]));

    // weight-grads must match (recompute produced the block's param grads)
    for (size_t k = 0; k < ckpt_params.size(); ++k) {
      assert(ckpt_params[k]->grad().numel() == (int64_t)plain_pgrad[k].size());
      for (int64_t i = 0; i < ckpt_params[k]->grad().numel(); ++i)
        assert(is_close(ckpt_params[k]->grad().data()[i], plain_pgrad[k][i]));
    }
  }

  // test Checkpoint: explicit double-count regression.
  // With an identity-weight Linear, y == x, so seeding dY = ones gives an
  // analytic input gradient of EXACTLY 1.0 per element (dX = ones @ W^T = ones).
  // The old aliasing bug made CheckpointBackward accumulate the input gradient
  // twice (dX = 2.0), so asserting the exact value 1.0 fails on the buggy code
  // and passes only when the recompute uses independent grad storage.
  {
    tl::nn::Linear id_lin(3, 3, false); // no bias
    tl::Tensor id_w = tl::zeros({3, 3});
    id_w.data()[0] = 1.0f; id_w.data()[4] = 1.0f; id_w.data()[8] = 1.0f; // identity
    id_lin.set_weight(id_w);

    tl::nn::Checkpoint id_ckpt(&id_lin);

    tl::Tensor id_x({1, 3});
    id_x.data()[0] = 1.0f; id_x.data()[1] = 2.0f; id_x.data()[2] = 3.0f;
    id_x.set_requires_grad(true);

    tl::Tensor id_y = id_ckpt.forward(id_x);
    // identity weight: output equals input
    CHECK(is_close(id_y.data()[0], 1.0f));
    CHECK(is_close(id_y.data()[1], 2.0f));
    CHECK(is_close(id_y.data()[2], 3.0f));

    id_y.backward(); // dY = ones -> dX = ones @ I = ones
    // exact analytic input grad is 1.0; the double-count bug yields 2.0
    CHECK(id_x.grad().numel() == 3);
    for (int64_t i = 0; i < id_x.grad().numel(); ++i)
      CHECK(is_close(id_x.grad().data()[i], 1.0f));
  }

  // test TimestepEmbedding: output shape and finiteness
  // dim=64 sinusoidal features, out_dim=128 conditioning vector
  {
    tl::nn::TimestepEmbedding te(64, 128);

    // batch of 3 noise levels
    tl::Tensor sigma({3});
    sigma.data()[0] = 0.1f;
    sigma.data()[1] = 1.0f;
    sigma.data()[2] = 10.0f;

    tl::Tensor te_out = te.forward(sigma);

    // output shape: [N, out_dim]
    assert(te_out.sizes().size() == 2);
    assert(te_out.sizes()[0] == 3);
    assert(te_out.sizes()[1] == 128);

    // all values finite
    for (int i = 0; i < te_out.numel(); ++i)
      assert(std::isfinite(te_out.data()[i]));

    // different sigma -> different embedding (not all identical)
    bool any_diff = false;
    for (int64_t i = 0; i < 128; ++i) {
      if (!is_close(te_out.data()[0 * 128 + i], te_out.data()[1 * 128 + i])) {
        any_diff = true;
        break;
      }
    }
    assert(any_diff);

    // parameter count: fc1 + fc2, each has weight + bias
    auto te_params = te.parameters();
    assert(te_params.size() == 4); // fc1 weight, fc1 bias, fc2 weight, fc2 bias
  }

  // test FourierFeatures: output shape, no parameters, same sigma -> same output
  {
    tl::nn::FourierFeatures ff(256);

    // no learned parameters
    assert(ff.parameters().empty());

    // output shape: [N, cond_dim]
    tl::Tensor sigma({3});
    sigma.data()[0] = 0.1f; sigma.data()[1] = 0.5f; sigma.data()[2] = 2.0f;
    tl::Tensor out = ff.forward(sigma);
    assert(out.sizes().size() == 2);
    assert(out.sizes()[0] == 3);
    assert(out.sizes()[1] == 256);

    // all values finite and in [-1, 1] since they are cos/sin
    for (int i = 0; i < out.numel(); ++i) {
      assert(std::isfinite(out.data()[i]));
      assert(out.data()[i] >= -1.0f && out.data()[i] <= 1.0f);
    }

    // same input -> same output (deterministic after init)
    tl::Tensor out2 = ff.forward(sigma);
    for (int i = 0; i < out.numel(); ++i)
      assert(is_close(out.data()[i], out2.data()[i]));

    // different sigma -> different output
    tl::Tensor sigma2({3});
    sigma2.data()[0] = 0.9f; sigma2.data()[1] = 1.5f; sigma2.data()[2] = 3.0f;
    tl::Tensor out3 = ff.forward(sigma2);
    bool any_diff = false;
    for (int i = 0; i < out.numel(); ++i)
      if (!is_close(out.data()[i], out3.data()[i])) { any_diff = true; break; }
    assert(any_diff);
  }

  // test Embedding: 1D input [N] -> [N, embedding_dim]
  {
    tl::nn::Embedding emb(10, 4); // 10 tokens, embedding dim 4

    tl::Tensor idx({3});
    idx.data()[0] = 0.0f; idx.data()[1] = 5.0f; idx.data()[2] = 9.0f;

    tl::Tensor out = emb.forward(idx);
    assert(out.sizes().size() == 2);
    assert(out.sizes()[0] == 3);
    assert(out.sizes()[1] == 4);

    // same index -> same embedding row
    tl::Tensor idx2({2});
    idx2.data()[0] = 5.0f; idx2.data()[1] = 5.0f;
    tl::Tensor out2 = emb.forward(idx2);
    for (int d = 0; d < 4; ++d)
      assert(is_close(out2.data()[0 * 4 + d], out2.data()[1 * 4 + d]));

    // 2D input [N, T] -> [N, T, embedding_dim]
    tl::Tensor idx3({2, 3});
    for (int i = 0; i < 6; ++i) idx3.data()[i] = (float)(i % 10);
    tl::Tensor out3 = emb.forward(idx3);
    assert(out3.sizes().size() == 3);
    assert(out3.sizes()[0] == 2);
    assert(out3.sizes()[1] == 3);
    assert(out3.sizes()[2] == 4);

    // one learned parameter (weight)
    assert(emb.parameters().size() == 1);
    assert(emb.parameters()[0]->numel() == 10 * 4);
  }

  std::cout << "nn tests passed" << std::endl;
}
