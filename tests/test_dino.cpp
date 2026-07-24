#include <iostream>
#include <tl/tensor.h>
#include <tl/dino.h>
#include <tl/ops.h>
#include <tl/factory.h>
#include <cassert>
#include <cmath>
#include "test_utils.h" // CHECK(): NDEBUG-proof, unlike assert()

void test_dino() {
  // test DinoAttention
  {
    // --- shape + finiteness, two configs ---
    // config A: dim=16, 4 heads (head_dim=4), 2x3 patch grid, prefix=2 -> T=8
    tl::nn::DinoAttention attn(16, 4);
    auto [cosA, sinA] = tl::dino_rope_cos_sin_2d(2, 3, 4); // [6, 4] (h*w patch tokens, head_dim)
    tl::Tensor xA = tl::randn({2, 8, 16});                 // [N=2, T=2+6, dim=16]
    tl::Tensor outA = attn.forward(xA, cosA, sinA, 2);
    CHECK(outA.sizes().size() == 3);
    CHECK(outA.sizes()[0] == 2);   // batch preserved
    CHECK(outA.sizes()[1] == 8);   // token count preserved
    CHECK(outA.sizes()[2] == 16);  // dim preserved
    for (int i = 0; i < outA.numel(); ++i) CHECK(std::isfinite(outA.data()[i]));

    // config B: dim=32, 8 heads (head_dim=4), 3x2 grid, prefix=1 -> T=7
    tl::nn::DinoAttention attnB(32, 8);
    auto [cosB, sinB] = tl::dino_rope_cos_sin_2d(3, 2, 4); // [6, 4]
    tl::Tensor xB = tl::randn({1, 7, 32});
    tl::Tensor outB = attnB.forward(xB, cosB, sinB, 1);
    CHECK(outB.sizes()[0] == 1);
    CHECK(outB.sizes()[1] == 7);
    CHECK(outB.sizes()[2] == 32);
    for (int i = 0; i < outB.numel(); ++i) CHECK(std::isfinite(outB.data()[i]));

    // --- prefix-invariance under an identity RoPE table ---
    // cos=1, sin=0 makes apply_rotary_half a no-op, so rotation does nothing for
    // ANY prefix. Same module + input must give identical output whether prefix
    // skips 0 or 3 leading tokens. Catches wrong cat dim / off-by-one slice bounds:
    // if the prefix split/rejoin were wrong, the two runs would diverge.
    tl::nn::DinoAttention attnC(16, 4);
    tl::Tensor xC = tl::randn({2, 8, 16});
    // prefix=0: table spans all T=8 tokens
    tl::Tensor id_cos0 = tl::ones({8, 4}), id_sin0 = tl::zeros({8, 4});
    tl::Tensor out_p0 = attnC.forward(xC, id_cos0, id_sin0, 0);
    // prefix=3: table spans the T-3=5 patch tokens
    tl::Tensor id_cos3 = tl::ones({5, 4}), id_sin3 = tl::zeros({5, 4});
    tl::Tensor out_p3 = attnC.forward(xC, id_cos3, id_sin3, 3);
    for (int i = 0; i < out_p0.numel(); ++i)
      CHECK(is_close(out_p0.data()[i], out_p3.data()[i], 1e-5f));

    // --- a real RoPE table actually changes the result ---
    // identity table (no-op) vs the real dino table, same module/input/prefix.
    // outputs must differ -> confirms RoPE is genuinely applied to patch tokens.
    tl::Tensor id_cos = tl::ones({6, 4}), id_sin = tl::zeros({6, 4});
    tl::Tensor out_id   = attnC.forward(xC, id_cos, id_sin, 2);
    tl::Tensor out_rope = attnC.forward(xC, cosA, sinA, 2); // cosA/sinA: real [6,4] table
    bool differs = false;
    for (int i = 0; i < out_id.numel(); ++i)
      if (!is_close(out_id.data()[i], out_rope.data()[i], 1e-5f)) { differs = true; break; }
    CHECK(differs);

    // --- parameter count: qkv (weight+bias) + proj (weight+bias) = 4 ---
    CHECK(attn.parameters().size() == 4);

    // --- dim not divisible by num_heads must throw ---
    bool threw = false;
    try {
      tl::nn::DinoAttention bad(16, 3);
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    CHECK(threw);
  }

  std::cout << "dino tests passed" << std::endl;
}
