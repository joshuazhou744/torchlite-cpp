#include <iostream>
#include <tl/tensor.h>
#include <tl/dino.h>
#include <tl/ops.h>
#include <tl/factory.h>
#include <cassert>
#include <cmath>
#include <vector>
#include "test_utils.h" // CHECK(): NDEBUG-proof, unlike assert()

void test_dino() {
  // test DinoAttention
  {
    // --- shape + finiteness, two configs ---
    // config A: dim=16, 4 heads (head_dim=4), 2x3 patch grid, prefix=2 -> T=8
    tl::dino::DinoAttention attn(16, 4);
    auto [cosA, sinA] = tl::dino_rope_cos_sin_2d(2, 3, 4); // [6, 4] (h*w patch tokens, head_dim)
    tl::Tensor xA = tl::randn({2, 8, 16});                 // [N=2, T=2+6, dim=16]
    tl::Tensor outA = attn.forward(xA, cosA, sinA, 2);
    CHECK(outA.sizes().size() == 3);
    CHECK(outA.sizes()[0] == 2);   // batch preserved
    CHECK(outA.sizes()[1] == 8);   // token count preserved
    CHECK(outA.sizes()[2] == 16);  // dim preserved
    for (int i = 0; i < outA.numel(); ++i) CHECK(std::isfinite(outA.data()[i]));

    // config B: dim=32, 8 heads (head_dim=4), 3x2 grid, prefix=1 -> T=7
    tl::dino::DinoAttention attnB(32, 8);
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
    tl::dino::DinoAttention attnC(16, 4);
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
      tl::dino::DinoAttention bad(16, 3);
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    CHECK(threw);
  }

  // test DinoBlock
  {
    // dim=16, 4 heads (head_dim=4), mlp_hidden=64, 2x3 grid, prefix=2 -> T=8
    tl::dino::DinoBlock blk(16, 4, 64);
    auto [cos, sin] = tl::dino_rope_cos_sin_2d(2, 3, 4); // [6, 4]
    tl::Tensor x = tl::randn({2, 8, 16});

    // --- shape preserved + finite ---
    tl::Tensor out = blk.forward(x, cos, sin, 2);
    CHECK(out.sizes().size() == 3);
    CHECK(out.sizes()[0] == 2);
    CHECK(out.sizes()[1] == 8);
    CHECK(out.sizes()[2] == 16);
    for (int i = 0; i < out.numel(); ++i) CHECK(std::isfinite(out.data()[i]));

    // --- parameter count and ordering ---
    // norm1(2) + attn(4) + ls1(1) + norm2(2) + fc1(2) + fc2(2) + ls2(1) = 14
    // ordering matters: the checkpoint loader maps weights by this index order.
    auto bp = blk.parameters();
    CHECK(bp.size() == 14);
    tl::Tensor* ls1 = bp[6];
    tl::Tensor* ls2 = bp[13];
    // LayerScale is the only param initialized to 1e-5, so this confirms we
    // grabbed the right slots (and that gamma init didn't drift).
    CHECK(ls1->sizes().size() == 1 && ls1->sizes()[0] == 16);
    CHECK(ls2->sizes().size() == 1 && ls2->sizes()[0] == 16);
    CHECK(is_close(ls1->data()[0], 1e-5f, 1e-9f));
    CHECK(is_close(ls2->data()[0], 1e-5f, 1e-9f));

    // --- at init the block is near-identity (CaiT LayerScale) ---
    // both branches are scaled by 1e-5, so out ~= x. this catches a missing
    // residual add (out would be ~0) or LayerScale applied to the wrong side.
    for (int i = 0; i < out.numel(); ++i)
      CHECK(is_close(out.data()[i], x.data()[i], 1e-3f));

    // --- ls = 0 makes the block exactly the identity ---
    // sharper version of the above: with both gammas zeroed, every branch
    // contributes exactly 0 and the residual must pass x through untouched.
    for (int i = 0; i < ls1->numel(); ++i) ls1->data()[i] = 0.0f;
    for (int i = 0; i < ls2->numel(); ++i) ls2->data()[i] = 0.0f;
    tl::Tensor out_zero = blk.forward(x, cos, sin, 2);
    for (int i = 0; i < out_zero.numel(); ++i)
      CHECK(is_close(out_zero.data()[i], x.data()[i], 1e-6f));

    // --- ls = 1 makes the branches actually contribute ---
    // guards against the degenerate case where the block is identity for the
    // wrong reason (e.g. branches silently producing zeros).
    for (int i = 0; i < ls1->numel(); ++i) ls1->data()[i] = 1.0f;
    for (int i = 0; i < ls2->numel(); ++i) ls2->data()[i] = 1.0f;
    tl::Tensor out_one = blk.forward(x, cos, sin, 2);
    bool differs = false;
    for (int i = 0; i < out_one.numel(); ++i)
      if (!is_close(out_one.data()[i], x.data()[i], 1e-3f)) { differs = true; break; }
    CHECK(differs);
    for (int i = 0; i < out_one.numel(); ++i) CHECK(std::isfinite(out_one.data()[i]));
  }

  // test DinoViT
  {
    // dim=16, depth=3, 4 heads (head_dim=4), patch=4, 2 storage tokens
    // input [2, 3, 8, 12] -> 2x3 patch grid = 6 patch tokens, prefix=3, T=9
    tl::dino::DinoViT vit(16, 3, 4, 4, 2);
    tl::Tensor x = tl::randn({2, 3, 8, 12});

    // --- one feature map per requested layer, each [N, dim, h, w] ---
    auto feats = vit.forward(x, {0, 1, 2});
    CHECK(feats.size() == 3);
    for (auto& f: feats) {
      CHECK(f.sizes().size() == 4);
      CHECK(f.sizes()[0] == 2);  // batch preserved
      CHECK(f.sizes()[1] == 16); // dim -> channels
      CHECK(f.sizes()[2] == 2);  // H / patch
      CHECK(f.sizes()[3] == 3);  // W / patch
      for (int i = 0; i < f.numel(); ++i) CHECK(std::isfinite(f.data()[i]));
    }

    // --- results correspond positionally to `layers`, not to block order ---
    // request {2, 0} reversed: slot 0 must be block 2's snapshot, slot 1 block 0's.
    // compared against single-layer calls, so this is exact regardless of magnitudes.
    auto only0 = vit.forward(x, {0});
    auto only2 = vit.forward(x, {2});
    auto rev = vit.forward(x, {2, 0});
    CHECK(rev.size() == 2);
    for (int i = 0; i < rev[0].numel(); ++i)
      CHECK(is_close(rev[0].data()[i], only2[0].data()[i], 1e-6f));
    for (int i = 0; i < rev[1].numel(); ++i)
      CHECK(is_close(rev[1].data()[i], only0[0].data()[i], 1e-6f));

    // --- duplicate indices are allowed and produce identical maps ---
    auto dup = vit.forward(x, {1, 1});
    CHECK(dup.size() == 2);
    for (int i = 0; i < dup[0].numel(); ++i)
      CHECK(is_close(dup[0].data()[i], dup[1].data()[i], 1e-9f));

    // --- snapshots at different depths are actually different ---
    // tolerance is tight: at init LayerScale is 1e-5, so each block only nudges
    // the residual stream. this confirms blocks run between snapshots at all.
    bool depth_differs = false;
    for (int i = 0; i < only0[0].numel(); ++i)
      if (!is_close(only0[0].data()[i], only2[0].data()[i], 1e-7f)) { depth_differs = true; break; }
    CHECK(depth_differs);

    // --- batch elements are independent ---
    // attention mixes across tokens, never across the batch. running sample 0
    // alone must match its slice of the batched run.
    tl::Tensor x0 = tl::slice(x, 0, 0, 1); // [1, 3, 8, 12]
    auto solo = vit.forward(x0, {2});
    int64_t per_sample = only2[0].numel() / 2;
    for (int i = 0; i < per_sample; ++i)
      CHECK(is_close(solo[0].data()[i], only2[0].data()[i], 1e-5f));

    // --- n_storage = 0 takes the other cat branch ---
    tl::dino::DinoViT vit0(16, 2, 4, 4, 0); // prefix = 1 (cls only)
    auto f0 = vit0.forward(x, {1});
    CHECK(f0.size() == 1);
    CHECK(f0[0].sizes()[1] == 16 && f0[0].sizes()[2] == 2 && f0[0].sizes()[3] == 3);
    for (int i = 0; i < f0[0].numel(); ++i) CHECK(std::isfinite(f0[0].data()[i]));

    // --- parameter count ---
    // patch_embed(2) + cls(1) + storage(1) + 3 blocks * 14 + norm(2) = 48
    CHECK(vit.parameters().size() == 48);

    // --- invalid inputs throw ---
    bool threw = false;
    try { vit.forward(x, {3}); }                       // index == depth
    catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);

    threw = false;
    try { vit.forward(x, {-1}); }                      // negative index
    catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);

    threw = false;
    try { vit.forward(tl::randn({2, 3, 9, 12}), {0}); } // H not divisible by patch
    catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);

    threw = false;
    try { vit.forward(tl::randn({2, 3, 8}), {0}); }     // not 4D
    catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);

    threw = false;
    try { tl::dino::DinoViT bad(16, 2, 3, 4, 1); }      // dim % num_heads != 0
    catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);

    threw = false;
    try { tl::dino::DinoViT bad(16, 2, 8, 4, 1); }      // head_dim=2, not div by 4
    catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);
  }

  std::cout << "dino tests passed" << std::endl;
}
