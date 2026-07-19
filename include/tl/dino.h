#pragma once
#include <tl/tensor.h>
#include <tl/nn.h>
#include <cstdint>

namespace tl {
namespace nn {

// DINOv3 attention: fused qkv, per-head RoPE (half-split) on patch tokens only
class DinoAttention {
public:
  DinoAttention(int64_t dim, int64_t num_heads); // x: [N, T, dim]
  Tensor forward(const Tensor& x, const Tensor& cos, const Tensor& sin, int64_t prefix) const;
  std::vector<Tensor*> parameters();
private:
  Linear qkv_; // dim -> 3*dim
  Linear proj_; // dim -> dim
  int64_t num_heads_;
  int64_t head_dim_;
};

// pre-norm block: x += ls1*attn(norm1(x))
// x += ls2*mlp(norm2(x))
class DinoBlock {
public:
  DinoBlock(int64_t dim, int64_t num_heads, int64_t mlp_hidden);
  Tensor forward(const Tensor& x, const Tensor& cos, const Tensor& sin int64_t prefix) const;
  std::vector<Tensor*> parameters();
private:
  LayerNorm norm1_;
  DinoAttention attn_;
  Tensor ls1_; // element-wise scalar / LayerScale gamma
  LayerNorm norm2_;
  Linear fc1_; // dim -> mlp_hidden
  GeLUExact act_;
  Linear fc2_; // mlp_hidden -> dim
  Tensor ls2_;
};

// DINOv3 ViT backbone
class DinoViT {
public:
  DinoViT(int64_t dim, int64_t depth, int64_t num_heads, int64_t patch_size, int64_t n_storage);
  // x [N, 3, H, W]
  // returns feature maps [N, dim, H/p, W/p] after each block index in layers
  // each passed through the shared final norm, prefix tokens stripped
  std::vector<Tensor> forward_features(const Tensor& x, const std::vector<int64_t>& layers) const;
  std::vector<Tensor*> parameters();
private:
  Conv2d patch_embed_; // 3 -> dim, k = s = patch
  Tensor cls_token_; // [1, 1, dim]
  Tensor storage_tokens_; // [1, n_storage, dim]
  std::vector<DinoBlock> blocks_;
  LayerNorm norm_; // shared final norm
  int64_t patch_size_;
  int64_t num_heads_;
};

}
}
