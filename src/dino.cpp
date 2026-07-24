#include <tl/dino.h>
#include <tl/tensor.h>
#include <tl/factory.h>
#include <tl/nn.h>

#include <cmath>
#include <stdexcept>

namespace tl {
namespace dino {

// DinoAttention
DinoAttention::DinoAttention(int64_t dim, int64_t num_heads)
  : qkv_(dim, 3 * dim), // fused Q, K, V with bias
    proj_(dim, dim),
    num_heads_(num_heads),
    head_dim_(dim / num_heads)
{
  if (dim % num_heads != 0) {
    throw std::invalid_argument("DinoAttention: dim must be divisible by num_heads");
  }
}

// x: [N, T, dim], cos/sin: [T - prefix, head_dim] (patch tokens only)
// prefix: leading tokens (cls + storage/register tokens) that don't get RoPE encoding
Tensor DinoAttention::forward(const Tensor& x, const Tensor& cos, const Tensor& sin, int64_t prefix) const {
  int64_t N = x.sizes()[0];
  int64_t T = x.sizes()[1];
  int64_t dim = num_heads_ * head_dim_;

  // fused projection then split: [N, T, dim]
  Tensor qkv = qkv_.forward(x);
  Tensor q = slice(qkv, 2, 0, dim);
  Tensor k = slice(qkv, 2, dim, 2*dim);
  Tensor v = slice(qkv, 2, 2*dim, 3*dim);

  // split heads: [N, num_heads, T, head_dim]
  q = transpose(reshape(q, {N, T, num_heads_, head_dim_}), 1, 2);
  k = transpose(reshape(k, {N, T, num_heads_, head_dim_}), 1, 2);
  v = transpose(reshape(v, {N, T, num_heads_, head_dim_}), 1, 2);

  // RoPE on patch tokens (skip prefix)
  auto rope = [&](const Tensor& t) -> Tensor {
    if (prefix <= 0) return apply_rotary_half(t, cos, sin);
    Tensor pre = slice(t, 2, 0, prefix); // [N, H, prefix, head_dim]
    Tensor patch = slice(t, 2, prefix, T); // [N, H, T - prefix, head_dim]
    return cat({pre, apply_rotary_half(patch, cos, sin)}, 2);
  };
  q = rope(q);
  k = rope(k);

  // scaled dot-product attention scores: [N, num_heads, T, T]
  Tensor scores = matmul(q, transpose(k, -2, -1));
  scores = scale(scores, 1.0f / std::sqrt(static_cast<float>(head_dim_)));
  Tensor attn = softmax(scores);

  // weighted sum of values: [N, num_heads, T, head_dim]
  Tensor out = matmul(attn, v);

  // merge heads: [N, T, dim]
  out = transpose(out, 1, 2);
  out = reshape(out, {N, T, dim});

  return proj_.forward(out);
}

std::vector<Tensor*> DinoAttention::parameters() {
  return nn::collect_params(qkv_, proj_);
}

// DinoBlock
DinoBlock::DinoBlock(int64_t dim, int64_t num_heads, int64_t mlp_hidden)
  : norm1_(dim, 1e-5f),
    attn_(dim, num_heads),
    ls1_(full({dim}, 1e-5f)),
    norm2_(dim, 1e-5f),
    fc1_(dim, mlp_hidden),
    fc2_(mlp_hidden, dim),
    ls2_(full({dim}, 1e-5f))
{}

// x = x + ls1 * attn(norm1(x))
// x = x + ls2 * mlp(norm2(x))
Tensor DinoBlock::forward(const Tensor& x, const Tensor& cos, const Tensor& sin, int64_t prefix) const {
  // attention
  Tensor a = attn_.forward(norm1_.forward(x), cos, sin, prefix);
  Tensor h = add(x, mul(a, ls1_));

  // mlp: dim -> mlp_hidden -> GeLU -> dim
  Tensor m = fc2_.forward(act_.forward(fc1_.forward(norm2_.forward(h))));
  return add(h, mul(m, ls2_));
}


std::vector<Tensor*> DinoBlock::parameters() {
  return nn::collect_params(norm1_, attn_, ls1_, norm2_, fc1_, fc2_, ls2_);
}

}
}
