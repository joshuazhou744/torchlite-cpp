#include <tl/nn.h>
#include <tl/ops.h>
#include <tl/factory.h>
#include <tl/activation.h>

#include <random>
#include <cmath>
#include <stdexcept>

namespace tl {
namespace nn {

// Linear layer
Linear::Linear(int64_t in_features, int64_t out_features, bool use_bias)
  : weight_(randn({out_features, in_features})),
    bias_(use_bias ? randn({out_features}): zeros({out_features})),
    use_bias_(use_bias)
{
  weight_.set_requires_grad(true);
  bias_.set_requires_grad(true);
}

Tensor Linear::forward(const Tensor& input) const {
  Tensor out = matmul(input, transpose(weight_, 0, 1)); // xW^T
  if (use_bias_) {
    out = add(out, bias_); // + bias
  }
  return out;
}

// get Linear layer parameters
std::vector<Tensor*> Linear::parameters() {
  std::vector<Tensor*> params = {&weight_};
  if (use_bias_) params.push_back(&bias_);
  return params;
}

// Layer normalization
LayerNorm::LayerNorm(int64_t normalized_shape, float eps)
  : gamma_(ones({normalized_shape})),
    beta_(zeros({normalized_shape})),
    normalized_shape_(normalized_shape),
    eps_(eps)
{
  gamma_.set_requires_grad(true);
  beta_.set_requires_grad(true);
}

Tensor LayerNorm::forward(const Tensor& input) const {
  int64_t dim = input.sizes().size() - 1; // get last dim
  Tensor m = mean(input, dim, true);
  Tensor v = variance(input, dim, true);

  Tensor num = sub(input, m); // x - mean
  Tensor denom = sqrt(add(v, full(v.sizes(), eps_))); // sqrt(var + eps)

  Tensor normed = div(num, denom); // normalized

  // learnable scale and offset to tune normalization
  return add(mul(normed, gamma_), beta_);
}

// get LayerNorm parameters
std::vector<Tensor*> LayerNorm::parameters() {
  return {&gamma_, &beta_};
}

// Dropout layer
// not needed because we are using torchlite for inference only, will keep here just in case
Dropout::Dropout(float p) : p_(p) {}
Tensor Dropout::forward(const Tensor& input, bool training) const {
  Tensor a = input.contiguous();
  if (!training || p_ == 0.0f) {
    return a;
  }

  Tensor out(a.sizes());
  const float* ap = a.data();
  float* op = out.data();

  std::mt19937 gen(std::random_device{}());
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);

  float scale = 1.0f / (1.0f - p_);
  const int64_t n = a.numel();
  for (int64_t i = 0; i < n; ++i) {
    op[i] = (dist(gen) < p_) ? 0.0f : ap[i] * scale;
  }
  return out;
}

// Multi headed self-attention layer
MultiHeadAttention::MultiHeadAttention(int64_t d_model, int64_t num_heads)
  : d_model_(d_model),
    num_heads_(num_heads),
    head_dim_(d_model / num_heads),
    q_proj_(d_model, d_model), // Linear(in_features, out_features)
    k_proj_(d_model, d_model),
    v_proj_(d_model, d_model),
    out_proj_(d_model, d_model)
{
  if (d_model % num_heads != 0) {
    throw std::invalid_argument("MultiHeadAttention: d_model must be divisible by num_heads");
  }
}

// get MSA head parameters (aggregate of all linear layer params)
std::vector<Tensor*> MultiHeadAttention::parameters() {
  std::vector<Tensor*> params;
  for (auto* sub: {&q_proj_, &k_proj_, &v_proj_, &out_proj_}) {
    auto sp = sub->parameters();
    params.insert(params.end(), sp.begin(), sp.end());
  }
  return params;
}

Tensor MultiHeadAttention::forward(const Tensor& input, const Tensor& mask) const {
  // input: [batch, seq, d_model]
  int64_t batch = input.sizes()[0];
  int64_t seq = input.sizes()[1];

  // project to Q, K, V: [batch, seq, d_model]
  Tensor q = q_proj_.forward(input);
  Tensor k = k_proj_.forward(input);
  Tensor v = v_proj_.forward(input);

  // reshape to split heads: [batch, seq, num_heads, head_dim]
  q = reshape(q, {batch, seq, num_heads_, head_dim_});
  k = reshape(k, {batch, seq, num_heads_, head_dim_});
  v = reshape(v, {batch, seq, num_heads_, head_dim_});

  // transpose each projection to [batch, num_heads, seq, head_dim]
  q = transpose(q, 1, 2);
  k = transpose(k, 1, 2);
  v = transpose(v, 1, 2);

  // scaled dot-product attention scores
  // Q @ K^T -> [batch, num_heads, seq, seq]
  Tensor scores = matmul(q, transpose(k, -2, -1));
  scores = scale(scores, 1.0f / std::sqrt(static_cast<float>(head_dim_)));

  // apply mask: set padded positions to -inf before softmax
  if (!mask.empty()) {
    scores = add(scores, mask);
  }

  // softmax over last dimension -> attention weights
  Tensor attn = softmax(scores);

  // apply attention to values: [batch, num_heads, seq, head_dim]
  Tensor out = matmul(attn, v);

  // transpose back to original shape: [batch, seq, num_heads, head_dim]
  out = transpose(out, 1, 2);

  // concatenate heads: [batch, seq, d_model]
  out = reshape(out, {batch, seq, d_model_});

  // final output projection layer: [batch, seq, d_model] (same shape as input)
  return out_proj_.forward(out);
}

// Transformer encoder layer
TransformerEncoderLayer::TransformerEncoderLayer(int64_t d_model, int64_t num_heads, int64_t d_ff, float dropout_p)
  : msa_(d_model, num_heads),
    norm1_(d_model),
    norm2_(d_model),
    ff1_(d_model, d_ff),
    ff2_(d_ff, d_model),
    dropout_(dropout_p)
{}

// get Transformer encoder layer parameters (aggregate of msa, norm and feed-forward layers)
std::vector<Tensor*> TransformerEncoderLayer::parameters() {
  std::vector<Tensor*> params;
  auto append = [&](auto& layer) {
    auto p = layer.parameters();
    params.insert(params.end(), p.begin(), p.end());
  };
  append(msa_);
  append(norm1_);
  append(norm2_);
  append(ff1_);
  append(ff2_);
  return params;
}

Tensor TransformerEncoderLayer::forward(const Tensor& input) const {
  // self-attention block with residuals
  Tensor attn_out = msa_.forward(input);
  attn_out = dropout_.forward(attn_out, false); // inert dropout
  Tensor x = norm1_.forward(add(input, attn_out)); // add residual (input)

  // feed-forward block with residuals
  Tensor ff_out = ff1_.forward(x);
  ff_out = gelu(ff_out);
  ff_out = ff2_.forward(ff_out);
  ff_out = dropout_.forward(ff_out, false); // inert dropout
  return norm2_.forward(add(x, ff_out)); // add residual to output
}

// Transformer encoder
TransformerEncoder::TransformerEncoder(int64_t d_model, int64_t num_heads, int64_t d_ff, int64_t num_layers, float dropout_p) {
  for (int64_t i = 0; i < num_layers; ++i) {
    layers_.emplace_back(d_model, num_heads, d_ff, dropout_p);
  }
}


Tensor TransformerEncoder::forward(const Tensor& input) const {
  Tensor x = input;
  for (const auto& layer: layers_) {
    x = layer.forward(x);
  }
  return x;
}

// get Transformer encoder parameters (aggregate of all layers)
std::vector<Tensor*> TransformerEncoder::parameters() {
  std::vector<Tensor*> params;
  for (auto& sub: layers_) {
    auto sp = sub.parameters();
    params.insert(params.end(), sp.begin(), sp.end());
  }
  return params;
}

// Position encoding
PositionalEncoding::PositionalEncoding(int64_t d_model, int64_t max_len)
  : pe_({max_len, d_model})
{
  float* p = pe_.data();
  for (int64_t pos = 0; pos < max_len; ++pos) {
    for (int64_t i = 0; i < d_model; i += 2) {
      float angle = pos / std::pow(10000.0f, static_cast<float>(i) / d_model);
      p[pos * d_model + i] = std::sin(angle);
      if (i + 1 < d_model) {
        p[pos * d_model + i + 1] = std::cos(angle);
      }
    }
  }
}
Tensor PositionalEncoding::forward(const Tensor& input) const {
  // input: [batch, seq, d_model]
  int64_t seq = input.sizes()[1];
  Tensor pe_slice = slice(pe_, 0, 0, seq); //grab first seq
  return add(input, pe_slice); // broadcast add accross batch
}

}
}
