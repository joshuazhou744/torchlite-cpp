#include <tl/nn.h>
#include <tl/ops.h>
#include <tl/factory.h>
#include <tl/activation.h>
#include <tl/autograd.h>

#include <random>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace tl {
namespace nn {

// Sequential model
Sequential::Sequential(std::vector<Module*> layers)
  : layers_(layers) {}

Tensor Sequential::forward(const Tensor& input) const {
  Tensor out = input;
  for (Module* layer: layers_) {
    out = layer->forward(out);
  }
  return out;
}

void Sequential::set_training(bool t) {
  for (Module* layer: layers_) {
    layer->set_training(t);
  }
}

std::vector<Tensor*> Sequential::parameters() {
  std::vector<Tensor*> params;
  for (Module* layer: layers_) {
    auto p = layer->parameters();
    params.insert(params.end(), p.begin(), p.end());
  }
  return params;
}

std::vector<Tensor*> Sequential::buffers() {
  std::vector<Tensor*> bufs;
  for (Module* layer: layers_) {
    auto b = layer->buffers();
    bufs.insert(bufs.end(), b.begin(), b.end());
  }
  return bufs;
}

// Checkpoint
Tensor Checkpoint::forward(const Tensor& input) const {
  // run block silently (no grad) so no intermediates are stored
  bool prev = grad_enabled();
  grad_enabled() = false;
  Tensor output = wrapped_->forward(input);
  grad_enabled() = prev;

  if (auto fn = track<CheckpointBackward>(output, {&input})) {
    fn->wrapped_ = wrapped_;
    fn->saved_input = input;
  }
  return output;
}

// Linear layer
Linear::Linear(int64_t in_features, int64_t out_features, bool use_bias)
  : weight_(scale(randn({in_features, out_features}), std::sqrt(2.0f / in_features))),
    bias_(zeros({out_features})),
    use_bias_(use_bias)
{
  weight_.set_requires_grad(true);
  bias_.set_requires_grad(true);
}

Tensor Linear::forward(const Tensor& input) const {
  Tensor out = matmul(input, weight_); // x @ W, W is pre-transposed
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

// Convolution 2D
Conv2d::Conv2d(int64_t in_channels, int64_t out_channels, int64_t kernel_size, int64_t stride, int64_t padding, int64_t groups, bool use_bias)
  : weight_(scale(randn({out_channels, in_channels / groups, kernel_size, kernel_size}), std::sqrt(2.0f / ((in_channels / groups) * kernel_size * kernel_size)))),
    bias_(zeros({out_channels})),
    stride_(stride),
    padding_(padding),
    groups_(groups),
    use_bias_(use_bias)
{
  if (in_channels % groups != 0) throw std::invalid_argument("Conv2d: in_channels not divisible by groups");
  if (out_channels % groups != 0) throw std::invalid_argument("Conv2d: out_channels not divisible by groups");
  weight_.set_requires_grad(true);
  if (use_bias_) bias_.set_requires_grad(true);
}

// get Conv2d layer parameters
std::vector<Tensor*> Conv2d::parameters() {
  std::vector<Tensor*> params = {&weight_};
  if (use_bias_) params.push_back(&bias_);
  return params;
}

Tensor Conv2d::forward(const Tensor& input) const {
  return conv2d(input, weight_, use_bias_ ? bias_ : Tensor(), stride_, padding_, groups_);
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
Tensor Dropout::forward(const Tensor& input) const {
  Tensor a = input.contiguous();
  if (!training_ || p_ == 0.0f) {
    return a;
  }

  Tensor out(a.sizes());
  Tensor mask_scale(a.sizes());
  const float* ap = a.data();
  float* op = out.data();
  float* mp = mask_scale.data();

  std::mt19937 gen(std::random_device{}());
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);

  float scale = 1.0f / (1.0f - p_);
  const int64_t n = a.numel();
  for (int64_t i = 0; i < n; ++i) {
    float m = (dist(gen) < p_) ? 0.0f : scale;
    mp[i] = m;
    op[i] = ap[i] * m;
  }

  if (input.requires_grad) {
    if (auto fn = track<DropoutBackward>(out, {&input})) {
      fn->mask_cache = mask_scale;
    }
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

// MSA set training mode
void MultiHeadAttention::set_training(bool t) {
  q_proj_.set_training(t);
  k_proj_.set_training(t);
  v_proj_.set_training(t);
  out_proj_.set_training(t);
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

// cross-attention forward
Tensor MultiHeadAttention::forward(const Tensor& query, const Tensor& context, const Tensor& mask) const {
  // query: [batch, target_seq, d_model] from decoder
  // context: [batch, source_seq, d_model] from encoder
  int64_t batch = query.sizes()[0];
  int64_t target_seq = query.sizes()[1];
  int64_t source_seq = context.sizes()[1];

  Tensor q = q_proj_.forward(query);
  Tensor k = k_proj_.forward(context);
  Tensor v = v_proj_.forward(context);

  q = reshape(q, {batch, target_seq, num_heads_, head_dim_});
  k = reshape(k, {batch, source_seq, num_heads_, head_dim_});
  v = reshape(v, {batch, source_seq, num_heads_, head_dim_});

  q = transpose(q, 1, 2);
  k = transpose(k, 1, 2);
  v = transpose(v, 1, 2);

  // scores: [batch, num_heads, target_seq, source_seq]
  Tensor scores = matmul(q, transpose(k, -2, -1));
  scores = scale(scores, 1.0f / std::sqrt(static_cast<float>(head_dim_)));

  if (!mask.empty()) {
    scores = add(scores, mask);
  }

  Tensor attn = softmax(scores);
  Tensor out = matmul(attn, v); // [batch, num_heads, target_seq, head_dim]

  out = transpose(out, 1, 2);
  out = reshape(out, {batch, target_seq, d_model_});

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

void TransformerEncoderLayer::set_training(bool t) {
  // forward to every Module child
  msa_.set_training(t);
  norm1_.set_training(t);
  norm2_.set_training(t);
  ff1_.set_training(t);
  ff2_.set_training(t);
  dropout_.set_training(t);
}

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
  attn_out = dropout_.forward(attn_out); // inert dropout
  Tensor x = norm1_.forward(add(input, attn_out)); // add residual (input)

  // feed-forward block with residuals
  Tensor ff_out = ff1_.forward(x);
  ff_out = gelu(ff_out);
  ff_out = ff2_.forward(ff_out);
  ff_out = dropout_.forward(ff_out); // inert dropout
  return norm2_.forward(add(x, ff_out)); // add residual to output
}

// Transformer encoder
TransformerEncoder::TransformerEncoder(int64_t d_model, int64_t num_heads, int64_t d_ff, int64_t num_layers, float dropout_p) {
  for (int64_t i = 0; i < num_layers; ++i) {
    layers_.emplace_back(d_model, num_heads, d_ff, dropout_p);
  }
}

void TransformerEncoder::set_training(bool t) {
  for (auto& layer: layers_) {
    layer.set_training(t);
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

// Transformer decoder layer (cross-attention)
TransformerDecoderLayer::TransformerDecoderLayer(int64_t d_model, int64_t num_heads, int64_t d_ff, float dropout_p)
  : self_attn_(d_model, num_heads),
    cross_attn_(d_model, num_heads),
    norm1_(d_model),
    norm2_(d_model),
    norm3_(d_model),
    ff1_(d_model, d_ff),
    ff2_(d_ff, d_model),
    dropout_(dropout_p)
{}

void TransformerDecoderLayer::set_training(bool t) {
  self_attn_.set_training(t);
  cross_attn_.set_training(t);
  norm1_.set_training(t);
  norm2_.set_training(t);
  norm3_.set_training(t);
  ff1_.set_training(t);
  ff2_.set_training(t);
  dropout_.set_training(t);
}

std::vector<Tensor*> TransformerDecoderLayer::parameters() {
  std::vector<Tensor*> params;
  auto append = [&](auto& layer) {
    auto p = layer.parameters();
    params.insert(params.end(), p.begin(), p.end());
  };
  append(self_attn_);
  append(cross_attn_);
  append(norm1_);
  append(norm2_);
  append(norm3_);
  append(ff1_);
  append(ff2_);
  return params;
}

Tensor TransformerDecoderLayer::forward(const Tensor& input, const Tensor& encoder_output, const Tensor& tgt_mask) const {
  // masked self-attention block
  Tensor attn_out = self_attn_.forward(input, tgt_mask);
  attn_out = dropout_.forward(attn_out);
  Tensor x = norm1_.forward(add(input, attn_out));

  // cross-attention block: Q from decoder, K/V from encoder
  Tensor cross_out = cross_attn_.forward(x, encoder_output, Tensor());
  cross_out = dropout_.forward(cross_out);
  x = norm2_.forward(add(x, cross_out));

  // feed-forward block
  Tensor ff_out = ff2_.forward(gelu(ff1_.forward(x)));
  ff_out = dropout_.forward(ff_out);
  return norm3_.forward(add(x, ff_out));
}

// stub forward
Tensor TransformerDecoderLayer::forward(const Tensor& input) const {
  (void)input;
  throw std::logic_error("TransformerDecoderLayer: use forward(input, encoder_output, tgt_mask)");
}

// Transformer decoder
TransformerDecoder::TransformerDecoder(int64_t d_model, int64_t num_heads, int64_t d_ff, int64_t num_layers, float dropout_p) {
  for (int64_t i = 0; i < num_layers; ++i) {
    layers_.emplace_back(d_model, num_heads, d_ff, dropout_p);
  }
}

void TransformerDecoder::set_training(bool t) {
  for (auto& layer: layers_) layer.set_training(t);
}

std::vector<Tensor*> TransformerDecoder::parameters() {
  std::vector<Tensor*> params;
  for (auto& layer: layers_) {
    auto p = layer.parameters();
    params.insert(params.end(), p.begin(), p.end());
  }
  return params;
}

Tensor TransformerDecoder::forward(const Tensor& input, const Tensor& encoder_output, const Tensor& tgt_mask) const {
  Tensor x = input;
  for (const auto& layer: layers_) {
    x = layer.forward(x, encoder_output, tgt_mask);
  }
  return x;
}

// stub forward
Tensor TransformerDecoder::forward(const Tensor& input) const {
  (void)input;
  throw std::logic_error("TransformerDecoder: use forward(input, encoder_output, tgt_mask)");
}

// Causal transformer layer
CausalTransformerLayer::CausalTransformerLayer(int64_t d_model, int64_t num_heads, int64_t d_ff, float dropout_p)
  : self_attn_(d_model, num_heads),
    norm1_(d_model),
    norm2_(d_model),
    ff1_(d_model, d_ff),
    ff2_(d_ff, d_model),
    dropout_(dropout_p)
{}

void CausalTransformerLayer::set_training(bool t) {
  self_attn_.set_training(t);
  norm1_.set_training(t);
  norm2_.set_training(t);
  ff1_.set_training(t);
  ff2_.set_training(t);
  dropout_.set_training(t);
}

std::vector<Tensor*> CausalTransformerLayer::parameters() {
  std::vector<Tensor*> params;
  auto append = [&](auto& layer) {
    auto p = layer.parameters();
    params.insert(params.end(), p.begin(), p.end());
  };
  append(self_attn_);
  append(norm1_);
  append(norm2_);
  append(ff1_);
  append(ff2_);
  return params;
}

Tensor CausalTransformerLayer::forward(const Tensor& input, const Tensor& mask) const {
  // masked self-attention block
  Tensor attn_out = self_attn_.forward(input, mask);
  attn_out = dropout_.forward(attn_out);
  Tensor x = norm1_.forward(add(input, attn_out));

  // feed-forward block
  Tensor ff_out = ff1_.forward(x);
  ff_out = gelu(ff_out);
  ff_out = ff2_.forward(ff_out);
  ff_out = dropout_.forward(ff_out);
  return norm2_.forward(add(x, ff_out));
}

// Causal transformer
CausalTransformer::CausalTransformer(int64_t d_model, int64_t num_heads, int64_t d_ff, int64_t num_layers, float dropout) {
  for (int64_t i = 0; i < num_layers; ++i) {
    layers_.emplace_back(d_model, num_heads, d_ff, dropout);
  }
}

void CausalTransformer::set_training(bool t) {
  for (auto& layer: layers_) layer.set_training(t);
}

std::vector<Tensor*> CausalTransformer::parameters() {
  std::vector<Tensor*> params;
  for (auto& layer: layers_) {
    auto p = layer.parameters();
    params.insert(params.end(), p.begin(), p.end());
  }
  return params;
}

Tensor CausalTransformer::forward(const Tensor& input, const Tensor& mask) const {
  Tensor x = input;
  for (const auto& layer: layers_) {
    x = layer.forward(x, mask);
  }
  return x;
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

// Upsample 2D
Upsample2d::Upsample2d(int64_t scale_factor) : scale_factor_(scale_factor) {}

Tensor Upsample2d::forward(const Tensor& input) const {
  // input: [N, C, H, W]
  int64_t N = input.sizes()[0];
  int64_t C = input.sizes()[1];
  int64_t H = input.sizes()[2];
  int64_t W = input.sizes()[3];
  int64_t H_out = H * scale_factor_;
  int64_t W_out = W * scale_factor_;

  Tensor out({N, C, H_out, W_out});
  const float* src = input.data();
  float* dst = out.data();

  for (int64_t n = 0; n < N; ++n) {
    for (int64_t c = 0; c < C; ++c) {
      for (int64_t h = 0; h < H; ++h) {
        for (int64_t w = 0; w < W; ++w) {
          float val = src[n*C*H*W + c*H*W + h*W + w];
          for (int64_t dh = 0; dh < scale_factor_; ++dh) {
            for (int64_t dw = 0; dw < scale_factor_; ++dw) {
              dst[n*C*H_out*W_out + c*H_out*W_out + (h*scale_factor_+dh)*W_out + (w*scale_factor_+dw)] = val;
            }
          }
        }
      }
    }
  }
  return out;
}

// Batch norm 2D
BatchNorm2d::BatchNorm2d(int64_t num_channels, float eps, float momentum)
  : gamma_(ones({num_channels})),
    beta_(zeros({num_channels})),
    num_channels_(num_channels),
    eps_(eps),
    momentum_(momentum),
    running_mean_(zeros({num_channels})),
    running_var_(ones({num_channels}))
{
  gamma_.set_requires_grad(true);
  beta_.set_requires_grad(true);
  running_mean_.set_requires_grad(false);
  running_var_.set_requires_grad(false);
}

Tensor BatchNorm2d::forward(const Tensor& input) const {
  if (training_) {
    // input: (N, C, H, W) -> reduce dims with keepdim -> (1, C, 1, 1)
    // get per-channel mean
    Tensor m = mean(input, 0, true);
    m = mean(m, 2, true);
    m = mean(m, 3, true);

    // per-channel variance
    Tensor diff = sub(input, m);
    Tensor sq = mul(diff, diff);
    Tensor v = mean(sq, 0, true);
    v = mean(v, 2, true);
    v = mean(v, 3, true);
    {
      // update running stats
      NoGradGuard no_grad;
      Tensor m_flat = reshape(m, {num_channels_});
      Tensor v_flat = reshape(v, {num_channels_});
      running_mean_ = add(scale(running_mean_, 1.0f - momentum_), scale(m_flat, momentum_));
      running_var_ = add(scale(running_var_, 1.0f - momentum_), scale(v_flat, momentum_));
    }

    // normalize: (x - mu) / sqrt(var + eps)
    Tensor denom = sqrt(add(v, full(v.sizes(), eps_)));
    Tensor normed = div(diff, denom);

    // learnable scale and shift
    Tensor g = reshape(gamma_, {1, num_channels_, 1, 1});
    Tensor b = reshape(beta_, {1, num_channels_, 1, 1});
    return add(mul(normed, g), b);
  } else {
    // eval mode, use running stats
    Tensor m = reshape(running_mean_, {1, num_channels_, 1, 1});
    Tensor v = reshape(running_var_,  {1, num_channels_, 1, 1});
    Tensor denom = sqrt(add(v, full(v.sizes(), eps_)));
    Tensor normed = div(sub(input, m), denom);

    Tensor g = reshape(gamma_, {1, num_channels_, 1, 1});
    Tensor b = reshape(beta_,  {1, num_channels_, 1, 1});
    return add(mul(normed, g), b);
  }
}

std::vector<Tensor*> BatchNorm2d::parameters() {
  return {&gamma_, &beta_};
}

// Group normalization
GroupNorm::GroupNorm(int64_t num_groups, int64_t num_channels, float eps)
  : gamma_(ones({num_channels})),
    beta_(zeros({num_channels})),
    num_groups_(num_groups),
    num_channels_(num_channels),
    eps_(eps)
{
  if (num_channels % num_groups != 0) {
    throw std::invalid_argument("GroupNorm: num_channels must be divisible by num_groups");
  }
  gamma_.set_requires_grad(true);
  beta_.set_requires_grad(true);
}

std::vector<Tensor*> GroupNorm::parameters() {
  return {&gamma_, &beta_};
}

Tensor GroupNorm::forward(const Tensor& input) const {
  // input: [N, C, *] (any spatial dims after channel dim, C)
  int64_t N = input.sizes()[0];
  int64_t C = input.sizes()[1];
  int64_t spatial = input.numel() / (N * C);
  int64_t group_size = (C / num_groups_) * spatial;

  Tensor x = reshape(input, {N, num_groups_, group_size});

  // mean and variance per (N, group)
  Tensor m = mean(x, 2, true); // [N, num_groups, 1]
  Tensor diff = sub(x, m);
  Tensor v = mean(mul(diff, diff), 2, true); // [N, num_groups, 1]
  Tensor denom = sqrt(add(v, full(v.sizes(), eps_)));

  Tensor normed = div(diff, denom);
  normed = reshape(normed, input.sizes()); // back to [N, C, *]

  // broadcast gamma and beta over N and spatial dims
  // reshape to [1, C, 1] for broadcasting
  std::vector<int64_t> param_shape(input.sizes().size(), 1);
  param_shape[1] = C;
  Tensor g = reshape(gamma_, param_shape);
  Tensor b = reshape(beta_, param_shape);

  return add(mul(normed, g), b);
}

// Adaptive group normalization
AdaptiveGroupNorm::AdaptiveGroupNorm(int64_t num_groups, int64_t num_channels, int64_t cond_dim)
  : norm_(num_groups, num_channels),
    proj_(cond_dim, 2*num_channels)
{}

std::vector<Tensor*> AdaptiveGroupNorm::parameters() {
  return proj_.parameters();
}

Tensor AdaptiveGroupNorm::forward(const Tensor& input, const Tensor& cond) const {
  // input: [N, C, H, W], cond: [N, cond_dim]
  int64_t N = input.sizes()[0];
  int64_t C = input.sizes()[1];

  Tensor normed = norm_.forward(input); // pure normalization, gamma=1, beta=0

  // project cond -> [N, 2*C], split into gamma and beta
  Tensor scale_shift = proj_.forward(cond);
  Tensor gamma = slice(scale_shift, 1, 0, C); // [N, C]
  Tensor beta = slice(scale_shift, 1, C, 2*C); // [N, C]

  // reshape to [N, C, 1, 1] for broadcasting over H, W
  std::vector<int64_t> shape(input.sizes().size(), 1);
  shape[0] = N;
  shape[1] = C;
  gamma = reshape(gamma, shape);
  beta = reshape(beta, shape);

  return add(mul(normed, add(gamma, ones(gamma.sizes()))), beta);
}

Tensor AdaptiveGroupNorm::forward(const Tensor& input) const {
  (void) input;
  throw std::logic_error("AdaptiveGroupNorm: use forward(input, cond)");
}

// Input normalization
InputNormalize::InputNormalize()
  : mean_(zeros({1})),
    std_(ones({1}))
{}

// forward: (input - mean) / std
Tensor InputNormalize::forward(const Tensor& input) const {
  return div(sub(input, mean_), std_);
}

void InputNormalize::set_stats(float m, float s) {
  mean_.data()[0] = m;
  std_.data()[0] = s;
}

// Timestep embedding
TimestepEmbedding::TimestepEmbedding(int64_t dim, int64_t out_dim)
  : fc1_(dim, out_dim),
    fc2_(out_dim, out_dim),
    dim_(dim)
{
  if (dim % 2 != 0) {
    throw std::invalid_argument("TimestepEmbedding: dim must be even");
  }
}

std::vector<Tensor*> TimestepEmbedding::parameters() {
  std::vector<Tensor*> params;
  auto p1 = fc1_.parameters();
  auto p2 = fc2_.parameters();
  params.insert(params.end(), p1.begin(), p1.end());
  params.insert(params.end(), p2.begin(), p2.end());
  return params;
}

Tensor TimestepEmbedding::forward(const Tensor& sigma) const {
  // sigma: [N], batch of scalars
  int64_t N = sigma.sizes()[0];
  int64_t half = dim_ / 2;

  // sinusoidal encoding of log(sigma): [N, dim]
  Tensor emb({N, dim_});
  float* ep = emb.data();
  const float* sp = sigma.data();

  for (int64_t n = 0; n < N; ++n) {
    float log_sigma = std::log(sp[n]);
    for (int64_t i = 0; i < half; ++i) {
      float freq = std::exp(-std::log(10000.0f) * i / (half - 1));
      float angle = log_sigma * freq;
      ep[n * dim_ + i] = std::sin(angle);
      ep[n * dim_ + half + i] = std::cos(angle);
    }
  }

  // Linear -> SiLU -> Linear
  Tensor x = fc1_.forward(emb);
  x = silu(x);
  return fc2_.forward(x);
}

// Fourier features
FourierFeatures::FourierFeatures(int64_t cond_dim)
  : weight_(randn({1, cond_dim / 2}))
{
  if (cond_dim % 2 != 0) {
    throw std::invalid_argument("FourierFeatures: cond_dim must be even");
  }
  weight_.requires_grad = false;
}

Tensor FourierFeatures::forward(const Tensor& input) const {
  // input: [N] -> unsqueeze to [N, 1]
  Tensor x = reshape(input, {input.sizes()[0], 1});
  // [N, 1] @ [1, cond_dim / 2] -> [N, cond_dim / 2]
  Tensor f = mul(matmul(x, weight_), 2.0f * M_PI);
  return cat({cos(f), sin(f)}, 1); // [N, cond_dim]
}

}
}
