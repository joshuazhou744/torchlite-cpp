#include <tl/nn.h>
#include <tl/ops.h>
#include <tl/factory.h>
#include <tl/activation.h>
#include <tl/autograd.h>

#include <random>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <tuple>

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
  if (use_bias_) return {&weight_, &bias_};
  return {&weight_};
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
  if (use_bias_) return {&weight_, &bias_};
  return {&weight_};
}

Tensor Conv2d::forward(const Tensor& input) const {
  return conv2d(input, weight_, use_bias_ ? bias_ : Tensor(), stride_, padding_, groups_);
}

// ConvTranspose2d
ConvTranspose2d::ConvTranspose2d(int64_t in_channels, int64_t out_channels, int64_t kernel_size, int64_t stride, bool use_bias)
  : weight_(scale(randn({in_channels, out_channels, kernel_size, kernel_size}), std::sqrt(2.0f / (in_channels * kernel_size * kernel_size)))),
    bias_(use_bias ? zeros({out_channels}) : Tensor()),
    stride_(stride),
    use_bias_(use_bias)
{
  if (kernel_size != stride) {
    throw std::invalid_argument("ConvTranspose2d: only kernel_size == stride is supported");
  }
  weight_.set_requires_grad(true);
  if (use_bias_) bias_.set_requires_grad(true);
}

// [N, C_in, H, W] -> [N, C_out, H*k, W*k]
Tensor ConvTranspose2d::forward(const Tensor& input) const {
  if (input.sizes().size() != 4) {
    throw std::invalid_argument("ConvTranspose2d: input must be shape [N, C_in, H, W]");
  }
  int64_t N = input.sizes()[0];
  int64_t C_in = input.sizes()[1];
  int64_t H = input.sizes()[2];
  int64_t W = input.sizes()[3];
  int64_t k = stride_;
  int64_t C_out = weight_.sizes()[1];

  if (C_in != weight_.sizes()[0]) {
    throw std::invalid_argument("ConvTranspose2d: input channels must match in_channels");
  }

  // gather every spatial position into a row
  // [N, C_in, H, W] -> [N * H * W, C_in]
  Tensor x = transpose(transpose(input, 1, 2), 2, 3); // [N, H, W, C_in]
  x = reshape(x, {N * H * W, C_in}); // [N * H * W, C_in]

  // each input channel's scatter pattern flattened: [C_in, C_out * k * k]
  Tensor w = reshape(weight_, {C_in, C_out * k * k});

  Tensor out = matmul(x, w); // [N * H * W, C_out * k * k]

  // interleave the tile offsets back into spatial dims
  out = reshape(out, {N, H, W, C_out, k, k});
  out = transpose(out, 1, 3); // [N, C_out, W, H, k, k]
  out = transpose(out, 2, 3); // [N, C_out, H, W, k, k]
  out = transpose(out, 3, 4); // [N, C_out, H, k, W, k]
  out = reshape(out, {N, C_out, H * k, W * k});

  // add bias
  if (use_bias_) {
    out = add(out, reshape(bias_, {1, C_out, 1, 1}));
  }
  return out;
}

std::vector<Tensor*> ConvTranspose2d::parameters() {
  if (use_bias_) return {&weight_, &bias_};
  return {&weight_};
}

// Layer normalization
LayerNorm::LayerNorm(const std::vector<int64_t>& normalized_shape, float eps, bool elementwise_affine)
  : gamma_(elementwise_affine ? ones(normalized_shape) : Tensor()),
    beta_(elementwise_affine ? zeros(normalized_shape) : Tensor()),
    normalized_shape_(normalized_shape),
    eps_(eps),
    affine_(elementwise_affine)
{
  if (affine_) {
    gamma_.set_requires_grad(true);
    beta_.set_requires_grad(true);
  }
}

Tensor LayerNorm::forward(const Tensor& input) const {
  const auto& in_sizes = input.sizes();
  int64_t nd = (int64_t)in_sizes.size();
  int64_t k = (int64_t)normalized_shape_.size();

  if (nd < k) {
    throw std::invalid_argument("LayerNorm: input has fewer dims than normalized_shape");
  }
  for (int64_t i = 0; i < k; ++i) {
    if (in_sizes[nd - k + i] != normalized_shape_[i]) {
      throw std::invalid_argument("LayerNorm: input's trailing dims must match normalized_shape");
    }
  }

  // flatten the trailing k dims into one
  std::vector<int64_t> flat(in_sizes.begin(), in_sizes.end() - k);
  int64_t tail = 1;
  for (int64_t i = 0; i < k; ++i) tail *= normalized_shape_[i];
  flat.push_back(tail);

  Tensor x = reshape(input, flat);
  int64_t dim = (int64_t)flat.size() - 1;
  Tensor m = mean(x, dim, true);
  Tensor v = variance(x, dim, true);
  Tensor normed = div(sub(x, m), sqrt(add(v, full(v.sizes(), eps_))));
  normed = reshape(normed, in_sizes);

  if (!affine_) return normed;
  return add(mul(normed, gamma_), beta_);
}

// get LayerNorm parameters
std::vector<Tensor*> LayerNorm::parameters() {
  if (!affine_) return {};
  return {&gamma_, &beta_};
}

// QK Layer normalization
QKLayerNorm::QKLayerNorm(const std::vector<int64_t>& scale_shape, float eps)
  : scale_(ones(scale_shape)),
    eps_(eps)
{
  scale_.set_requires_grad(true);
}

Tensor QKLayerNorm::forward(const Tensor& input) const {
  const auto& in_sizes = input.sizes();
  int64_t nd = (int64_t)in_sizes.size();
  int64_t k = (int64_t)scale_.sizes().size();

  if (nd < k) {
    throw std::invalid_argument("QKLayerNorm: input has fewer dims than scale_shape");
  }
  for (int64_t i = 0; i < k; ++i) {
    if (in_sizes[nd - k + i] != scale_.sizes()[i]) {
      throw std::invalid_argument("QKLayerNorm: input's trailing dims must match scale_shape");
    }
  }

  int64_t dim = nd - 1; // last dim
  Tensor m = mean(input, dim, true);
  Tensor v = variance(input, dim, true);
  Tensor normed = div(sub(input, m), sqrt(add(v, full(v.sizes(), eps_))));

  return mul(normed, scale_);
}

std::vector<Tensor*> QKLayerNorm::parameters() {
  return {&scale_};
}

// QK RMS normalization
QKRMSNorm::QKRMSNorm(const std::vector<int64_t>& scale_shape, float eps)
  : scale_(ones(scale_shape)),
    eps_(eps)
{
  scale_.set_requires_grad(true);
}

Tensor QKRMSNorm::forward(const Tensor& input) const {
  const auto& in_sizes = input.sizes();
  int64_t nd = (int64_t)in_sizes.size();
  int64_t k = (int64_t)scale_.sizes().size();

  if (nd < k) {
    throw std::invalid_argument("QKRMSNorm: input has fewer dims than scale_shape");
  }
  for (int64_t i = 0; i < k; ++i) {
    if (in_sizes[nd - k + i] != scale_.sizes()[i]) {
      throw std::invalid_argument("QKRMSNorm: input's trailing dims must match scale_shape");
    }
  }

  int64_t dim = nd - 1; // last dim
  Tensor ms = mean(mul(input, input), dim, true);
  Tensor normed = div(input, sqrt(add(ms, full(ms.sizes(), eps_))));

  return mul(normed, scale_);
}

std::vector<Tensor*> QKRMSNorm::parameters() {
  return {&scale_};
}

// RMS normalization
RMSNorm::RMSNorm(const std::vector<int64_t>& normalized_shape, float eps, bool elementwise_affine)
  : gamma_(elementwise_affine ? ones(normalized_shape) : Tensor()),
    normalized_shape_(normalized_shape),
    eps_(eps),
    affine_(elementwise_affine)
{
  if (affine_) gamma_.set_requires_grad(true);
}

Tensor RMSNorm::forward(const Tensor& input) const {
  const auto& in_sizes = input.sizes();
  int64_t nd = (int64_t)in_sizes.size();
  int64_t k = (int64_t)normalized_shape_.size();

  if (nd < k) {
    throw std::invalid_argument("RMSNorm: input has fewer dims than normalized_shape");
  }
  for (int64_t i = 0; i < k; ++i) {
    if (in_sizes[nd - k + i] != normalized_shape_[i]) {
      throw std::invalid_argument("RMSNorm: input's trailing dims must match normalized_shape");
    }
  }

  // flatten the trailing k dims into one
  std::vector<int64_t> flat(in_sizes.begin(), in_sizes.end() - k);
  int64_t tail = 1;
  for (int64_t i = 0; i < k; ++i) tail *= normalized_shape_[i];
  flat.push_back(tail);

  Tensor x = reshape(input, flat);
  int64_t dim = (int64_t)flat.size() - 1;
  Tensor ms = mean(mul(x, x), dim, true);
  Tensor normed = div(x, sqrt(add(ms, full(ms.sizes(), eps_))));
  normed = reshape(normed, in_sizes);

  // learnable scale
  if (!affine_) return normed;
  return mul(normed, gamma_);
}

std::vector<Tensor*> RMSNorm::parameters() {
  if (!affine_) return {};
  return {&gamma_};
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
  return collect_params(q_proj_, k_proj_, v_proj_, out_proj_);
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

// SelfAttention2d: spatial self-attention using Conv1x1 QKV projection
SelfAttention2d::SelfAttention2d(int64_t in_channels, int64_t num_heads)
  : norm_(std::max(int64_t(1), in_channels / 32), in_channels),
    qkv_proj_(in_channels, 3 * in_channels, 1, 1, 0),
    out_proj_(in_channels, in_channels, 1, 1, 0),
    in_channels_(in_channels),
    num_heads_(num_heads),
    head_dim_(in_channels / num_heads)
{
  if (in_channels % num_heads != 0) {
    throw std::invalid_argument("SelfAttention2d: in_channels must be divisble by num_heads");
  }

  out_proj_.set_weight(zeros(out_proj_.weight().sizes()));
}

std::vector<Tensor*> SelfAttention2d::parameters() {
  return collect_params(norm_, qkv_proj_, out_proj_);
}

Tensor SelfAttention2d::forward(const Tensor& input) const {
  int64_t N = input.sizes()[0];
  int64_t C = input.sizes()[1];
  int64_t H = input.sizes()[2];
  int64_t W = input.sizes()[3];

  // normalize over channels first
  Tensor x = norm_.forward(input);

  // project to QKV: [N, 3C, H, W]
  Tensor qkv = qkv_proj_.forward(x);

  // split into Q, K, V each [N, C, H, W]
  Tensor q = slice(qkv, 1, 0, C);
  Tensor k = slice(qkv, 1, C, C*2);
  Tensor v = slice(qkv, 1, C*2, C*3);

  // reshape to [N, num_heads, H*W, head_dim]
  q = transpose(reshape(q, {N, num_heads_, head_dim_, H*W}), 2, 3);
  Tensor kt = reshape(k, {N, num_heads_, head_dim_, H*W});
  v = transpose(reshape(v, {N, num_heads_, head_dim_, H*W}), 2, 3);

  // scaled dot-product attention
  Tensor scores = scale(matmul(q, kt), 1.0f / std::sqrt((float)head_dim_)); // [N, num_heads, H*W, H*W]
  scores = softmax(scores); // softmax over last dim
  Tensor attn = matmul(scores, v); // [N, num_heads, H*W, head_dim]

  // merge heads: [N, C, H, W]
  attn = transpose(attn, 2, 3); // [N, num_heads, head_dim, H*W]
  attn = reshape(attn, {N, C, H, W});

  // output projection then residual
  return add(input, out_proj_.forward(attn));
}

// Transformer encoder layer
TransformerEncoderLayer::TransformerEncoderLayer(int64_t d_model, int64_t num_heads, int64_t d_ff, float dropout_p)
  : msa_(d_model, num_heads),
    norm1_({d_model}),
    norm2_({d_model}),
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
  return collect_params(msa_, norm1_, norm2_, ff1_, ff2_);
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
  return collect_params(layers_);
}

// Transformer decoder layer (cross-attention)
TransformerDecoderLayer::TransformerDecoderLayer(int64_t d_model, int64_t num_heads, int64_t d_ff, float dropout_p)
  : self_attn_(d_model, num_heads),
    cross_attn_(d_model, num_heads),
    norm1_({d_model}),
    norm2_({d_model}),
    norm3_({d_model}),
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
  return collect_params(self_attn_, cross_attn_, norm1_, norm2_, norm3_, ff1_, ff2_);
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
  return collect_params(layers_);
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
    norm1_({d_model}),
    norm2_({d_model}),
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
  return collect_params(self_attn_, norm1_, norm2_, ff1_, ff2_);
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
  return collect_params(layers_);
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

// Upsample
Upsample::Upsample(int64_t scale_factor, int64_t in_channels)
  : scale_factor_(scale_factor),
    in_channels_(in_channels),
    conv_(std::max(in_channels, int64_t(1)), std::max(in_channels, int64_t(1)), 3, 1, 1)
{}

std::vector<Tensor*> Upsample::parameters() {
  if (in_channels_ > 0) return conv_.parameters();
  return {};
}

Tensor Upsample::forward(const Tensor& input) const {
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

  if (in_channels_ > 0) {
    return conv_.forward(out);
  }
  return out;
}

// Downsample
Downsample::Downsample(int64_t in_channels)
  : conv_(in_channels, in_channels, 3, 2, 1) // kernel_size=3, stride=2, padding=1
{}

Tensor Downsample::forward(const Tensor& input) const {
  return conv_.forward(input);
}

// Batch norm 2D
BatchNorm2d::BatchNorm2d(int64_t num_channels, float eps, float momentum, bool affine)
  : gamma_(affine ? ones({num_channels}) : Tensor()),
    beta_(affine ? zeros({num_channels}) : Tensor()),
    num_channels_(num_channels),
    eps_(eps),
    momentum_(momentum),
    affine_(affine),
    running_mean_(zeros({num_channels})),
    running_var_(ones({num_channels}))
{
  if (affine_) {
    gamma_.set_requires_grad(true);
    beta_.set_requires_grad(true);
  }
  running_mean_.set_requires_grad(false);
  running_var_.set_requires_grad(false);
}

Tensor BatchNorm2d::forward(const Tensor& input) const {
  Tensor normed;

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
    normed = div(diff, denom);
  } else {
    // eval mode, use running stats
    Tensor m = reshape(running_mean_, {1, num_channels_, 1, 1});
    Tensor v = reshape(running_var_,  {1, num_channels_, 1, 1});
    Tensor denom = sqrt(add(v, full(v.sizes(), eps_)));
    normed = div(sub(input, m), denom);
  }

  if (!affine_) return normed;

  // learnable scale and shift
  Tensor g = reshape(gamma_, {1, num_channels_, 1, 1});
  Tensor b = reshape(beta_, {1, num_channels_, 1, 1});
  return add(mul(normed, g), b);
}

std::vector<Tensor*> BatchNorm2d::parameters() {
  if (!affine_) return {};
  return {&gamma_, &beta_};
}

// Group normalization
GroupNorm::GroupNorm(int64_t num_groups, int64_t num_channels, float eps, bool affine)
  : gamma_(affine ? ones({num_channels}) : Tensor()),
    beta_(affine ? zeros({num_channels}) : Tensor()),
    num_groups_(num_groups),
    num_channels_(num_channels),
    eps_(eps),
    affine_(affine)
{
  if (num_channels % num_groups != 0) {
    throw std::invalid_argument("GroupNorm: num_channels must be divisible by num_groups");
  }
  if (affine_) {
    gamma_.set_requires_grad(true);
    beta_.set_requires_grad(true);
  }
}

std::vector<Tensor*> GroupNorm::parameters() {
  if (!affine_) return {};
  return {&gamma_, &beta_};
}

Tensor GroupNorm::forward(const Tensor& input) const {
  // input: [N, C, *] (any spatial dims after channel dim, C)
  int64_t N = input.sizes()[0];
  int64_t C = input.sizes()[1];

  if (C != num_channels_) {
    throw std::invalid_argument("GroupNorm: input channels must match num_channels");
  }

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

  if (!affine_) return normed;

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
  : norm_(num_groups, num_channels, 1e-5f, false),
    proj_(cond_dim, 2*num_channels)
{}

std::vector<Tensor*> AdaptiveGroupNorm::parameters() {
  return collect_params(norm_, proj_);
}

Tensor AdaptiveGroupNorm::forward(const Tensor& input, const Tensor& cond) const {
  // input: [N, C, H, W], cond: [N, cond_dim]
  int64_t N = input.sizes()[0];
  int64_t C = input.sizes()[1];

  Tensor normed = norm_.forward(input); // pure normalization, no affine

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
  return collect_params(fc1_, fc2_);
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

// Embedding LUT
Embedding::Embedding(int64_t num_embeddings, int64_t embedding_dim)
  : weight_(randn({num_embeddings, embedding_dim}))
{}

Tensor Embedding::forward(const Tensor& input) const {
  // input: [...] integer indices
  // ouput: [..., embedding_dim]
  auto in_sizes = input.sizes();
  int64_t num_indices = input.numel();
  int64_t embedding_dim = weight_.sizes()[1];

  std::vector<int64_t> out_sizes(in_sizes.begin(), in_sizes.end());
  out_sizes.push_back(embedding_dim);

  Tensor out(out_sizes);
  const float* ip = input.data();
  const float* wp = weight_.data();
  float* op = out.data();

  for (int64_t i = 0; i < num_indices; ++i) {
    int64_t index = static_cast<int64_t>(ip[i]);
    for (int64_t d = 0; d < embedding_dim; ++d) {
      op[i * embedding_dim + d] = wp[index * embedding_dim + d];
    }
  }
  return out;
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
  // input: [N] -> [N, 1]
  Tensor x = unsqueeze(input, 1);
  // [N, 1] @ [1, cond_dim / 2] -> [N, cond_dim / 2]
  Tensor f = scale(matmul(x, weight_), 2.0f * (float)M_PI);
  return cat({cos(f), sin(f)}, 1); // [N, cond_dim]
}

// LSTMCell
LSTMCell::LSTMCell(int64_t input_size, int64_t hidden_size)
  : forget_linear_(input_size + hidden_size, hidden_size),
    input_linear_(input_size + hidden_size, hidden_size),
    candidate_linear_(input_size + hidden_size, hidden_size),
    output_linear_(input_size + hidden_size, hidden_size),
    hidden_size_(hidden_size)
{
  // init forget gate bias to 1.0 so cell starts off remembering lots from c_prev
  Tensor b = full({hidden_size}, 1.0f);
  b.set_requires_grad(true);
  forget_linear_.set_bias(b);
}


std::pair<Tensor, Tensor> LSTMCell::forward(const Tensor& x_t, const Tensor& h_prev, const Tensor& c_prev) const {
  // current input and previous hidden state
  Tensor z = cat({x_t, h_prev}, 1);

  Tensor f_t = sigmoid(forget_linear_.forward(z)); // keep fraction of old memory
  Tensor i_t = sigmoid(input_linear_.forward(z)); // write fraction of candidate
  Tensor g_t = tanh(candidate_linear_.forward(z)); // candidate cell state

  // update cell state: weighted sum of old memory and candidate cell state
  Tensor c_t = add(mul(f_t, c_prev), mul(i_t, g_t));

  // hidden state
  Tensor o_t = sigmoid(output_linear_.forward(z));
  Tensor h_t = mul(o_t, tanh(c_t));

  return {h_t, c_t};
}

std::vector<Tensor*> LSTMCell::parameters() {
  return collect_params(forget_linear_, input_linear_, candidate_linear_, output_linear_);
}

// LSTM
LSTM::LSTM(int64_t input_size, int64_t hidden_size)
  : cell_(input_size, hidden_size),
    input_size_(input_size),
    hidden_size_(hidden_size)
{}

std::pair<Tensor, Tensor> LSTM::forward(const Tensor& x) const {
  if (x.sizes().size() != 3 || x.sizes()[2] != input_size_) {
    throw std::invalid_argument("LSTM: expected input [N, T, input_size]");
  }

  int64_t N = x.sizes()[0];
  int64_t T = x.sizes()[1];

  // initial state are zero
  Tensor h_t = zeros({N, hidden_size_});
  Tensor c_t = zeros({N, hidden_size_});

  // loop over each timestep
  for (int64_t t = 0; t < T; ++t) {
    Tensor x_t = squeeze(slice(x, 1, t, t + 1), 1); // [N, 1, input] -> [N, input]
    std::tie(h_t, c_t) = cell_.forward(x_t, h_t, c_t);
  }
  return {h_t, c_t};
}

std::vector<Tensor*> LSTM::parameters() {
  return cell_.parameters();
}

// SwiGLU

// helper to calculate swiglu hidden dim
// apply formula and round up to nearest multiple of 256
// hidden = round_up(2 * dim_multiplier * dim / 3, multiple_of)
static int64_t swiglu_hidden(int64_t dim, int64_t dim_multiplier, int64_t multiple_of) {
  int64_t h = (2 * dim_multiplier * dim) / 3;
  return multiple_of * ((h + multiple_of - 1) / multiple_of);
}

SwiGLU::SwiGLU(int64_t dim, int64_t dim_multiplier, int64_t multiple_of)
  : hidden_dim_(swiglu_hidden(dim, dim_multiplier, multiple_of)),
    swish_(dim, hidden_dim_, false),
    gate_(dim, hidden_dim_, false),
    out_(hidden_dim_, dim, false)
{}

Tensor SwiGLU::forward(const Tensor& input) const {
  return out_.forward(mul(silu(swish_.forward(input)), gate_.forward(input)));
}

std::vector<Tensor*> SwiGLU::parameters() {
  return collect_params(swish_, gate_, out_);
}

}
}
