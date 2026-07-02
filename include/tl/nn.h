#pragma once

#include <tl/tensor.h>
#include <tl/activation.h>
#include <tl/ops.h>

namespace tl {
namespace nn {

// Base nn module (abstract class)
class Module {
public:
  virtual ~Module() = default; // use default destructor
  virtual Tensor forward(const Tensor& input) const = 0; // pure virtual forward
  virtual std::vector<Tensor*> parameters() = 0; // pure virtual parameters
  virtual std::vector<Tensor*> buffers() { return {}; } // generic buffer holder
  virtual void set_training(bool) {}
};

// Sequential: chain of modules
class Sequential: public Module {
public:
  Sequential(std::vector<Module*> layers);
  Tensor forward(const Tensor& input) const override;
  std::vector<Tensor*> parameters() override;
  std::vector<Tensor*> buffers() override;
  void set_training(bool t) override;
  void train() { set_training(true); }
  void eval() { set_training(false); }

private:
  std::vector<Module*> layers_;
};

// Checkpoint: module wrapper for gradient checkpointing
class Checkpoint: public Module {
public:
  Checkpoint(Module* wrapped) : wrapped_(wrapped) {}
  Tensor forward(const Tensor& input) const override;
  std::vector<Tensor*> parameters() override { return wrapped_->parameters(); }
  std::vector<Tensor*> buffers() override { return wrapped_->buffers(); }
  void set_training(bool t) override { wrapped_->set_training(t); }
private:
  Module* wrapped_;
};

// Linear layer: y = xW^T + b
class Linear: public Module {
public:
  Linear(int64_t in_features, int64_t out_features, bool use_bias = true);
  // forward function to do input @ weight + bias (matmul and add bias)
  Tensor forward(const Tensor& input) const override;
  std::vector<Tensor*> parameters() override;
  void set_weight(const Tensor& w) { weight_ = w; }
  void set_bias(const Tensor& b) { bias_ = b; }
  const Tensor& weight() const { return weight_; }
  const Tensor& bias() const { return bias_; }

private:
  Tensor weight_; // shape: [in_features, out_features], pre-transposed
  Tensor bias_; // shape: [out_features]
  bool use_bias_;
};

// Conv2d layer: slide C_out filters of shape (C_in, kH, kW) over (N, C_in, H, W) input
// square kernels only for simplicity
class Conv2d: public Module {
public:
  Conv2d(int64_t in_channels, int64_t out_channels, int64_t kernel_size, int64_t stride = 1, int64_t padding = 0, int64_t groups = 1, bool use_bias = true);
  Tensor forward(const Tensor& input) const override;
  std::vector<Tensor*> parameters() override;
  const Tensor& weight() const { return weight_; }
  const Tensor& bias() const { return bias_; }

private:
  Tensor weight_; // (out_channels, in_channels, kernel_size, kernel_size)
  Tensor bias_; // (out_channels,)
  int64_t stride_, padding_, groups_;
  bool use_bias_;
};

// Layer normalization: normalize across last dimension
class LayerNorm: public Module {
public:
  LayerNorm(int64_t normalized_shape, float eps = 1e-5);
  Tensor forward(const Tensor& input) const override;
  std::vector<Tensor*> parameters() override;
  void set_gamma(const Tensor& g) { gamma_ = g; }
  void set_beta(const Tensor& b) { beta_ = b; }
  const Tensor& gamma() const { return gamma_; }
  const Tensor& beta() const { return beta_; }

private:
  Tensor gamma_; // learnable scale, shape: [normalized_shape]
  Tensor beta_; // learnable offset, shape: [normalized_shape]
  int64_t normalized_shape_;
  float eps_; // tiny epsilon to avoid division by 0
};

// Dropout: turn random elements to zero during training
class Dropout: public Module {
public:
  Dropout(float p = 0.1f);
  Tensor forward(const Tensor& input) const override;
  std::vector<Tensor*> parameters() override { return {}; }

  void set_training(bool t) override { training_ = t; }
  bool training() const { return training_; }

private:
  float p_; // zeroing probability (dropout rate)
  bool training_ = true;
};

// MultiHeadAttention: split input into parallel attention heads
class MultiHeadAttention {
public:
  MultiHeadAttention(int64_t d_model, int64_t num_heads);
  Tensor forward(const Tensor& input, const Tensor& mask = Tensor()) const;
  Tensor forward(const Tensor& query, const Tensor& context, const Tensor& mask) const; // cross-attn forward
  void set_training(bool t);

  Linear& q_proj() { return q_proj_; }
  Linear& k_proj() { return k_proj_; }
  Linear& v_proj() { return v_proj_; }
  Linear& out_proj() { return out_proj_; }

  std::vector<Tensor*> parameters();

private:
  int64_t d_model_; // embedding dimension, hidden dim of output tensor, attention is all you need paper uses d_model, hidden_dim makes more sense to me though
  int64_t num_heads_; // number of attention heads
  int64_t head_dim_; // dimensions per head, d_model / num_heads

  Linear q_proj_; // query projection
  Linear k_proj_; // key projection
  Linear v_proj_; // value projection
  Linear out_proj_; // output projection
};

// TransformerEncoderLayer: MSA and FFN with residual connections and layer normalization
class TransformerEncoderLayer: public Module {
public:
  TransformerEncoderLayer(int64_t d_model, int64_t num_heads, int64_t d_ff, float dropout_p = 0.1f);
  Tensor forward(const Tensor& input) const override;
  std::vector<Tensor*> parameters() override;
  void set_training(bool t) override;

  MultiHeadAttention& msa() { return msa_; }
  LayerNorm& norm1() { return norm1_; }
  LayerNorm& norm2() { return norm2_; }
  Linear& ff1() { return ff1_; }
  Linear& ff2() { return ff2_; }

private:
  MultiHeadAttention msa_;
  LayerNorm norm1_;
  LayerNorm norm2_;
  Linear ff1_;
  Linear ff2_;
  Dropout dropout_; // not needed in this inference only framework, here for convention
};

// TransformerEncoder: stack of N encoder layers
class TransformerEncoder: public Module {
public:
  TransformerEncoder(int64_t d_model, int64_t num_heads, int64_t d_ff, int64_t num_layers, float dropout_p = 0.1f);
  Tensor forward(const Tensor& input) const override;
  std::vector<Tensor*> parameters() override;
  void set_training(bool t) override;

  TransformerEncoderLayer& layer(int64_t i) { return layers_[i]; }

private:
  std::vector<TransformerEncoderLayer> layers_;
};

// TransformerDecoderLayer: masked self-attn + cross-attn + FFN
class TransformerDecoderLayer: public Module {
public:
  TransformerDecoderLayer(int64_t d_model, int64_t num_heads, int64_t d_ff, float dropout_p = 0.1f);
  Tensor forward(const Tensor& input, const Tensor& encoder_output, const Tensor& tgt_mask = Tensor()) const;
  Tensor forward(const Tensor& input) const override; // stub for Module override
  std::vector<Tensor*> parameters() override;
  void set_training(bool t) override;

  MultiHeadAttention& self_attn() { return self_attn_; }
  MultiHeadAttention& cross_attn() { return cross_attn_; }
  LayerNorm& norm1() { return norm1_; }
  LayerNorm& norm2() { return norm2_; }
  LayerNorm& norm3() { return norm3_; }
  Linear& ff1() { return ff1_; }
  Linear& ff2() { return ff2_; }
private:
  MultiHeadAttention self_attn_;
  MultiHeadAttention cross_attn_;
  LayerNorm norm1_, norm2_, norm3_;
  Linear ff1_, ff2_;
  Dropout dropout_;
};

// TransformerDecoder: stack of N decoder layers
class TransformerDecoder: public Module {
public:
  TransformerDecoder(int64_t d_model, int64_t num_heads, int64_t d_ff, int64_t num_layers, float dropout_p = 0.1f);
  Tensor forward(const Tensor& input, const Tensor& encoder_output, const Tensor& tgt_mask = Tensor()) const;
  Tensor forward(const Tensor& input) const override;
  std::vector<Tensor*> parameters() override;
  void set_training(bool t) override;
  TransformerDecoderLayer& layer(int64_t i) { return layers_[i]; }

private:
  std::vector<TransformerDecoderLayer> layers_;
};

// CausalTransformerLayer: decoder-only (GPT-style) transformer layer
class CausalTransformerLayer {
public:
  CausalTransformerLayer(int64_t d_model, int64_t num_heads, int64_t d_ff, float dropout_p = 0.1f);
  Tensor forward(const Tensor& input, const Tensor& mask = Tensor()) const;
  std::vector<Tensor*> parameters();
  void set_training(bool t);

  MultiHeadAttention& self_attn() { return self_attn_; }
  LayerNorm& norm1() { return norm1_; }
  LayerNorm& norm2() { return norm2_; }
  Linear& ff1() { return ff1_; }
  Linear& ff2() { return ff2_; }

private:
  MultiHeadAttention self_attn_;
  LayerNorm norm1_, norm2_;
  Linear ff1_, ff2_;
  Dropout dropout_;
};

// CausalTransformer: stack of N causal layers
class CausalTransformer {
public:
  CausalTransformer(int64_t d_model, int64_t num_heads, int64_t d_ff, int64_t num_layers, float dropout_p = 0.1f);
  Tensor forward(const Tensor& input, const Tensor& mask = Tensor()) const;
  std::vector<Tensor*> parameters();
  void set_training(bool t);
  CausalTransformerLayer& layer(int64_t i) { return layers_[i]; }

private:
  std::vector<CausalTransformerLayer> layers_;
};

// PositionalEncoding: add sinusoidal position information to input tensors
class PositionalEncoding: public Module {
public:
  PositionalEncoding(int64_t d_model, int64_t max_len = 5000);
  Tensor forward(const Tensor& input) const override;
  std::vector<Tensor*> parameters() override { return {}; }

private:
  Tensor pe_; // precomputed positional encoding table with shape: [max_len, d_model]
};

class ReLU: public Module {
public:
  Tensor forward(const Tensor& input) const override { return relu(input); }
  std::vector<Tensor*> parameters() override { return {}; }
};

class Flatten: public Module {
public:
  Tensor forward(const Tensor& input) const override {
    int64_t N = input.sizes()[0];
    return reshape(input, {N, input.numel() / N});
  }
  std::vector<Tensor*> parameters() override { return {}; }
};

class MaxPool2d: public Module {
public:
  MaxPool2d(int64_t kernel_size, int64_t stride = 0, int64_t padding = 0)
    : kernel_size_(kernel_size), stride_(stride), padding_(padding) {}
  Tensor forward(const Tensor& input) const override {
    return max_pool2d(input, kernel_size_, stride_, padding_);
  }
  std::vector<Tensor*> parameters() override { return {}; }
private:
  int64_t kernel_size_, stride_, padding_;
};

class AvgPool2d: public Module {
public:
  AvgPool2d(int64_t kernel_size, int64_t stride = 0, int64_t padding = 0)
    : kernel_size_(kernel_size), stride_(stride), padding_(padding) {}
  Tensor forward(const Tensor& input) const override {
    return avg_pool2d(input, kernel_size_, stride_, padding_);
  }
  std::vector<Tensor*> parameters() override { return {}; }
private:
  int64_t kernel_size_, stride_, padding_;
};

// Upsample: nearest-neighbour spatial upsampling by integer scale factor
class Upsample: public Module {
public:
  Upsample(int64_t scale_factor, int64_t in_channels = 0);
  Tensor forward(const Tensor& input) const override;
  std::vector<Tensor*> parameters() override;
private:
  int64_t scale_factor_;
  int64_t in_channels_;
  Conv2d conv_; // when in_channels > 0
};

// Downsample: strided conv2d to halve spatial dimensions
class Downsample: public Module {
public:
  Downsample(int64_t in_channels);
  Tensor forward(const Tensor& input) const override;
  std::vector<Tensor*> parameters() override { return conv_.parameters(); }
private:
  Conv2d conv_; // stride = 2 to halve
};

// BatchNorm2d: normalize per channel across (N, H, W)
class BatchNorm2d: public Module {
public:
  BatchNorm2d(int64_t num_channels, float eps = 1e-5, float momentum = 0.1f);
  Tensor forward(const Tensor& input) const override;
  std::vector<Tensor*> parameters() override;
  void set_gamma(const Tensor& g) { gamma_ = g; }
  void set_beta(const Tensor& b) { beta_ = b; }
  const Tensor& gamma() const { return gamma_; }
  const Tensor& beta() const { return beta_; }
  void set_training(bool t) override { training_ = t; }
  bool training() const { return training_; }
  std::vector<Tensor*> buffers() override {
    return {&running_mean_, &running_var_};
  }


private:
  mutable Tensor running_mean_; // per-channel mean, shape [C]
  mutable Tensor running_var_; // per-channel variance, shape [C]
  Tensor gamma_; // learnable scale
  Tensor beta_; // learnable shift
  int64_t num_channels_;
  float eps_;
  float momentum_;
  bool training_ = true;
};

// GroupNorm: normalize over groups of channels, input [N, C, *]
class GroupNorm: public Module {
public:
  GroupNorm(int64_t num_groups, int64_t num_channels, float eps = 1e-5);
  Tensor forward(const Tensor& input) const override;
  std::vector<Tensor*> parameters() override;
  void set_gamma(const Tensor& g) { gamma_ = g; }
  void set_beta(const Tensor& b) { beta_ = b; }
  const Tensor& gamma() const { return gamma_; }
  const Tensor& beta() const { return beta_; }

private:
  Tensor gamma_; // learnable scale
  Tensor beta_; // learnable shift
  int64_t num_groups_;
  int64_t num_channels_;
  float eps_;
};

// AdaptiveGroupNorm: GroupNorm with gamma and beta computed from a conditioning vector
class AdaptiveGroupNorm: public Module {
public:
  AdaptiveGroupNorm(int64_t num_groups, int64_t num_channels, int64_t cond_dim);
  Tensor forward(const Tensor& input, const Tensor& cond) const;
  Tensor forward(const Tensor& input) const override;
  std::vector<Tensor*> parameters() override;
private:
  GroupNorm norm_;
  Linear proj_; // cond_dim -> 2 * num_channels (gamma and beta)
};

// SelfAttention2d: spatial self-attention for [N, C, H, W] feature maps
// uses single Conv1x1 qkv_proj
class SelfAttention2d {
public:
  SelfAttention2d(int64_t in_channels, int64_t num_heads);
  Tensor forward(const Tensor& input) const;
  std::vector<Tensor*> parameters();
private:
  GroupNorm norm_;
  Conv2d qkv_proj_; // [3*C, C, 1, 1] Q, K, V packed together
  Conv2d out_proj_; // [C, C, 1, 1]
  int64_t in_channels_;
  int64_t num_heads_;
  int64_t head_dim_;
};

// InputNormalize: scalar (x - mean) / std_deviation applied to input
// stats (mean and std_deviation) stored as buffers
class InputNormalize: public Module {
public:
  InputNormalize();
  Tensor forward(const Tensor& input) const override;
  std::vector<Tensor*> parameters() override { return {}; }
  std::vector<Tensor*> buffers() override { return {&mean_, &std_}; }
  void set_stats(float m, float s);
private:
  mutable Tensor mean_; // [1]
  mutable Tensor std_; // [1]
};

// Embedding: learnable lookup table mapping integer indices to dense vectors
// weight: [num_embeddings, embedding_dim]
class Embedding: public Module {
public:
  Embedding(int64_t num_embeddings, int64_t embedding_dim);
  Tensor forward(const Tensor& input) const override;
  std::vector<Tensor*> parameters() override { return {&weight_}; }
private:
  Tensor weight_; // lookup table [num_embeddings, embedding_dim]
};

// TimestepEmbedding: maps scalar noise level sigma to conditioning vector
// log(sigma) -> sinusoidal encoding -> Linear -> SiLU -> Linear
class TimestepEmbedding: public Module {
public:
  TimestepEmbedding(int64_t dim, int64_t out_dim);
  Tensor forward(const Tensor& sigma) const override; // sigma: [N]
  std::vector<Tensor*> parameters() override;
private:
  Linear fc1_;
  Linear fc2_;
  int64_t dim_;
};

// FourierFeatures: random fourier features for noise level embedding
// weight: fixed random [1, cond_dim / 2]
class FourierFeatures: public Module {
public:
  FourierFeatures(int64_t cond_dim);
  Tensor forward(const Tensor& input) const override;
  std::vector<Tensor*> parameters() override { return {}; }
  std::vector<Tensor*> buffers() override { return {&weight_}; }

private:
  Tensor weight_; // random frequencies for random resolution
};

} // nn
} // tl
