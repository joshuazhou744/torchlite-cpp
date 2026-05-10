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
};

// Sequential: chain of modules
class Sequential: public Module {
public:
  Sequential(std::vector<Module*> layers);
  Tensor forward(const Tensor& input) const override;
  std::vector<Tensor*> parameters() override;

private:
  std::vector<Module*> layers_;
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
  Tensor weight_; // shape: [out_features, in_features]
  Tensor bias_; // shape: [out_features]
  bool use_bias_;
};

// Conv2d layer: slide C_out filters of shape (C_in, kH, kW) over (N, C_in, H, W) input
// square kernels only for simplicity
class Conv2d: public Module {
public:
  Conv2d(int64_t in_channels, int64_t out_channels, int64_t kernel_size, int64_t stride = 1, int64_t padding = 0, bool use_bias = true);
  Tensor forward(const Tensor& input) const override;
  std::vector<Tensor*> parameters() override;
  const Tensor& weight() const { return weight_; }
  const Tensor& bias() const { return bias_; }

private:
  Tensor weight_; // (out_channels, in_channels, kernel_size, kernel_size)
  Tensor bias_; // (out_channels,)
  int64_t stride_, padding_;
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
class Dropout {
public:
  Dropout(float p = 0.1f);
  Tensor forward(const Tensor& input, bool training = true) const;

private:
  float p_; // zeroing probability (dropout rate)
};

// MultiHeadAttention: split input into parallel attention heads
class MultiHeadAttention {
public:
  MultiHeadAttention(int64_t d_model, int64_t num_heads);
  Tensor forward(const Tensor& input, const Tensor& mask = Tensor()) const;

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
class TransformerEncoderLayer {
public:
  TransformerEncoderLayer(int64_t d_model, int64_t num_heads, int64_t d_ff, float dropout_p = 0.1f);
  Tensor forward(const Tensor& input) const;

  MultiHeadAttention& msa() { return msa_; }
  LayerNorm& norm1() { return norm1_; }
  LayerNorm& norm2() { return norm2_; }
  Linear& ff1() { return ff1_; }
  Linear& ff2() { return ff2_; }

  std::vector<Tensor*> parameters();

private:
  MultiHeadAttention msa_;
  LayerNorm norm1_;
  LayerNorm norm2_;
  Linear ff1_;
  Linear ff2_;
  Dropout dropout_; // not needed in this inference only framework, here for convention
};

// TransformerEncoder: stack of N encoder layers
class TransformerEncoder {
public:
  TransformerEncoder(int64_t d_model, int64_t num_heads, int64_t d_ff, int64_t num_layers, float dropout_p = 0.1f);
  Tensor forward(const Tensor& input) const;

  TransformerEncoderLayer& layer(int64_t i) { return layers_[i]; }

  std::vector<Tensor*> parameters();

private:
  std::vector<TransformerEncoderLayer> layers_;
};

// PositionalEncoding: add sinusoidal position information to input tensors
class PositionalEncoding {
public:
  PositionalEncoding(int64_t d_model, int64_t max_len = 5000);
  Tensor forward(const Tensor& input) const;

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

} // nn
} // tl
