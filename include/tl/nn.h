#pragma once

#include <tl/tensor.h>

namespace tl {
namespace nn {

// Linear layer: y = xW^T + b
class Linear {
public:
  Linear(int64_t in_features, int64_t out_features, bool use_bias = true);
  // forward function to do input @ weight + bias (matmul and add bias)
  Tensor forward(const Tensor& input) const;
  void set_weight(const Tensor& w) { weight_ = w; }
  void set_bias(const Tensor& b) { bias_ = b; }
  std::vector<Tensor*> parameters();

private:
  Tensor weight_; // shape: [out_features, in_features]
  Tensor bias_; // shape: [out_features]
  bool use_bias_;
};

// Layer normalization: normalize across last dimension
class LayerNorm {
public:
  LayerNorm(int64_t normalized_shape, float eps = 1e-5);
  Tensor forward(const Tensor& input) const;
  void set_gamma(const Tensor& g) { gamma_ = g; }
  void set_beta(const Tensor& b) { beta_ = b; }

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

} // nn
} // tl
