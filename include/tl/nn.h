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
  Tensor forward(const Tensor& input) const;

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

private:
  MultiHeadAttention msa_;
  LayerNorm norm1_;
  LayerNorm norm2_;
  Linear ff1_;
  Linear ff2_;
  Dropout dropout_; // not needed in this inference only framework, here for convention
};

} // nn
} // tl
