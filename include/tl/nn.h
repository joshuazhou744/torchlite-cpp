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

} // nn
} // tl
