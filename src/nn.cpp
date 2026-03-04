#include <tl/nn.h>
#include <tl/ops.h>
#include <tl/factory.h>

namespace tl {
namespace nn {

// Linear layer
Linear::Linear(int64_t in_features, int64_t out_features, bool use_bias)
  : weight_(randn({out_features, in_features})),
    bias_(use_bias ? randn({out_features}): zeros({out_features})),
    use_bias_(use_bias)
{}
Tensor Linear::forward(const Tensor& input) const {
  Tensor out = matmul(input, transpose(weight_, 0, 1)); // xW^T
  if (use_bias_) {
    out = add(out, bias_); // + bias
  }
  return out;
}

// Layer normalization
LayerNorm::LayerNorm(int64_t normalized_shape, float eps)
  : gamma_(ones({normalized_shape})),
    beta_(zeros({normalized_shape})),
    normalized_shape_(normalized_shape),
    eps_(eps)
{}
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

// Dropout layer: not needed really for my purposes, will keep here just in case
Dropout::Dropout(float p) : p_(p) {}
Tensor Dropout::forward(const Tensor& input, bool training) const {
  return input.contiguous();
}

}
}
