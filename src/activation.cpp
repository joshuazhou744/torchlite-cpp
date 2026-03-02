#include <tl/activation.h>
#include <cmath> // exp, tanh, sqrt
#include <algorithm> // max
#include <cstdint> // int64_t

namespace tl {

// unary sigmoid
Tensor sigmoid(const Tensor& input) {
  Tensor a = input.contiguous();
  Tensor out(a.sizes());
  const float* ap = a.data();
  float* op = out.data();

  for (int64_t i = 0; i < a.numel(); ++i) {
    op[i] = 1.0f / (1.0f + std::exp(-ap[i]));
  }
  return out;
}

// unary relu
Tensor relu(const Tensor& input) {
  Tensor a = input.contiguous();
  Tensor out(a.sizes());

  const float* ap = a.data();
  float* op = out.data();

  const int64_t n = a.numel();
  for (int64_t i = 0; i < n; ++i) {
      op[i] = std::max(0.0f, ap[i]);
  }

  return out;
}

// unary gelu
Tensor gelu(const Tensor& input) {
  Tensor a = input.contiguous();
  Tensor out(a.sizes());
  const float* ap = a.data();
  float* op = out.data();

  const float sqrt_2_over_pi = std::sqrt(2.0f / M_PI);
  const int64_t n = a.numel();
  for (int64_t i = 0; i < n; ++i) {
    float x = ap[i];
    op[i] = 0.5f * x * (1.0f + std::tanh(sqrt_2_over_pi * (x + 0.044715f * x * x * x)));
  }
  return out;
}

}
