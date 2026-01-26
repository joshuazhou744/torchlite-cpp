#include <tl/ops.h>
#include <cstdint> // for int64_t
#include <stdexcept>
#include <algorithm> // for max()

namespace tl {

// helper function to validate two tensors have same shape
static void check_same_shape(const Tensor& a, const Tensor& b) {
  if (a.sizes() != b.sizes()) {
      throw std::invalid_argument("Tensor shapes must match");
  }
}

// element-wise addition
Tensor add(const Tensor& a, const Tensor& b) {
  check_same_shape(a, b);

  Tensor out(a.sizes());

  const float* ap = a.data();
  const float* bp = b.data();
  float* op = out.data();

  const int64_t n = a.numel();
  for (int64_t i = 0; i < n; ++i) {
      op[i] = ap[i] + bp[i];
  }

  return out;
}

// element-wise multiply
Tensor mul(const Tensor& a, const Tensor& b) {
  check_same_shape(a, b);

  Tensor out(a.sizes());

  const float* ap = a.data();
  const float* bp = b.data();
  float* op = out.data();

  const int64_t n = a.numel();
  for (int64_t i = 0; i < n; ++i) {
      op[i] = ap[i] * bp[i];
  }

  return out;
}

// matrix multiplication
Tensor matmul(const Tensor& a, const Tensor& b) {
  const auto& s_a = a.sizes();
  const auto& s_b = b.sizes();

  if (s_a.size() != 2 || s_b.size() != 2) {
    throw std::invalid_argument("matmul requires 2D tensors");
  }

  if (s_a[1] != s_b[0]) {
    throw std::invalid_argument("Incompatible dimensions for matmul");
  }

  int64_t M = s_a[0];
  int64_t K = s_a[1];
  int64_t N = s_b[1];

  Tensor out({M, N});
  const float* ap = a.data();
  const float* bp = b.data();
  float* op = out.data();

  for (int64_t i = 0; i < M; ++i) {
    for (int64_t j = 0; i < N; ++j) {
      float sum = 0.0f;
      for (int64_t k = 0; k < K; ++k) {
        // index = row * total_cols + col
        sum += ap[i * K + k] * bp[k * N + j];
      }
      op[i * N + j] = sum;
    }
  }

  return out;
}

// unary sigmoid
Tensor sigmoid(const Tensor& a) {
  Tensor out(a.sizes());
  const float* ap = a.data();
  float* op = out.data();

  for (int64_t i = 0; i < a.numel(); ++i) {
    op[i] = 1.0f / (1.0f + std::exp(-ap[i]));
  }
  return out;
}

// unary relu
Tensor relu(const Tensor& a) {
  Tensor out(a.sizes());

  const float* ap = a.data();
  float* op = out.data();

  const int64_t n = a.numel();
  for (int64_t i = 0; i < n; ++i) {
      op[i] = std::max(0.0f, ap[i]);
  }

  return out;
}

}
