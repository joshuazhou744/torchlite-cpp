#include <tl/ops.h>
#include <cstdint> // for int64_t
#include <stdexcept>
#include <algorithm> // for max()

namespace tl {

// helper function to validate two tensors have same shape
//static void check_same_shape(const Tensor& a, const Tensor& b) {
  //if (a.sizes() != b.sizes()) {
  //throw std::invalid_argument("Tensor shapes must match");
//}
//}

// helper function to map a multidimensional index to a flat index considering broadcasting
static int64_t get_broadcast_index(int64_t linear_index,
                                  const std::vector<int64_t>& t_sizes,
                                  const std::vector<int64_t>& t_strides,
                                  const std::vector<int64_t>& target_sizes) {
  int64_t physical_index = 0;
  int ndim = target_sizes.size();
  int t_ndim = t_sizes.size();

  for (int i = ndim - 1; i >= 0; --i) {
    int64_t dim_index = linear_index % target_sizes[i];
    linear_index /= target_sizes[i];

    // if dimension exists int he smaller tensor and isn't size 1, use stride
    int t_dim_index = i - (ndim - t_ndim);
    if (t_dim_index >= 0 && t_sizes[t_dim_index] != 1) {
      physical_index += dim_index * t_strides[t_dim_index];
    }
  }

  return physical_index;
}

// helper function to find the final broadcasted shape
static std::vector<int64_t> compute_broadcast_shape(const std::vector<int64_t>& a, const std::vector<int64_t>& b) {
  int ndim_a = a.size();
  int ndim_b = b.size();
  int ndim_out = std::max(ndim_a, ndim_b);
  std::vector<int64_t> out_shape(ndim_out);

  for (int i = ndim_out - 1; i >= 0; --i) {
    int64_t size_a = (i < ndim_out - ndim_a) ? 1 : a[i - (ndim_out - ndim_a)];
    int64_t size_b = (i < ndim_out - ndim_b) ? 1 : b[i - (ndim_out - ndim_b)];

    if (size_a != size_b && size_a != 1 && size_b != 1) {
      throw std::invalid_argument("Incompatible dimensions for broadcasting");
    }

    out_shape[i] = std::max(size_a, size_b);
  }
  return out_shape;
}

// element-wise addition
Tensor add(const Tensor& a, const Tensor& b) {
  // determine resulting shape
  std::vector<int64_t> out_shape = compute_broadcast_shape(a.sizes(), b.sizes());
  Tensor out(out_shape);

  const auto& sizes_a = a.sizes();
  const auto& strides_a = a.strides();
  const auto& sizes_b = b.sizes();
  const auto& strides_b = b.strides();

  float* op = out.data();
  const float* ap = a.data();
  const float* bp = b.data();

  // iterate through the outputs flat memory
  const int64_t n = out.numel();
  for (int64_t i = 0; i < n; ++i) {
    // map outputs linear index to physical index
    int64_t index_a = get_broadcast_index(i, sizes_a, strides_a, out_shape);
    int64_t index_b = get_broadcast_index(i, sizes_b, strides_b, out_shape);

    op[i] = ap[index_a] + bp[index_b];
  }
  return out;
}


Tensor mul(const Tensor& a, const Tensor& b) {
  // determine resulting shape
  std::vector<int64_t> out_shape = compute_broadcast_shape(a.sizes(), b.sizes());
  Tensor out(out_shape);

  const auto& sizes_a = a.sizes();
  const auto& strides_a = a.strides();
  const auto& sizes_b = b.sizes();
  const auto& strides_b = b.strides();

  float* op = out.data();
  const float* ap = a.data();
  const float* bp = b.data();

  // iterate through the outputs flat memory
  const int64_t n = out.numel();
  for (int64_t i = 0; i < n; ++i) {
    // map outputs linear index to physical index
    int64_t index_a = get_broadcast_index(i, sizes_a, strides_a, out_shape);
    int64_t index_b = get_broadcast_index(i, sizes_b, strides_b, out_shape);

    op[i] = ap[index_a] * bp[index_b];
  }
  return out;
}

// matrix multiplication
Tensor matmul(const Tensor& a, const Tensor& b) {
  const auto& s_a = a.sizes();
  const auto& s_b = b.sizes();

  if (s_a.size() < 2 || s_b.size() < 2) {
    throw std::invalid_argument("matmul requires at least 2D tensors");
  }

  // identify matrix dims
  int64_t M = s_a[s_a.size() - 2];
  int64_t K = s_a[s_a.size() - 1];
  int64_t K2 = s_b[s_b.size() - 2];
  int64_t N = s_b[s_b.size() - 1];

  if (K != K2) {
    throw std::invalid_argument("Incompatible dimensions for matmul");
  }

  // determine batch dimensions
  std::vector<int64_t> batch_a(s_a.begin(), s_a.end() - 2);
  std::vector<int64_t> batch_b(s_b.begin(), s_b.end() - 2);
  std::vector<int64_t> batch_out = compute_broadcast_shape(batch_a, batch_b);

  const auto& strides_a = a.strides();
  const auto& strides_b = b.strides();
  std::vector<int64_t> strides_a_batch(strides_a.begin(), strides_a.end() - 2);
  std::vector<int64_t> strides_b_batch(strides_b.begin(), strides_b.end() - 2);

  // construct final output shape
  std::vector<int64_t> out_shape = batch_out;
  out_shape.push_back(M);
  out_shape.push_back(N);
  Tensor out(out_shape);

  // batched execution
  int64_t num_batches = 1;
  for (int64_t dim: batch_out) {
    num_batches *= dim;
  }

  for (int64_t i = 0; i < num_batches; ++i) {
    int64_t index_a = get_broadcast_index(i, batch_a, strides_a_batch, batch_out);
    int64_t index_b = get_broadcast_index(i, batch_b, strides_b_batch, batch_out);

    const float* ap = a.data() + index_a;
    const float* bp = b.data() + index_b;
    float* op = out.data() + (i * M * N);

    for (int64_t m = 0; m < M; ++m) {
      for (int64_t n = 0; n < N; ++n) {
        float sum = 0.0f;
        for (int64_t k = 0; k < K; ++k) {
          sum += ap[m * K + k] * bp[k * N + n];
        }
        op[m * N + n] = sum;
      }
    }
  }
  return out;
}

// matrix transpose
Tensor transpose(const Tensor& a, int64_t dim0, int64_t dim1) {
  auto out_sizes = a.sizes();
  std::swap(out_sizes[dim0], out_sizes[dim1]);
  Tensor out(out_sizes);

  const auto& a_sizes = a.sizes();

  std::vector<int64_t> coords(a_sizes.size());

  for (int64_t i = 0; i < a.numel(); ++i) {
    // convert linear index to multi-dim coords
    int64_t temp_index = i;
    for (int d = a_sizes.size() - 1; d >= 0; --d) {
      coords[d] = temp_index % a_sizes[d];
      temp_index /= a_sizes[d];
    }

    // swap coordinates for the new tensor
    std::swap(coords[dim0], coords[dim1]);

    // calculate output linear index
    int64_t out_index = 0;
    const auto& out_strides = out.strides();
    for (size_t d = 0; d < out_sizes.size(); ++d) {
      out_index += coords[d] * out_strides[d];
    }
    out.data()[out_index] = a.data()[i];
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

// scale a tensor by a factor (scalar)
Tensor scale(const Tensor& a, float scalar) {
  Tensor out(a.sizes());

  const float* ap = a.data();
  float* op = out.data();

  const int64_t n = a.numel();
  for (int64_t i = 0; i < n; ++i) {
    op[i] = ap[i] * scalar;
  }

  return out;
}

// softmax, squeeze everything between 0-1 (into probabilities) in the last dim
Tensor softmax(const Tensor& a) {
  const auto& sizes = a.sizes();
  if (sizes.empty()) {
    throw std::invalid_argument("Softmax requires at least 1D tensor");
  }

  Tensor out(sizes);

  const float* ap = a.data();
  float* op = out.data();

  const int64_t D = sizes.back();
  const int64_t outer = a.numel() / D;

  for (int64_t i = 0; i < outer; ++i) {
    const float* row = ap + i * D;
    float* out_row = op + i * D;

    // find maximum
    float max_val = row[0];
    for (int64_t j = 1; j < D; ++j) {
      if (row[j] > max_val) {
        max_val = row[j];
      }
    }

    // exp(x - max) and sum
    float sum = 0.0f;
    for (int64_t j = 0; j < D; ++j) {
      out_row[j] = std::exp(row[j] - max_val);
      sum += out_row[j];
    }

    // normalize
    for (int64_t j = 0; j < D; ++j) {
      out_row[j] /= sum;
    }

  }

  return out;
}

}
