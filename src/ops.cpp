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
Tensor matmul(const Tensor& a_in, const Tensor& b_in) {
  Tensor a = a_in.contiguous();
  Tensor b = b_in.contiguous();
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
        op[m * N + n] = 0.0f;
      }
      for (int64_t k = 0; k < K; ++k) {
        float a_val = ap[m * K + k];
        for (int64_t n = 0; n < N; ++n) {
          op[m * N + n] += a_val * bp[k * N + n];
        }
      }
    }
  }
  return out;
}

// matrix transpose
Tensor transpose(const Tensor& a, int64_t dim0, int64_t dim1) {
  int64_t ndim = a.sizes().size();
  if (dim0 < 0 || dim0 >= ndim || dim1 < 0 || dim1 >= ndim) {
    throw std::invalid_argument("Tranpose: dimension out of range");
  }
  auto out_sizes = a.sizes();
  auto out_strides = a.strides();

  // simply swap the metadata
  std::swap(out_sizes[dim0], out_sizes[dim1]);
  std::swap(out_strides[dim0], out_strides[dim1]);

  // use private constructor to create new view
  return Tensor(a.data_, out_sizes, out_strides, a.offset_);
}

Tensor reshape(const Tensor& a, const std::vector<int64_t>& new_sizes) {
  // verify total elements match
  int64_t new_numel = 1;
  for (int64_t s: new_sizes) {
    new_numel *= s;
  }

  if (new_numel != a.numel()) {
    throw std::invalid_argument("Reshape: new shape must have same number of elements");
  }

  // make sure tensor is contiguous
  Tensor c = a.is_contiguous() ? a : a.contiguous();

  // compute new strides of new shape
  std::vector<int64_t> new_strides(new_sizes.size());
  int64_t stride = 1;
  for (int i = static_cast<int>(new_sizes.size()) - 1; i >= 0; --i) {
    new_strides[i] = stride;
    stride *= new_sizes[i];
  }

  // create new view using private constructor
  return Tensor(c.data_, new_sizes, new_strides, c.offset_);
}

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

// scale a tensor by a factor (scalar)
Tensor scale(const Tensor& input, float scalar) {
  Tensor a = input.contiguous();
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
Tensor softmax(const Tensor& input) {
  Tensor a = input.contiguous();
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

// sum along the dimension of a tensor
Tensor sum(const Tensor& input, int64_t dim, bool keepdim) {
  Tensor a = input.contiguous();
  const auto& sizes = a.sizes();
  int64_t ndim = sizes.size();

  if (dim < 0 || dim >= ndim) {
    throw std::invalid_argument("Sum: dimension out of range");
  }

  // build output shape
  std::vector<int64_t> out_sizes;
  for (int64_t i = 0; i < ndim; ++i) {
    if (i == dim) {
      if (keepdim) out_sizes.push_back(1);
    } else {
      out_sizes.push_back(sizes[i]);
    }
  }

  Tensor out(out_sizes);
  float* op = out.data();
  const float* ap = a.data();

  // outer = product of dims before dim
  // D = size of dim being reduced
  // inner = product of dims after dim
  int64_t outer = 1, inner = 1;
  int64_t D = sizes[dim];
  for (int64_t i = 0; i < dim; ++i) outer *= sizes[i];
  for (int64_t i = dim + 1; i < ndim; ++i) inner *= sizes[i];

  for (int64_t o = 0; o < outer; ++o) {
    for (int64_t n = 0; n < inner; ++n) {
      float acc = 0.0f;
      for (int64_t d = 0; d < D; ++d) {
        acc += ap[o * D * inner + d * inner + n];
      }
      op[o * inner + n] = acc;
    }
  }

  return out;
}

Tensor mean(const Tensor& input, int64_t dim, bool keepdim) {
  Tensor s = sum(input, dim, keepdim);
  int64_t D = input.sizes()[dim];
  return scale(s, 1.0f / static_cast<float>(D));
}

}
