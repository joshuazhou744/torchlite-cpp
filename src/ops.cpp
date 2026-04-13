#include <tl/ops.h>
#include <tl/factory.h>
#include <tl/autograd.h>

#include <cstdint> // for int64_t
#include <stdexcept>
#include <algorithm> // for max()
#include <cmath> // for sqrt() and exp()

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

  if (a.requires_grad || b.requires_grad) {
    out.requires_grad = true;
    auto fn = std::make_shared<AddBackward>();
    fn->inputs = {const_cast<Tensor*>(&a), const_cast<Tensor*>(&b)};
    out.grad_fn = fn;
  }

  if (a.requires_grad || b.requires_grad) {
    track<AddBackward>(out, {&a, &b});
  }
  return out;
}

// element-wise subtraction
Tensor sub(const Tensor& a, const Tensor& b) {
  std::vector<int64_t> out_shape = compute_broadcast_shape(a.sizes(), b.sizes());
  Tensor out(out_shape);

  const auto& sizes_a = a.sizes();
  const auto& strides_a = a.strides();
  const auto& sizes_b = b.sizes();
  const auto& strides_b = b.strides();

  float* op = out.data();
  const float* ap = a.data();
  const float* bp = b.data();

  const int64_t n = out.numel();
  for (int64_t i = 0; i < n; ++i) {
    int64_t index_a = get_broadcast_index(i, sizes_a, strides_a, out_shape);
    int64_t index_b = get_broadcast_index(i, sizes_b, strides_b, out_shape);
    op[i] = ap[index_a] - bp[index_b];
  }

  if (a.requires_grad || b.requires_grad) {
    track<SubBackward>(out, {&a, &b});
  }

  return out;
}

// unary sqrt
Tensor sqrt(const Tensor& input) {
  Tensor a = input.contiguous();
  Tensor out(a.sizes());
  const float* ap = a.data();
  float* op = out.data();

  const int64_t n = a.numel();
  for (int64_t i = 0; i < n; ++i) {
    op[i] = std::sqrt(ap[i]);
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

Tensor div(const Tensor& a, const Tensor& b) {
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

    op[i] = ap[index_a] / bp[index_b];
  }
  return out;
}

// matrix multiplication
Tensor matmul(const Tensor& a_in, const Tensor& b_in) {
  Tensor a = a_in.contiguous();
  Tensor b = b_in.contiguous();

  // squeeze and track 1D matrices
  bool squeeze_a = false, squeeze_b = false;
  if (a.sizes().size() == 1) {
    a = reshape(a, {1, a.sizes()[0]}); // [K] -> [1, K]
    squeeze_a = true;
  }
  if (b.sizes().size() == 1) {
    b = reshape(b, {b.sizes()[0], 1}); // [K] -> [K, 1]
    squeeze_b = true;
  }

  // handle 0D tensors
  if (a.sizes().size() < 2 || b.sizes().size() < 2) {
    throw std::invalid_argument("matmul: requires at least 1D tensors");
  }

  const auto& s_a = a.sizes();
  const auto& s_b = b.sizes();

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
  if (squeeze_a && squeeze_b) { // both squeezed
    return reshape(out, {}); // [1, K] @ [K, 1] -> [1, 1] -> scalar
  } else if (squeeze_a) { // only first input matrix squeezed
    std::vector<int64_t> s(out.sizes().begin(), out.sizes().end());
    s.erase(s.end() - 2); // [..., 1, N] @ [..., N] remove fake row dim
    return reshape(out, s);
  } else if (squeeze_b) { // only second input matrix squeezed
    std::vector<int64_t> s(out.sizes().begin(), out.sizes().end());
    s.erase(s.end() - 1); // [..., M, 1] @ [..., M], remove fake col dim
    return reshape(out, s);
  }
  return out;
}

// matrix transpose
Tensor transpose(const Tensor& a, int64_t dim0, int64_t dim1) {
  int64_t ndim = a.sizes().size();

  // negative dim wrapping
  if (dim0 < 0) dim0 += ndim;
  if (dim1 < 0) dim1 += ndim;

  if (dim0 < 0 || dim0 >= ndim || dim1 < 0 || dim1 >= ndim) {
    throw std::invalid_argument("Tranpose: dimension out of range");
  }
  auto out_sizes = a.sizes();
  auto out_strides = a.strides();

  // simply swap the metadata
  std::swap(out_sizes[dim0], out_sizes[dim1]);
  std::swap(out_strides[dim0], out_strides[dim1]);

  Tensor out(a.data_, out_sizes, out_strides, a.offset_);

  if (a.requires_grad) {
    auto fn = track<TransposeBackward>(out, {&a});
    fn->dim0 = dim0;
    fn->dim1 = dim1;
  }

  // use private constructor to create new view
  return out;
}

Tensor reshape(const Tensor& a, const std::vector<int64_t>& new_sizes) {
  // verify total elements match
  int64_t new_numel = 1;
  for (int64_t s: new_sizes) {
    if (s < 0) {
      throw std::invalid_argument("Reshape: negative dims not allowed");
    }
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

  Tensor out(c.data_, new_sizes, new_strides, c.offset_);

  if (a.requires_grad) {
    auto fn = track<ReshapeBackward>(out, {&a});
    fn->input_shape = a.sizes();
  }

  // create new view using private constructor
  return out;
}

// tensor concatenation along existing dimension
Tensor cat(const std::vector<Tensor>& tensors, int64_t dim) {
  if (tensors.empty()) {
    throw std::invalid_argument("cat: empty tensor list");
  }

  int64_t ndim = tensors[0].sizes().size();
  if (dim < 0) dim += ndim;
  if (dim < 0 || dim >= ndim) {
    throw std::invalid_argument("cat: dimension out of range");
  }

  // validate shapes match except along given dim
  for (size_t t = 1; t < tensors.size(); ++t) {
    if ((int64_t)tensors[t].sizes().size() != ndim) {
      throw std::invalid_argument("cat: all tensors must have same number of dimensions");
    }

    for (int64_t d = 0; d < ndim; ++d) {
      if (d != dim && tensors[t].sizes()[d] != tensors[0].sizes()[d]) {
        throw std::invalid_argument("cat: shapes must match except in cat dimension");
      }
    }
  }

  // compute output shape
  std::vector<int64_t> out_sizes(tensors[0].sizes().begin(), tensors[0].sizes().end());
  int64_t total_cat_size = 0;
  for (const auto& t: tensors) {
    total_cat_size += t.sizes()[dim];
  }
  out_sizes[dim] = total_cat_size;

  Tensor out(out_sizes);
  float* op = out.data();

  // outer = product of dimensions before dim
  // inner = product of dimensions after dim
  int64_t outer = 1, inner = 1;
  for (int64_t i = 0; i < dim; ++i) outer *= out_sizes[i];
  for (int64_t i = dim+1; i < ndim; ++i) inner *= out_sizes[i];

  int64_t cat_offset = 0;
  for (const auto& t: tensors) {
    Tensor tc = t.contiguous();
    const float* tp = tc.data();
    int64_t t_dim_size = tc.sizes()[dim];

    for (int64_t o = 0; o < outer; ++o) {
      for (int64_t d = 0; d < t_dim_size; ++d) {
        for (int64_t n = 0; n < inner; ++n) {
          op[o * total_cat_size * inner + (cat_offset + d) * inner + n] = tp[o * t_dim_size * inner + d * inner + n];
        }
      }
    }
    cat_offset += t_dim_size;
  }
  return out;
}

// tensor stacking along a new dimension
Tensor stack(const std::vector<Tensor>& tensors, int64_t dim) {
  if (tensors.empty()) {
    throw std::invalid_argument("stack: empty tensor list");
  }

  int64_t ndim = tensors[0].sizes().size();
  if (dim < 0) dim += ndim + 1;
  if (dim < 0 || dim > ndim) {
    throw std::invalid_argument("stack: dimension out of range");
  }

  // validate all tensors have the same shape
  for (size_t t = 1; t < tensors.size(); ++t) {
    if (tensors[t].sizes() != tensors[0].sizes()) {
      throw std::invalid_argument("stack: all tensors must have same shape");
    }
  }

  // unsqueeze each tensor at given dimension, then concatenate
  std::vector<Tensor> unsqueezed;
  for (const auto& t: tensors) {
    std::vector<int64_t> new_sizes(t.sizes().begin(), t.sizes().end());
    new_sizes.insert(new_sizes.begin() + dim, 1);
    unsqueezed.push_back(reshape(t, new_sizes));
  }

  return cat(unsqueezed, dim);
}

// tensor slicing along a given dimension
Tensor slice(const Tensor& input, int64_t dim, int64_t start, int64_t end) {
  Tensor a = input.contiguous();
  const auto& sizes = a.sizes();
  int64_t ndim = sizes.size();

  if (dim < 0) dim += ndim;
  if (dim < 0 || dim >= ndim) {
    throw std::invalid_argument("slice: dimension out of range");
  }
  if (start < 0 || end > sizes[dim] || start >= end) {
    throw std::invalid_argument("slice: invalid start/end range");
  }

  // build output shape
  std::vector<int64_t> out_sizes(sizes.begin(), sizes.end());
  out_sizes[dim] = end - start;

  Tensor out(out_sizes);
  float* op = out.data();
  const float* ap = a.data();

  int64_t outer = 1, inner = 1;
  int64_t D = sizes[dim];
  int64_t slice_size = end - start;
  for (int64_t i = 0; i < dim; ++i) outer *= sizes[i];
  for (int64_t i = dim+1; i < ndim; ++i) inner *= sizes[i];

  for (int64_t o = 0; o < outer; ++o) {
    for (int64_t d = 0; d < slice_size; ++d) {
      for (int64_t n = 0; n < inner; ++n) {
        op[o * slice_size * inner + d * inner + n] = ap[o * D * inner + (start + d) * inner + n];
      }
    }
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

  if (input.requires_grad) {
    auto fn = track<ScaleBackward>(out, {&input});
    fn->scalar = scalar;
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

  if (sizes.back() == 0) {
    return Tensor(sizes); // no last dim to normalize
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

// argmax, get index of maximum value along a given dimension
Tensor argmax(const Tensor& input, int64_t dim) {
  if (dim < 0) dim += input.sizes().size(); // dimension wrapping

  const auto& sizes = input.sizes();
  int64_t ndim = sizes.size();

  // output shape is input shape with target dim collapsed
  std::vector<int64_t> out_sizes;
  for (int64_t i = 0; i < ndim; ++i) {
    if (i != dim) out_sizes.push_back(sizes[i]);
  }
  if (out_sizes.empty()) out_sizes.push_back(1); // scalar case

  Tensor out(out_sizes);
  Tensor c = input.contiguous();
  const float* ap = c.data();
  float* op = out.data();

  int64_t outer = 1, inner = 1, D = sizes[dim];
  for (int64_t i = 0; i < dim; ++i) outer *= sizes[i];
  for (int64_t i = dim + 1; i < ndim; ++i) inner *= sizes[i];

  for (int64_t o = 0; o < outer; ++o) {
    for (int64_t i = 0; i < inner; ++i) {
      float best = ap[o * D * inner + i];
      int64_t best_index = 0;
      for (int64_t d = 1; d < D; ++d) {
        float val = ap[o * D * inner + d * inner + i];
        if (val > best) {
          best = val;
          best_index = d;
        }
      }
      op[o * inner + i] = static_cast<float>(best_index);
    }
  }

  return out;
}

// sum along the dimension of a tensor
Tensor sum(const Tensor& input, int64_t dim, bool keepdim) {
  Tensor a = input.contiguous();
  const auto& sizes = a.sizes();
  int64_t ndim = sizes.size();

  if (dim < 0) dim += ndim; // negative dim wrapping

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

  if (input.requires_grad) {
    auto fn = track<SumBackward>(out, {&input});
    fn->input_shape = input.sizes();
  }

  return out;
}

// calculate mean of a dimension
Tensor mean(const Tensor& input, int64_t dim, bool keepdim) {
  // negative dim wrapping
  int64_t ndim = input.sizes().size();
  if (dim < 0) dim += ndim;

  Tensor s = sum(input, dim, keepdim);
  int64_t D = input.sizes()[dim];
  if (D == 0) {
    throw std::invalid_argument("Mean: cannot reduce a zero-sized dimension");
  }

  if (input.requires_grad) {
    auto fn = track<MeanBackward>(out, {&input});
    fn->input_shape = input.sizes();
    fn->dim_size = input.sizes()[dim];
  }

  return scale(s, 1.0f / static_cast<float>(D));
}

// calculate the variance of a dimension
// var = mean((x - mean)^2)
Tensor variance(const Tensor& input, int64_t dim, bool keepdim) {
  int64_t ndim = input.sizes().size();
  if (dim < 0) dim += ndim; // dim wrapping

  Tensor m = mean(input, dim, true);
  Tensor diff = sub(input, m);
  Tensor sq = mul(diff, diff);
  Tensor var = mean(sq, dim, keepdim);
  return var;
}

Tensor neg(const Tensor& input) {
  Tensor a = input.contiguous();
  Tensor out(a.sizes());
  const float* ap = a.data();
  float* op = out.data();

  const int64_t n = a.numel();
  for (int64_t i = 0; i < n; ++i) {
    op[i] = -ap[i];
  }

  if (input.requires_grad) {
    track<NegBackward>(out, {&input});
  }

  return out;
}

Tensor exp(const Tensor& input) {
  Tensor a = input.contiguous();
  Tensor out(a.sizes());
  const float* ap = a.data();
  float* op = out.data();

  const int64_t n = a.numel();
  for (int64_t i = 0; i < n; ++i) {
    op[i] = std::exp(ap[i]);
  }
  return out;
}

Tensor pow(const Tensor& input, float x) {
  Tensor a = input.contiguous();
  Tensor out(a.sizes());
  const float* ap = a.data();
  float* op = out.data();

  const int64_t n = a.numel();
  for (int64_t i = 0; i < n; ++i) {
    op[i] = std::pow(ap[i], x);
  }
  return out;
}

Tensor log(const Tensor& input) {
  Tensor a = input.contiguous();
  Tensor out(a.sizes());
  const float* ap = a.data();
  float* op = out.data();

  const int64_t n = a.numel();
  for (int64_t i = 0; i < n; ++i) {
    op[i] = std::log(ap[i]);
  }
  return out;
}

Tensor clamp(const Tensor& input, float min_val, float max_val) {
  Tensor a = input.contiguous();
  Tensor out(a.sizes());
  const float* ap = a.data();
  float* op = out.data();

  const int64_t n = a.numel();
  for (int64_t i = 0; i < n; ++i) {
    op[i] = std::max(min_val, std::min(max_val, ap[i]));
  }
  return out;
}

// pad a dimension with a given value to a target length
Tensor pad(const Tensor& input, int64_t dim, int64_t target_len, float value) {
  if (dim < 0) dim += input.sizes().size();

  const auto& sizes = input.sizes();
  int64_t ndim = sizes.size();
  int64_t current = sizes[dim];

  if (target_len <= current) return input.contiguous();

  // build output tensor shape
  std::vector<int64_t> out_sizes(sizes.begin(), sizes.end());
  out_sizes[dim] = target_len;

  Tensor out = full(out_sizes, value);
  Tensor c = input.contiguous();
  const float* ap = c.data();
  float* op = out.data();

  int64_t outer = 1, inner = 1;
  for (int64_t i = 0; i < dim; ++i) outer *= sizes[i];
  for (int64_t i = dim + 1; i < ndim; ++i) inner *= sizes[i];

  for (int64_t o = 0; o < outer; ++o) {
    for (int64_t d = 0; d < current; ++d) { // copy up to original len, rest remains as pad value
      for (int64_t i = 0; i < inner; ++i) {
        op[o * target_len * inner + d * inner + i] = ap[o * current * inner + d * inner + i];
      }
    }
  }

  return out;
}

}
