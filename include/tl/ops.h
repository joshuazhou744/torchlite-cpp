#pragma once

#include <tl/tensor.h>
#include <cstdint>
#include <utility>

namespace tl {

// element-wise addition: out[i] = a[i] + b[i]
Tensor add(const Tensor& a, const Tensor& b);

// element-wise subtraction: out[i] = a[i] - b[i]
Tensor sub(const Tensor& a, const Tensor& b);

// element-wise multiplication: out[i] = a[i] * b[i]
Tensor mul(const Tensor& a, const Tensor& b);

// element-wise division: out[i] = a[i] / b[i]
Tensor div(const Tensor& a, const Tensor& b);

// element-wise square root: out[i] = sqrt(in[i])
Tensor sqrt(const Tensor& input);

// matrix multiplication: out = a @ b
// a: (N x M), b: (M x K), out: (N x K)
Tensor matmul(const Tensor& a_in, const Tensor& b_in);

// matrix transpose
Tensor transpose(const Tensor& a, int64_t dim0, int64_t dim1);

// tensor reshape
Tensor reshape(const Tensor& a, const std::vector<int64_t>& new_sizes);

// tensor concatenation along existing dimension
Tensor cat(const std::vector<Tensor>& tensors, int64_t dim);

// tensor stacking along a new dimension
Tensor stack(const std::vector<Tensor>& tensors, int64_t dim = 0);

// tensor slicing along a given dimension
Tensor slice(const Tensor& input, int64_t dim, int64_t start, int64_t end);

// scale tensor by a scalar: out = a * scalar
Tensor scale(const Tensor& input, float scalar);

// softmax along the last dimension of a tensor
Tensor softmax(const Tensor& input);

// flash attention (forward): out = softmax(Q @ K^T * sm_scale) @ V
Tensor flash_attention(const Tensor& Q, const Tensor& K, const Tensor& V, float sm_scale);

// returns index of maximum value along a given dimension
Tensor argmax(const Tensor& input, int64_t dim);

// tensor sum along dimension
Tensor sum(const Tensor& input, int64_t dim, bool keepdim = false);

// tensor mean along dimension
Tensor mean(const Tensor& input, int64_t dim, bool keepdim = false);

// tensor absolution value along a dimension
Tensor abs(const Tensor& input);

// tensor variance along dimension
Tensor variance(const Tensor& input, int64_t dim, bool keepdim = false);

// unary negation: out[i] = -in[i]
Tensor neg(const Tensor& input);

// unary exponentiation: out[i] = e^in[i]
Tensor exp(const Tensor& input);

// unary natural logarithm: out[i] = ln(in[i])
Tensor log(const Tensor& input);

// unary power exponentiation: out[i] = in[i]^x
Tensor pow(const Tensor& input, float x);

// unary clamp: out[i] = max(min_val, min(max_val, in[i]))
Tensor clamp(const Tensor& input, float min_val, float max_val);

// pad: add values to a tensor along a dimension to a target length using a given value
Tensor pad(const Tensor& input, int64_t dim, int64_t target_len, float value = 0.0f);

// conv2d: slide (C_out, C_in, kH, kW) filters (kernel) over (N, C_in, H, W) input to produce (N, C_out, H_out, W_out)
Tensor conv2d(const Tensor& input, const Tensor& weight, const Tensor& bias, int64_t stride = 1, int64_t padding = 0, int64_t groups = 1);

// max pooling: take maximum over each window sized (kernel_size, kernel_size)
Tensor max_pool2d(const Tensor& input, int64_t kernel_size, int64_t stride = 0, int64_t padding = 0);

// avg pooling: take average over each window sized (kernel_size, kernel_size)
Tensor avg_pool2d(const Tensor& input, int64_t kernel_size, int64_t stride = 0, int64_t padding = 0);

// cos: take cosine of entire tensor
Tensor cos(const Tensor& input);

// sin: take sine of entire tensor
Tensor sin(const Tensor& input);

// RoPE frequencies for positions [0, len): returns {cos, sin}, each [len, dim]
// interleaved pair: channels (2i, 2i+1) form one rotation pair
std::pair<Tensor, Tensor> rope_cos_sin(const Tensor& positions, int64_t dim, float theta);

// axial 2D RoPE over an h x w grid: first half of dim encodes row, second encodes col
// returns {cos, sin}: [h*w, dim] each
std::pair<Tensor, Tensor> rope_cos_sin_2d(int64_t h, int64_t w, int64_t dim, float theta);

// rotate x by RoPE angles: out = x * cos + rotate_half(x) * sin
// x: [..., T, dim] with cos/sin [T, dim] broadcasting over leading dims
Tensor apply_rotary(const Tensor& x, const Tensor& cos, const Tensor& sin);

}
