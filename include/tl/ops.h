#pragma once

#include <tl/tensor.h>
#include <cstdint>

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

// matrix reshape
Tensor reshape(const Tensor& a, const std::vector<int64_t>& new_sizes);

// scale tensor by a scalar: out = a * scalar
Tensor scale(const Tensor& input, float scalar);

// softmax along the last dimension of a tensor
Tensor softmax(const Tensor& input);

// tensor sum along dimension
Tensor sum(const Tensor& input, int64_t dim, bool keepdim = false);

// tensor mean along dimension
Tensor mean(const Tensor& input, int64_t dim, bool keepdim = false);

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
}
