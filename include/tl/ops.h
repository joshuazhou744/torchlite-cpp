#pragma once

#include <tl/tensor.h>
#include <cmath> // for exp() in sigmoid activation function
#include <cstdint>

namespace tl {

// element-wise addition: out[i] = a[i] + b[i]
Tensor add(const Tensor& a, const Tensor& b);

// element-wise multiplication: out[i] = a[i] * b[i]
Tensor mul(const Tensor& a, const Tensor& b);

// matrix multiplication: out = a @ b
// a: (N x M), b: (M x K), out: (N x K)
Tensor matmul(const Tensor& a_in, const Tensor& b_in);

// matrix transpose
Tensor transpose(const Tensor& a, int64_t dim0, int64_t dim1);

// matrix reshape
Tensor reshape(const Tensor& a, const std::vector<int64_t>& new_sizes);

// ReLU activation function: out[i] = max(0, out[i])
Tensor relu(const Tensor& input);

// Sigmoid activation function: out[i] = 1 / (1 + exp(-in[i]))
Tensor sigmoid(const Tensor& input);

// scale tensor by a scalar: out = a * scalar
Tensor scale(const Tensor& input, float scalar);

// softmax along the last dimension of a tensor
Tensor softmax(const Tensor& input);

// tensor sum along dimension
Tensor sum(const Tensor& input, int64_t dim, bool keepdim = false);

// tensor mean along dimension
Tensor mean(const Tensor& input, int64_t dim, bool keepdim = false);
}
