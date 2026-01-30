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
Tensor matmul(const Tensor& a, const Tensor& b);

// matrix transpose
Tensor transpose(const Tensor& a, int64_t dim0, int64_t dim1);

// ReLU activation function: out[i] = max(0, out[i])
Tensor relu(const Tensor& a);

// Sigmoid activation function: out[i] = 1 / (1 + exp(-in[i]))
Tensor sigmoid(const Tensor& a);

}
