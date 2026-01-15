#pragma once

#include <tensor/tl>

namespace tl {

// element-wise addition: out[i] = a[i] + b[i]
Tensor add(const Tensor& a, const Tensor& b);

// element-wise multiplication: out[i] = a[i] * b[i]
Tensor mul(const Tensor& a, const Tensor& b);

// ReLU activation function: out[i] = max(0, out[i])
Tensor relu(const Tensor& a);

}
