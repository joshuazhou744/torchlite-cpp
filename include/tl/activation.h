#pragma once

#include <tl/tensor.h>

namespace tl {

// ReLU activation function: out[i] = max(0, x)
Tensor relu(const Tensor& input);

// GeLU activation function: out[i] = 0.5 * x * (1 + tanh(sqrt(2/n) * (x + 0.044715 * x^3)))
Tensor gelu(const Tensor& input);

// Sigmoid activation function: out[i] = 1 / (1 + exp(-x))
Tensor sigmoid(const Tensor& input);

// SiLU activation function: out[i] = x * sigmoid(x)
Tensor silu(const Tensor& input);

// Tanh activation function: out[i] = tanh(x)
Tensor tanh(const Tensor& input);

}
