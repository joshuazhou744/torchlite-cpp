#pragma once

#include <tl/tensor.h>

namespace tl {

// ReLU activation function: out[i] = max(0, out[i])
Tensor relu(const Tensor& input);

// GeLU activation function: out[i] = 0.5 * x * (1 + tanh(sqrt(2/n) * (x + 0.044715 * x^3)))
Tensor gelu(const Tensor& input);

// Sigmoid activation function: out[i] = 1 / (1 + exp(-in[i]))
Tensor sigmoid(const Tensor& input);

}
