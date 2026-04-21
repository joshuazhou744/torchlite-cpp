#pragma once

#include <tl/tensor.h>

namespace tl {

// mean squared error: mean((x -y)^2)
Tensor mse_loss(const Tensor& pred, const Tensor& target);

// mean absolute error: mean(|x - y|)
Tensor mae_loss(const Tensor& pred, const Tensor& target);

// L2 regularization: lambda * sum(w^2), penalizes large weights
Tensor l2_reg(const std::vector<Tensor*>& params, float lambda);

// L1 regularization: lambda * sum(|w|), promotes sparsity
Tensor l1_reg(const std::vector<Tensor*>& params, float lambda);

// binary cross entropy: -mean(y * log(x) + (1 - y) * log(1 - x))
// x must be in (0, 1), apply sigmoid before calling
Tensor bce_loss(const Tensor& pred, const Tensor& target);

// negative log likelihood: -mean(log_probs[y])
// log_probs: (N, C), y: (N,) class indices
Tensor nll_loss(const Tensor& log_probs, const std::vector<int>& targets);

// cross entropy loss (softmax + nll): -mean(log(softmax(logits))[y])
// logits: (N, C), y: (N,) class indices
Tensor cross_entropy_loss(const Tensor& logits, const std::vector<int>& targets);

} // namespace tl
