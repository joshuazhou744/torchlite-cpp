#include <tl/loss.h>
#include <tl/ops.h>
#include <tl/tensor.h>

namespace tl {

Tensor mse_loss(const Tensor& pred, const Tensor& target) {
  Tensor diff = sub(pred, target);
  Tensor sq = mul(diff, diff);
  std::vector<Tensor> steps;
  steps.push_back(sq);
  while (steps.back().sizes().size() > 0) {
    steps.push_back(sum(steps.back(), 0, false));
  }
  return scale(steps.back(), 1.0f / static_cast<float>(sq.numel()));
}

Tensor mae_loss(const Tensor& pred, const Tensor& target) {
  Tensor diff = sub(pred, target);
  Tensor abs_diff = abs(diff);
  std::vector<Tensor> steps;
  steps.push_back(abs_diff);
  while (steps.back().sizes().size() > 0) {
    steps.push_back(sum(steps.back(), 0, false));
  }
  return scale(steps.back(), 1.0f / static_cast<float>(abs_diff.numel()));
}


} // namespace tl

