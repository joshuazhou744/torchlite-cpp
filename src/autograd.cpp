#include <tl/autograd.h>
#include <tl/ops.h>
#include <tl/factory.h>

namespace tl {

void AddBackward::backward(const Tensor& grad_output) {
  if (inputs[0]->requires_grad) {
    if (inputs[0]->grad().empty()) {
      inputs[0]->grad() = zeros(inputs[0]->sizes());
    }
    inputs[0]->grad() = add(inputs[0]->grad(), grad_output);
  }

  if (inputs[1]->requires_grad) {
    if (inputs[1]->grad().empty()) {
      inputs[1]->grad() = zeros(inputs[1]->sizes());
    }
    inputs[1]->grad() = add(inputs[1]->grad(), grad_output);
  }
}

} // tl
