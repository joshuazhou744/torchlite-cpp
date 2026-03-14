#pragma once

#include <tl/tensor.h>
#include <vector>
#include <memory>
#include <functional>

namespace tl {

// base class for ALL backward functions
class GradFunction {
public:
  virtual ~GradFunction() = default;

  // compute gradients and propagate to inputs
  virtual void backward(const Tensor& grad_output) = 0;

  // inputs that need gradients
  std::vector<Tensor*> inputs;
};

class AddBackward: public GradFunction {
public:
  void backward(const Tensor& grad_output) override;
};

}
