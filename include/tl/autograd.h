#pragma once

#include <tl/tensor.h>

#include <vector>
#include <cstdint>
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

class SubBackward: public GradFunction {
public:
  void backward(const Tensor& grad_output) override;
};

class NegBackward: public GradFunction {
public:
  void backward(const Tensor& grad_output) override;
};

class ScaleBackward: public GradFunction {
public:
  float scalar;
  void backward(const Tensor& grad_output) override;
};

class ReluBackward: public GradFunction {
public:
  Tensor input_cache; // remember where input > 0
  void backward(const Tensor& grad_output) override;
};

class SumBackward: public GradFunction {
public:
  std::vector<int64_t> input_shape;
  void backward(const Tensor& grad_output) override;
};

class ReshapeBackward: public GradFunction {
public:
  std::vector<int64_t> input_shape;
  void backward(const Tensor& grad_output) override;
};

class TransposeBackward: public GradFunction {
public:
  int64_t dim0, dim1;
  void backward(const Tensor& grad_output) override;
};

class MulBackward: public GradFunction {
public:
  Tensor a_cache, b_cache; // need both inputs for gradient
  void backward(const Tensor& grad_output) override;
};

class DivBackward: public GradFunction {
public:
  Tensor a_cache, b_cache;
  void backward(const Tensor& grad_output) override;
};

class SigmoidBackward: public GradFunction {
public:
  Tensor output_cache; // sigmoid output, not input
  void backward(const Tensor& grad_output) override;
};

class GeluBackward: public GradFunction {
public:
  Tensor input_cache;
  void backward(const Tensor& grad_output) override;
};

class SoftmaxBackward: public GradFunction {
public:
  Tensor output_cache;
  void backward(const Tensor& grad_output) override;
};

class SqrtBackward: public GradFunction {
public:
  Tensor output_cache; // cache output for 1/(2*y)
  void backward(const Tensor& grad_output) override;
};

class PowBackward: public GradFunction {
public:
  Tensor input_cache;
  float exponent;
  void backward(const Tensor& grad_output) override;
};

class LogBackward: public GradFunction {
public:
  Tensor input_cache;
  void backward(const Tensor& grad_output) override;
};

class ExpBackward: public GradFunction {
public:
  Tensor output_cache; // derivative is the output itself
  void backward(const Tensor& grad_output) override;
};

class ClampBackward: public GradFunction {
public:
  Tensor input_cache;
  float min_val, max_val;
  void backward(const Tensor& grad_output) override;
};

class MatmulBackward: public GradFunction {
public:
  Tensor a_cache, b_cache;
  bool squeeze_a = false, squeeze_b = false;
  void backward(const Tensor& grad_output) override;
};

// Helper functions

template<typename BackwardFn>
std::shared_ptr<BackwardFn> track(Tensor& out, std::initializer_list<const Tensor*> inputs) {
  out.requires_grad = true;
  auto fn = std::make_shared<BackwardFn>();
  for (auto* t: inputs) {
    fn->inputs.push_back(const_cast<Tensor*>(t));
  }
  out.grad_fn = fn;
  return fn;
}

}
