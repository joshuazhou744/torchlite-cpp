#pragma once

#include <tl/tensor.h>

#include <vector>
#include <cstdint>
#include <memory>
#include <functional>

namespace tl {

namespace nn { class Module; } // forward declaration

inline bool& grad_enabled() { static bool e = true; return e; }

// traverse computation graph and clear all grad function pointers to save memory
void release_graph(Tensor& root);

struct NoGradGuard {
  NoGradGuard() { grad_enabled() = false; }
  ~NoGradGuard() { grad_enabled() = true; }
};

// base class for ALL backward functions
class GradFunction {
public:
  virtual ~GradFunction() = default;

  // compute gradients and propagate to inputs
  virtual void backward(const Tensor& grad_output) = 0;

  // inputs that need gradients
  std::vector<Tensor> inputs;
};

class CheckpointBackward: public GradFunction {
public:
  nn::Module* wrapped_; // checkpointed block
  Tensor saved_input; // seed
  void backward(const Tensor& grad_output) override;
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
  int64_t dim_;
  bool keepdim_;
};

class AbsBackward: public GradFunction {
public:
  Tensor input_cache;
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

class Conv2dBackward: public GradFunction {
public:
  Tensor weight_cache;
  Tensor input_cache;
  int64_t stride, padding;
  int64_t groups = 1;
  int64_t N, C_in, H, W;
  void backward(const Tensor& grad_output) override;
};

class MaxPool2dBackward: public GradFunction {
public:
  std::vector<int64_t> argmax_indices; // flat input index per output cell, -1 if window all out of bounds
  int64_t N, C, H, W;
  void backward(const Tensor& grad_output) override;
};

class AvgPool2dBackward: public GradFunction {
public:
  int64_t kernel_size, stride, padding;
  int64_t N, C, H, W;
  void backward(const Tensor& grad_output) override;
};

class DropoutBackward: public GradFunction {
public:
  Tensor mask_cache;
  void backward(const Tensor& grad_output) override;
};

class FlashAttentionBackward: public GradFunction {
public:
  Tensor Q_cache, K_cache, V_cache;
  Tensor O_cache;
  Tensor L_cache;
  float sm_scale;
  void backward(const Tensor& grad_output) override;
};

class CosBackward: public GradFunction {
public:
  Tensor input_cache;
  void backward(const Tensor& grad_output) override;
};

class SinBackward: public GradFunction {
public:
  Tensor input_cache;
  void backward(const Tensor& grad_output) override;
};

// Helper functions

template<typename BackwardFn>
std::shared_ptr<BackwardFn> track(Tensor& out, std::initializer_list<const Tensor*> inputs) {
  if (!grad_enabled()) return nullptr;
  out.requires_grad = true;
  out.ensure_grad();
  auto fn = std::make_shared<BackwardFn>();
  for (auto* t: inputs) {
    const_cast<Tensor*>(t)->ensure_grad();
    fn->inputs.push_back(*const_cast<Tensor*>(t)); // shallow copy, shared grad_
  }
  out.grad_fn = fn;
  return fn;
}

}
