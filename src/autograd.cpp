#include <tl/autograd.h>
#include <tl/ops.h>
#include <tl/factory.h>

#include <cstdint>

namespace tl {

// Helper functions

static void accumulate_grad(Tensor* t, const Tensor& grad) {
  if (!t->requires_grad) return;
  if (t->grad().empty()) {
    t->grad() = zeros(t->sizes());
  }
  t->grad() = add(t->grad(), grad);
}

void AddBackward::backward(const Tensor& grad_output) {
  accumulate_grad(inputs[0], grad_output);
  accumulate_grad(inputs[1], grad_output);
}

void SubBackward::backward(const Tensor& grad_output) {
  accumulate_grad(inputs[0], grad_output);
  accumulate_grad(inputs[1], neg(grad_output));
}

void NegBackward::backward(const Tensor& grad_output) {
  accumulate_grad(inputs[0], neg(grad_output));
}

void ScaleBackward::backward(const Tensor& grad_output) {
  accumulate_grad(inputs[0], scale(grad_output, scalar));
}

void ReluBackward::backward(const Tensor& grad_output) {
  // gradient passes through where input > 0, zero elsewhere
  Tensor mask = input_cache.contiguous(); // original tensor
  Tensor grad = grad_output.contiguous(); // incoming downstream gradients
  Tensor out(grad.sizes());
  float* op = out.data();
  const float* gp = grad.data();
  const float* mp = mask.data();
  for (int64_t i = 0; i < out.numel(); ++i) {
    op[i] = mp[i] > 0.0f ? gp[i] : 0.0f;
  }
  accumulate_grad(inputs[0], out);
}

void SumBackward::backward(const Tensor& grad_output) {
  Tensor expanded = full(input_shape, 0.0f);
  Tensor grad = add(expanded, grad_output);
  accumulate_grad(inputs[0], grad);
}

void MeanBackward::backward(const Tensor& grad_output) {
  Tensor expanded = full(input_shape, 0.0f);
  Tensor grad = add(expanded, grad_output);
  grad = scale(grad, 1.0f / static_cast<float>(dim_size));
  accumulate_grad(inputs[0], grad);
}

void ReshapeBackward::backward(const Tensor& grad_output) {
  accumulate_grad(inputs[0], reshape(grad_output, input_shape));
}

void TransposeBackward::backward(const Tensor& grad_output) {
  accumulate_grad(inputs[0], transpose(grad_output, dim0, dim1));
}

void MulBackward::backward(const Tensor& grad_output) {
  accumulate_grad(inputs[0], mul(grad_output, b_cache));
  accumulate_grad(inputs[1], mul(grad_output, a_cache));
}

void DivBackward::backward(const Tensor& grad_output) {
  // z = a / b
  // d_a = grad / b
  accumulate_grad(inputs[0], div(grad_output, b_cache));

  // d_b = -grad * a / (b^2)
  Tensor b_sq = mul(b_cache, b_cache);
  Tensor num = mul(grad_output, a_cache);
  Tensor grad_b = neg(div(num, b_sq));
  accumulate_grad(inputs[1], grad_b);
}

void SigmoidBackward::backward(const Tensor& grad_output) {
  // d_in = grad * out * (1 - out)
  Tensor out = output_cache.contiguous();
  Tensor g = grad_output.contiguous();
  Tensor result(out.sizes());
  const float* op = out.data();
  const float* gp = g.data();
  float* rp = result.data();
  for (int64_t i = 0; i < out.numel(); ++i) {
    rp[i] = gp[i] * op[i] * (1.0f - op[i]);
  }
  accumulate_grad(inputs[0], result);
}

} // tl
