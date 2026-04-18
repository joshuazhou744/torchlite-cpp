#include <tl/autograd.h>
#include <tl/ops.h>
#include <tl/factory.h>

#include <cstdint>
#include <cmath>

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

void GeluBackward::backward(const Tensor& grad_output) {
  // forward gelu uses tanh approximation:
  // u = sqrt(2/n) * (x + 0.044715 * x^3)
  // btw n = pi
  // gelu(x) = 0.5 * x * (1 + tanh(u))
  Tensor x = input_cache.contiguous();
  Tensor g = grad_output.contiguous();
  Tensor result(x.sizes());
  const float* xp = x.data();
  const float* gp = g.data();
  float* rp = result.data();

  // derivative (product + chain rule)
  // d/dx = 0.5 * (1 + tanh(u)) + 0.5 * x * (1 - tanh^2(u)) * du/dx
  // du/dx = sqrt(2/n) * (1 + 3 * 0.044715 * x^2)
  const float k = std::sqrt(2.0f / M_PI);
  for (int64_t i = 0; i < x.numel(); ++i) {
    float xi = xp[i];
    float u = k * (xi + 0.044715f * xi * xi * xi);
    float t = std::tanh(u);
    float du_dx = k * (1.0f + 3.0f * 0.044715f * xi * xi);
    float d = 0.5f * (1.0f + t) + 0.5f * xi * (1.0f - t * t) * du_dx;
    rp[i] = gp[i] * d;
  }
  accumulate_grad(inputs[0], result);
}

void SoftmaxBackward::backward(const Tensor& grad_output) {
  Tensor out = output_cache.contiguous();
  Tensor g = grad_output.contiguous();
  Tensor result(out.sizes());

  const auto& sizes = out.sizes();
  int64_t last = sizes.back(); // size of final dim (softmax axis)
  int64_t rows = out.numel() / last;

  const float* op = out.data();
  const float* gp = g.data();
  float* rp = result.data();

  for (int64_t r = 0; r < rows; ++r) {
    const float* or_ = op + r * last; // row r of output_cache
    const float* gr = gp + r * last; // row r of grad_output
    float* rr = rp + r * last; // row r of result

    // per-row dot product
    float dot = 0.0f;
    // dot = sum_j(grad_j * out_j)
    for (int64_t j = 0; j < last; ++j) dot += gr[j] * or_[j];

    // apply softmax derivative formula accross last row
    // d_in_i = out_i * (grad_i - sum_j(grad_j * out_j))
    for (int64_t i = 0; i < last; ++i) rr[i] = or_[i] * (gr[i] - dot);
  }
  accumulate_grad(inputs[0], result);
}

void SqrtBackward::backward(const Tensor& grad_output) {
  // d(sqrt(x))/dx = 1 / (2 * sqrt(x)) = 1 / (2 * y)
  Tensor y = output_cache.contiguous();
  Tensor g = grad_output.contiguous();
  Tensor result(y.sizes());

  const float* yp = y.data();
  const float* gp = g.data();
  float* rp = result.data();
  for (int64_t i = 0; i < y.numel(); ++i) {
    // multiply local derivative by grad_output (running gradient)
    rp[i] = gp[i] / (2.0f * yp[i]);
  }
  accumulate_grad(inputs[0], result);
}

void PowBackward::backward(const Tensor& grad_output) {
  // d(x^n)/dx = n * x^(n-1)
  Tensor x = input_cache.contiguous();
  Tensor g = grad_output.contiguous();
  Tensor result(x.sizes());

  const float* xp = x.data();
  const float* gp = g.data();
  float* rp = result.data();
  for (int64_t i = 0; i < x.numel(); ++i) {
    rp[i] = gp[i] * exponent * std::pow(xp[i], exponent - 1.0f);
  }
  accumulate_grad(inputs[0], result);
}

void LogBackward::backward(const Tensor& grad_output) {
  // d(log(x))/dx = 1/x
  Tensor x = input_cache.contiguous();
  Tensor g = grad_output.contiguous();
  Tensor result(x.sizes());

  const float* xp = x.data();
  const float* gp = g.data();
  float* rp = result.data();
  for (int64_t i = 0; i < x.numel(); ++i) {
    rp[i] = gp[i] / xp[i];
  }
  accumulate_grad(inputs[0], result);
}

void ExpBackward::backward(const Tensor& grad_output) {
  // d(exp(x))/dx = exp(x) = y
  Tensor y = output_cache.contiguous();
  Tensor g = grad_output.contiguous();
  Tensor result(y.sizes());

  const float* yp = y.data();
  const float* gp = g.data();
  float* rp = result.data();
  for (int64_t i = 0; i < y.numel(); ++i) {
    rp[i] = gp[i] * yp[i];
  }
  accumulate_grad(inputs[0], result);
}

void ClampBackward::backward(const Tensor& grad_output) {
  // d/dx = 1 if min < x < max else 0
  Tensor x = input_cache.contiguous();
  Tensor g = grad_output.contiguous();
  Tensor result(x.sizes());

  const float* xp = x.data();
  const float* gp = g.data();
  float* rp = result.data();
  for (int64_t i = 0; i < x.numel(); ++i) {
    rp[i] = (xp[i] > min_val && xp[i] < max_val) ? gp[i] : 0.0f;
  }
  accumulate_grad(inputs[0], result);
}


} // tl
