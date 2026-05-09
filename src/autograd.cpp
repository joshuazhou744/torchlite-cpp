#include <tl/autograd.h>
#include <tl/ops.h>
#include <tl/factory.h>

#include <cstdint>
#include <cmath>
#include <unordered_set>
#include <vector>

namespace tl {

// Helper functions

static void accumulate_grad(Tensor& t, const Tensor& grad) {
  if (!t.requires_grad) return;
  if (t.grad().empty()) {
    t.grad() = zeros(t.sizes());
  }
  t.grad() = add(t.grad(), grad);
}

static Tensor sum_to(Tensor grad, const std::vector<int64_t>& target_shape) {
  while (grad.sizes().size() > target_shape.size()) {
    grad = sum(grad, 0, false);
  }
  return grad;
}

// Graph utility

void release_graph(Tensor& root) {
  if (!root.grad_fn) return;

  std::vector<std::shared_ptr<GradFunction>> stack;
  std::unordered_set<GradFunction*> visited;

  stack.push_back(root.grad_fn);
  visited.insert(root.grad_fn.get()); // track raw grad function pointer

  while (!stack.empty()) {
    auto fn = stack.back();
    stack.pop_back();

    for (auto& input: fn->inputs) {
      if (input.grad_fn && !visited.count(input.grad_fn.get())) {
        visited.insert(input.grad_fn.get());
        stack.push_back(input.grad_fn);
      }
      input.grad_fn = nullptr;
    }
  }

  root.grad_fn = nullptr;
}


// Backward functions

void AddBackward::backward(const Tensor& grad_output) {
  accumulate_grad(inputs[0], sum_to(grad_output, inputs[0].sizes()));
  accumulate_grad(inputs[1], sum_to(grad_output, inputs[1].sizes()));
}

void SubBackward::backward(const Tensor& grad_output) {
  accumulate_grad(inputs[0], sum_to(grad_output, inputs[0].sizes()));
  accumulate_grad(inputs[1], neg(sum_to(grad_output, inputs[1].sizes())));
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

void AbsBackward::backward(const Tensor& grad_output) {
  Tensor x = input_cache.contiguous();
  Tensor g = grad_output.contiguous();
  Tensor result(x.sizes());

  const float* xp = x.data();
  const float* gp = g.data();
  float* rp = result.data();
  for (int64_t i = 0; i < x.numel(); ++i) {
    rp[i] = xp[i] > 0.0f ? gp[i] : (xp[i] < 0.0f ? -gp[i]: 0.0f);
  }
  accumulate_grad(inputs[0], result);
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

void MatmulBackward::backward(const Tensor& grad_output) {
  // A: (M, K), B:(K, N), C: (M, N)
  // dA = grad @ B^T
  // dB = A^T @ grad

  // Tensor grad = grad_output;
  // if (squeeze_a) grad = reshape(grad, {1, grad.sizes()[0]});
  // if (squeeze_b) grad = reshape(grad, {grad.sizes()[0], 1});

  // grad_output comes squeezed, unsqueeze it
  Tensor grad = grad_output;
  if (squeeze_a) {
    // (..., N) -> (..., 1, N)
    auto s = std::vector<int64_t>(grad.sizes().begin(), grad.sizes().end());
    s.insert(s.end() - 1, 1);
    grad = reshape(grad, s);
  }
  if (squeeze_b) {
    // (..., M) -> (..., M, 1)
    auto s = std::vector<int64_t>(grad.sizes().begin(), grad.sizes().end());
    s.push_back(1);
    grad = reshape(grad, s);
  }

  // reverse squeeze handling
  Tensor a = squeeze_a ? reshape(a_cache, {1, a_cache.sizes()[0]}) : a_cache;
  Tensor b = squeeze_b ? reshape(b_cache, {b_cache.sizes()[0], 1}) : b_cache;

  // transpose each input
  Tensor at = transpose(a, -2, -1);
  Tensor bt = transpose(b, -2, -1);

  // compute derivatives with batch handling
  Tensor da = sum_to(matmul(grad, bt), a.sizes());
  Tensor db = sum_to(matmul(at, grad), b.sizes());

  // reverse unsqueeze so grads match cached input sizes
  if (squeeze_a) da = reshape(da, a_cache.sizes());
  if (squeeze_b) db = reshape(db, b_cache.sizes());

  // accumulate gradients
  accumulate_grad(inputs[0], da);
  accumulate_grad(inputs[1], db);
}


// flipped convolution (backward of convolution is another convolution)
void Conv2dBackward::backward(const Tensor& grad_output) {
  const int64_t C_out = grad_output.sizes()[1];
  const int64_t H_out = grad_output.sizes()[2];
  const int64_t W_out = grad_output.sizes()[3];
  const int64_t kH = weight_cache.sizes()[2];
  const int64_t kW = weight_cache.sizes()[3];
  const int64_t CkHkW = C_in * kH * kW;

  Tensor go = grad_output.contiguous();
  const float* gop = go.data();

  Tensor w = reshape(weight_cache, {C_out, CkHkW});
  Tensor grad_w = zeros({C_out, CkHkW});
  Tensor grad_input = zeros({N, C_in, H, W});
  Tensor grad_bias = zeros({C_out});

  float* gi = grad_input.data();
  float* gb = grad_bias.data();

  for (int64_t n = 0; n < N; ++n) {
    // grad_out_n: (C_out, H_out*W_out)
    Tensor grad_out_n = reshape(slice(go, 0, n, n+1), {C_out, H_out*W_out});
    // col_n: (CkHkW, H_out*W_out)
    Tensor col_n = reshape(slice(col_cache, 0, n, n+1), {CkHkW, H_out*W_out});
    // grad_weight += grad_out_n @ col_n.T
    grad_w = add(grad_w, matmul(grad_out_n, transpose(col_n, 0, 1)));
    // grad_col_n = w.T @ grad_out_n: (CkHkW, H_out*W_out)
    Tensor grad_col_n = matmul(transpose(w, 0, 1), grad_out_n);
    const float* gcp = grad_col_n.data();

    // col2im: scatter grad_col_n back to grad_input
    for (int64_t c = 0; c < C_in; ++c) {
      for (int64_t kh = 0; kh < kH; ++kh) {
        for (int64_t kw = 0; kw < kW; ++kw) {
          for (int64_t oh = 0; oh < H_out; ++oh) {
            for (int64_t ow = 0; ow < W_out; ++ow) {
              int64_t ih = oh * stride - padding + kh;
              int64_t iw = ow * stride - padding + kw;
              if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue; // out of bounds

              int64_t col_row = c * kH * kW + kh * kW + kw;
              int64_t col_col = oh * W_out + ow;
              int64_t col_i = col_row * (H_out * W_out) + col_col;
              int64_t in_i = n * (C_in * H * W) + c * (H * W) + ih * W + iw;

              gi[in_i] += gcp[col_i];
            }
          }
        }
      }
    }
  }

  // grad_bias: sum grad_output over N, H_out, W_out
  for (int64_t n = 0; n < N; ++n) {
    for (int64_t c = 0; c < C_out; ++c) {
      for (int64_t oh = 0; oh < H_out; ++oh) {
        for (int64_t ow = 0; ow < W_out; ++ow) {
          gb[c] += gop[n*(C_out*H_out*W_out) + c*(H_out*W_out) + oh*W_out + ow];
        }
      }
    }
  }


  accumulate_grad(inputs[0], grad_input);
  accumulate_grad(inputs[1], reshape(grad_w, weight_cache.sizes()));
  if (inputs.size() > 2) accumulate_grad(inputs[2], grad_bias);
}


} // tl
