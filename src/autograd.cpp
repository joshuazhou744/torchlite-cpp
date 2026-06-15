#include <tl/autograd.h>
#include <tl/ops.h>
#include <tl/factory.h>

#include <cstdint>
#include <algorithm>
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
  // reduce ndim by summing leading dims not in target
  while (grad.sizes().size() > target_shape.size()) {
    grad = sum(grad, 0, false);
  }
  // for each remaining dim where target > 1 but grad > 1, sum along that dim
  for (size_t i = 0; i < target_shape.size(); ++i) {
    if (target_shape[i] == 1 && grad.sizes()[i] != 1) {
      grad = sum(grad, static_cast<int64_t>(i), true);
    }
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
  Tensor go = grad_output;
  // summed dim dropped, put it back as size-1
  if (!keepdim_) {
    std::vector<int64_t> kd_shape = go.sizes();
    kd_shape.insert(kd_shape.begin() + dim_, 1);
    go = reshape(go, kd_shape);
  }
  Tensor expanded = full(input_shape, 0.0f);
  Tensor grad = add(expanded, go);
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
  accumulate_grad(inputs[0], sum_to(mul(grad_output, b_cache), inputs[0].sizes()));
  accumulate_grad(inputs[1], sum_to(mul(grad_output, a_cache), inputs[1].sizes()));
}

void DivBackward::backward(const Tensor& grad_output) {
  // z = a / b
  // d_a = grad / b
  accumulate_grad(inputs[0], sum_to(div(grad_output, b_cache), inputs[0].sizes()));

  // d_b = -grad * a / (b^2)
  Tensor b_sq = mul(b_cache, b_cache);
  Tensor num = mul(grad_output, a_cache);
  Tensor grad_b = neg(div(num, b_sq));
  accumulate_grad(inputs[1], sum_to(grad_b, inputs[1].sizes()));
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
  const int64_t C_in_g = C_in / groups;
  const int64_t C_out_g = C_out / groups;
  const int64_t patch = C_in_g * kH * kW;

  Tensor go = grad_output.contiguous();
  const float* gop = go.data();

  Tensor w = reshape(weight_cache, {C_out, patch});
  Tensor grad_w = zeros({C_out, patch});
  Tensor grad_input = zeros({N, C_in, H, W});
  Tensor grad_bias = zeros({C_out});

  float* gi = grad_input.data();
  float* gb = grad_bias.data();

  // pre-slice w by groups
  std::vector<Tensor> w_groups;
  w_groups.reserve(groups);
  for (int64_t g = 0; g < groups; ++g) {
    w_groups.push_back(slice(w, 0, g * C_out_g, (g + 1) * C_out_g));
  }
  float* gwp = grad_w.data();

  for (int64_t n = 0; n < N; ++n) {
    // grad_out_n: (C_out, H_out*W_out)
    Tensor grad_out_n = reshape(slice(go, 0, n, n+1), {C_out, H_out*W_out});
    for (int64_t g = 0; g < groups; ++g) {
      // slice grad_out for this group: (C_out_g, H_out*W_out)
      Tensor grad_out_ng = slice(grad_out_n, 0, g * C_out_g, (g + 1) * C_out_g);

      // recompute col for image n, group g: (patch, H_out * W_out)
      Tensor col_ng({patch, H_out * W_out});
      float* cp = col_ng.data();
      const float* ip = input_cache.data();
      for (int64_t c = 0; c < C_in_g; ++c) {
        int64_t c_actual = g * C_in_g + c;
        const float* in_ch = ip + n * (C_in * H * W) + c_actual * (H * W);
        for (int64_t kh = 0; kh < kH; ++kh) {
          for (int64_t kw = 0; kw < kW; ++kw) {
            int64_t col_row = c * kH * kW + kh * kW + kw;
            float* col_row_ptr = cp + col_row * (H_out * W_out);

            int64_t ow_start = 0;
            while (ow_start < W_out && (ow_start * stride - padding + kw) < 0) ++ow_start;
            int64_t ow_end = W_out;
            while (ow_end > 0 && ((ow_end - 1) * stride - padding + kw) >= W) --ow_end;
            for (int64_t oh = 0; oh < H_out; ++oh) {
              int64_t ih = oh * stride - padding + kh;
              float* out_row = col_row_ptr + oh * W_out;

              if (ih < 0 || ih >= H) {
                std::fill(out_row, out_row + W_out, 0.0f);
                continue;
              }

              const float* in_row = in_ch + ih * W;

              std::fill(out_row, out_row + ow_start, 0.0f);
              std::fill(out_row + ow_end, out_row + W_out, 0.0f);

              for (int64_t ow = ow_start; ow < ow_end; ++ow) {
                int64_t iw = ow * stride - padding + kw;
                out_row[ow] = in_row[iw];
              }
            }
          }
        }
      }

      // grad_w_g += grad_out_ng @ col_ng.T: (C_out_g, patch)
      Tensor grad_w_g = matmul(grad_out_ng, transpose(col_ng, 0, 1));
      // accumulate into the right row band of grad_w
      const float* gwgp = grad_w_g.data();
      for (int64_t r = 0; r < C_out_g; ++r) {
        for (int64_t i = 0; i < patch; ++i) {
          gwp[(g * C_out_g + r) * patch + i] += gwgp[r * patch + i];
        }
      }

      // grad_col_ng = w_g.T @ grad_out_ng: (patch, H_out*W_out)
      Tensor grad_col_ng = matmul(transpose(w_groups[g], 0, 1), grad_out_ng);
      const float* gcp = grad_col_ng.data();

      // col2im: scatter grad_col_ng back to grad_input
      for (int64_t c = 0; c < C_in_g; ++c) {
        int64_t c_actual = g * C_in_g + c;
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
                int64_t in_i = n * (C_in * H * W) + c_actual * (H * W) + ih * W + iw;

                gi[in_i] += gcp[col_i];
              }
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

// maximum 2D pooling
void MaxPool2dBackward::backward(const Tensor& grad_output) {
  // input gradient is zero everywhere except at cached argmax indices
  Tensor input_grad = zeros({N, C, H, W});
  float* igp = input_grad.data();
  const float* gop = grad_output.data();

  for (int64_t i = 0; i < (int64_t)argmax_indices.size(); ++i) {
    int64_t max_index = argmax_indices[i];
    if (max_index < 0) continue; // entirely padded window
    igp[max_index] += gop[i];
  }
  accumulate_grad(inputs[0], input_grad);
}

// average 2D pooling
void AvgPool2dBackward::backward(const Tensor& grad_output) {
  // every input cell in window contributes 1 / k^2
  // each cell gets grad_output / k^2
  Tensor input_grad = zeros({N, C, H, W});
  float* igp = input_grad.data();
  const float* gop = grad_output.data();

  const int64_t H_out = grad_output.sizes()[2];
  const int64_t W_out = grad_output.sizes()[3];
  const float window_area = static_cast<float> (kernel_size * kernel_size);

  for (int64_t n = 0; n < N; ++n) {
    for (int64_t c = 0; c < C; ++c) {
      for (int64_t oh = 0; oh < H_out; ++oh) {
        for (int64_t ow = 0; ow < W_out; ++ow) {
          int64_t ih_start = oh * stride - padding;
          int64_t iw_start = ow * stride - padding;

          float spread = gop[n*(C*H_out*W_out) + c*(H_out*W_out) + oh*W_out + ow] / window_area;

          for (int64_t kh = 0; kh < kernel_size; ++kh) {
            for (int64_t kw = 0; kw < kernel_size; ++kw) {
              int64_t ih = ih_start + kh;
              int64_t iw = iw_start + kw;
              if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;
              igp[n*(C*H*W) + c*(H*W) + ih*W + iw] += spread;
            }
          }
        }
      }
    }
  }
  accumulate_grad(inputs[0], input_grad);
}

void DropoutBackward::backward(const Tensor& grad_output) {
  // dropout in forward: y = x * mask_scale (0s and scale mask)
  // dy/dx = mask_scale (element-wise)
  accumulate_grad(inputs[0], mul(grad_output, mask_cache));
}

} // tl
