#include "denoiser.h"
#include <tl/ops.h>
#include <tl/factory.h>
#include <cmath>
#include <cstdint>

Denoiser::Denoiser(tl::diamond::InnerModel& inner_model, float sigma_data, float sigma_offset_noise)
  : inner_model_(inner_model), sigma_data_(sigma_data), sigma_offset_noise_(sigma_offset_noise)
{}

Conditioners Denoiser::compute_conditioners(const tl::Tensor& sigma) const {
  using namespace tl;

  // effective sigma: sqrt(sigma^2 + sigma_offset_noise^2)
  Tensor sigma_sq = pow(sigma, 2.0f);
  Tensor off_sq = full({1}, sigma_offset_noise_ * sigma_offset_noise_);
  Tensor sigma_eff = sqrt(add(sigma_sq, off_sq)); // [N]

  Tensor data_sq = full({1}, sigma_data_ * sigma_data_);
  Tensor sigma_eff_sq = pow(sigma_eff, 2.0f);
  Tensor denom = add(sigma_eff_sq, data_sq);

  // scalars, [N]
  Tensor c_in = div(full({1}, 1.0f), sqrt(denom));
  Tensor c_skip = div(data_sq, denom);
  Tensor c_out = mul(sigma_eff, sqrt(c_skip));
  Tensor c_noise = scale(log(sigma_eff), 0.25f);

  Conditioners cs;
  int64_t n = sigma.sizes()[0];
  cs.c_in = reshape(c_in, {n, 1, 1, 1});
  cs.c_skip = reshape(c_skip, {n, 1, 1, 1});
  cs.c_out = reshape(c_out, {n, 1, 1, 1});
  cs.c_noise = c_noise; // stays [N] to match FourierFeatures input shape
  return cs;
}

tl::Tensor Denoiser::compute_model_output(const tl::Tensor& noisy_next_obs, const tl::Tensor& obs, const tl::Tensor& act, const Conditioners& cs) const {
  using namespace tl;
  Tensor rescaled_obs = scale(obs, 1.0f / sigma_data_);
  Tensor rescaled_noise = mul(noisy_next_obs, cs.c_in);
  return inner_model_.forward(rescaled_noise, cs.c_noise, rescaled_obs, act);
}


tl::Tensor Denoiser::wrap_model_output(const tl::Tensor& noisy_next_obs, const tl::Tensor& model_output, const Conditioners& cs) const {
  using namespace tl;

  Tensor d = add(mul(noisy_next_obs, cs.c_skip), mul(model_output, cs.c_out));
  d = clamp(d, -1.0f, 1.0f);

  // quantize to {0, ..., 255} then back to [-1, 1] to simulate 8-bit image precision
  Tensor out = d.contiguous();
  float* p = out.data();
  for (int64_t i = 0; i < out.numel(); ++i) {
    float v = std::floor((p[i] + 1.0f) * 0.5f * 255.0f);
    p[i] = (v / 255.0f) * 2.0f - 1.0f;
  }
  return out;
}

tl::Tensor Denoiser::denoise(const tl::Tensor& noisy_next_obs, const tl::Tensor& sigma, const tl::Tensor& obs, const tl::Tensor& act) const {
  using namespace tl;

  Conditioners cs = compute_conditioners(sigma);
  Tensor model_output = compute_model_output(noisy_next_obs, obs, act, cs);
  return wrap_model_output(noisy_next_obs, model_output, cs);
}
