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

  int64_t n = sigma.sizes()[0];
  cs.c_in = reshape(c_in, {n, 1, 1, 1});
  cs.c_skip = reshape(c_skip, {n, 1, 1, 1});
  cs.c_out = reshape(c_out, {n, 1, 1, 1});
  cs.c_noise = c_noise; // stays [N] to match FourierFeatures input shape
  return cs;
}

