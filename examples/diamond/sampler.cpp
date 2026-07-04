#include "sampler.h"
#include <tl/factory.h>
#include <tl/ops.h>
#include <cmath>
#include <cstdint>

// sigma schedule: num_steps values from sigma_max to sigma_min (rho-warped), then a 0 sigma at end
static tl::Tensor build_sigmas(int64_t num_steps, float sigma_min, float sigma_max, float rho) {
  using namespace tl;
  float min_inv_rho = std::pow(sigma_min, 1.0f / rho);
  float max_inv_rho = std::pow(sigma_max, 1.0f / rho);

  // sigmas = (max_inv_rho + linspace(0, 1, num_steps) * (min_inv_rho - max_inv_rho)) ^ rho
  Tensor l = linspace(0.0f, 1.0f, num_steps);
  Tensor warped = add(scale(l, min_inv_rho - max_inv_rho), full({1}, max_inv_rho));
  Tensor sigmas = pow(warped, rho);
  return cat({sigmas, zeros({1})}, 0);
}

DiffusionSampler::DiffusionSampler(const Denoiser& denoiser, DiffusionSamplerConfig cfg) {
  
}


tl::Tensor DiffusionSampler::sample(const tl::Tensor& prev_obs, const tl::Tensor& prev_act) const {

}
