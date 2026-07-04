#include "sampler.h"
#include <tl/factory.h>
#include <tl/ops.h>
#include <cmath>
#include <cstdint>

// sigma schedule: num_steps values from sigma_max to sigma_min (rho-warped), then a 0 sigma at end
static std::vector<float> build_sigmas(int64_t num_steps, float sigma_min, float sigma_max, float rho) {
  float min_inv_rho = std::pow(sigma_min, 1.0f / rho);
  float max_inv_rho = std::pow(sigma_max, 1.0f / rho);

  std::vector<float> sigmas;
  for (int64_t i = 0; i < num_steps; ++i) {
    float l = (num_steps == 1) ? 0.0f : (float) i / (num_steps - 1); // linspace(0, 1, num_steps)
    float s = std::pow(max_inv_rho + l * (min_inv_rho - max_inv_rho), rho);
    sigmas.push_back(s);
  }
  sigmas.push_back(0.0f); // placeholder for last next_sigma
  return sigmas;
}

DiffusionSampler::DiffusionSampler(const Denoiser& denoiser, DiffusionSamplerConfig cfg) {
  
}


tl::Tensor DiffusionSampler::sample(const tl::Tensor& prev_obs, const tl::Tensor& prev_act) const {

}
