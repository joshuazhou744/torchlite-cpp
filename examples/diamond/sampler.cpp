#include "sampler.h"
#include <tl/factory.h>
#include <tl/ops.h>
#include <tl/autograd.h>
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

DiffusionSampler::DiffusionSampler(const Denoiser& denoiser, DiffusionSamplerConfig cfg)
  : denoiser_(denoiser), cfg_(cfg),
    sigmas_(build_sigmas(cfg.num_steps_denoising, cfg.sigma_min, cfg.sigma_max, cfg.rho))
{}


tl::Tensor DiffusionSampler::sample(const tl::Tensor& prev_obs, const tl::Tensor& prev_act) const {
  using namespace tl;
  NoGradGuard no_grad;

  int64_t b = prev_obs.sizes()[0];
  int64_t t = prev_obs.sizes()[1];
  int64_t c = prev_obs.sizes()[2];
  int64_t h = prev_obs.sizes()[3];
  int64_t w = prev_obs.sizes()[4];

  // flatten conditioning frames: [b, t, c, h, w] -> [b, t*c, h, w]
  Tensor obs = reshape(prev_obs, {b, t * c, h, w});

  // start from pure noise
  Tensor x = randn({b, c, h, w});

  int64_t num_pairs = sigmas_.numel() - 1;
  for (int64_t i = 0; i < num_pairs; ++i) {
    float sigma = sigmas_.at(i);
    float next_sigma = sigmas_.at(i+1);

    Tensor sigma_t = full({b}, sigma); // denoiser expects sigma shaped [N]
    Tensor denoised = denoiser_.denoise(x, sigma_t, obs, prev_act);

    // Euler step: d = (x - denoised) / sigma ; x = x + d*(next_sigma - sigma)
    Tensor d = scale(sub(x, denoised), 1.0f / sigma);
    float dt = next_sigma - sigma;
    x = add(x, scale(d, dt));
  }
  return x;
}
