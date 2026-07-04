#pragma once

#include "denoiser.h"
#include <tl/tensor.h>
#include <vector>
#include <cstdint>

// config for EDM sigma scheduler and ODE solver
// first-order Euler step only
struct DiffusionSamplerConfig {
  int64_t num_steps_denoising;
  float sigma_min = 2e-3f;
  float sigma_max = 5.0f;
  float rho = 7.0f;
};

// DiffusionSampler: EDM ODE solver
// generates one next frame by starting from pure noise and iteratively denoising it down a precomputed sigma schedule using Euler steps
class DiffusionSampler {
public:
  DiffusionSampler(const Denoiser& denoiser, DiffusionSamplerConfig cfg);
  // generate next frame given previous 4 frames (prev_obs) and their respective actions (prev_act)
  tl::Tensor sample(const tl::Tensor& prev_obs, const tl::Tensor& prev_act) const;

private:
  const Denoiser& denoiser_;
  DiffusionSamplerConfig cfg_;
  std::vector<float> sigmas_;
};
