#pragma once

#include <tl/diamond.h>
#include <tl/tensor.h>

// four EDM scalars derived from sigma used to rescale InnerModels input/output image tensor
struct Conditioners {
  tl::Tensor c_in;
  tl::Tensor c_out;
  tl::Tensor c_skip;
  tl::Tensor c_noise;
};

// Denoiser config
struct DenoiserConfig {
  tl::diamond::InnerModelConfig inner_model;
  float sigma_data; // natural variation clean image pixel values
  float sigma_offset_noise; // extra noise per-channel before pixel noise (baseline noise)
};

// EDM preconditioning wrapper around InnerModel
// also converts InnerModel output -> denoised frame prediction
class Denoiser {
public:
  Denoiser(tl::diamond::InnerModel& inner_model, DenoiserConfig cfg);
  // single denoising step
  tl::Tensor denoise(const tl::Tensor& noisy_next_obs, const tl::Tensor& sigma, const tl::Tensor& obs, const tl::Tensor& act) const;

private:
  Conditioners compute_conditioners(const tl::Tensor& sigma) const;
  tl::Tensor compute_model_output(const tl::Tensor& noisy_next_obs, const tl::Tensor& obs, const tl::Tensor& act, const Conditioners& cs) const;
  tl::Tensor wrap_model_output(const tl::Tensor& noisy_next_obs, const tl::Tensor& model_output, const Conditioners& cs) const;

  tl::diamond::InnerModel& inner_model_;
  DenoiserConfig cfg_;
};
