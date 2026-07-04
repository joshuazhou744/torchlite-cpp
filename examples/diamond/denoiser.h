#pragma once

#include <tl/diamond.h>
#include <tl/tensor.h>

// EDM preconditioning wrapper around InnerModel
// also converts InnerModel output -> denoised frame prediction
class Denoiser {
public:
  Denoiser(tl::diamond::InnerModel& inner_model, float sigma_data, float sigma_offset_noise);
  // single denoising step
  tl::Tensor denoise(const tl::Tensor& noisy_next_obs, const tl::Tensor& sigma, const tl::Tensor& obs, const tl::Tensor& act) const;

private:
  tl::diamond::InnerModel& inner_model_;
  float sigma_data_; // natural variation clean image pixel values
  float sigma_offset_noise_; // extra noise per-channel before pixel noise (baseline noise)
};
