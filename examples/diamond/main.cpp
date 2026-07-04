#include <tl/diamond.h>
#include <tl/model_io.h>
#include <tl/factory.h>
#include "denoiser.h"
#include "sampler.h"
#include <iostream>
#include <string>

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: " << argv[0] << " <path/to/model.tl>" << std::endl;
    return 1;
  }
  std::string model_path = argv[1];

  DenoiserConfig cfg;
  cfg.inner_model.img_channels = 3;
  cfg.inner_model.num_steps_conditioning = 4;
  cfg.inner_model.cond_dim = 256;
  cfg.inner_model.depths = {2, 2, 2, 2};
  cfg.inner_model.channels = {64, 64, 64, 64};
  cfg.inner_model.attn_depths = {0, 0, 0, 0};
  cfg.inner_model.num_actions = 4; // Breakout: NOOP, FIRE, RIGHT, LEFT
  cfg.sigma_data = 0.5f;
  cfg.sigma_offset_noise = 0.3f;

  // build inner model
  tl::diamond::InnerModel model(cfg.inner_model);
  auto state = model.state();
  tl::load_model(model_path, state);
  std::cout << "Loaded " << state.size() << " tensors" << std::endl;

  // wrap in denoiser and sampler
  Denoiser denoiser(model, cfg);
  DiffusionSamplerConfig scfg;
  scfg.num_steps_denoising = 3;
  DiffusionSampler sampler(denoiser, scfg);

  // generate a frame
  //tl::Tensor next_frame = sampler.sample(prev_obs, prev_act);

  return 0;
}
