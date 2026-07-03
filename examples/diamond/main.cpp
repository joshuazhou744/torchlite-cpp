#include <tl/diamond.h>
#include <tl/model_io.h>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: " << argv[0] << " <path/to/model.tl>" << std::endl;
    return 1;
  }
  std::string model_path = argv[1];

  tl::diamond::InnerModelConfig cfg;
  cfg.img_channels = 3;
  cfg.num_steps_conditioning = 4;
  cfg.cond_dim = 256;
  cfg.depths = {2, 2, 2, 2};
  cfg.channels = {64, 64, 64, 64};
  cfg.attn_depths = {0, 0, 0, 0};
  cfg.num_actions = 4; // Breakout: NOOP, FIRE, RIGHT, LEFT

  tl::diamond::InnerModel model(cfg);

  auto state = model.state();
  std::cout << "InnerModel has " << state.size() << " tensors (params + buffers)" << std::endl;

  tl::load_model(model_path, state);
  std::cout << "Loaded weights successfully!" << std::endl;

  tl::save_model("test.tl", state);

  return 0;
}
