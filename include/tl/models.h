#pragma once

#include <tl/nn.h>
#include <cstdint>
#include <vector>

namespace tl {
namespace models {

// ResidualBlock: Conv -> AdaGN -> SiLU -> Conv -> AdaGN -> SiLU + skip connection
// cond_dim: size of conditioning vector (timestep and action embedding concatenated)
class ResidualBlock {
public:
  ResidualBlock(int64_t in_channels, int64_t out_channels, int64_t num_groups, int64_t cond_dim);
  Tensor forward(const Tensor& x, const Tensor& cond) const;
  std::vector<Tensor*> parameters();

private:
  nn::Conv2d conv1_;
  nn::Conv2d conv2_;
  nn::AdaptiveGroupNorm agn1_;
  nn::AdaptiveGroupNorm agn2_;
  nn::Conv2d skip_conv_; // 1x1 conv to match channels if input dim doesnt match output dim
  bool needs_skip_conv_;
};

} // models
} // tl
