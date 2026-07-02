#pragma once

#include <tl/nn.h>
#include <cstdint>
#include <vector>

namespace tl {
namespace models {

// ResidualBlock: Conv1x1 -> AdaGN -> SiLU -> Conv -> AdaGN -> SiLU + skip connection
// cond_dim: size of conditioning vector (timestep and action embedding concatenated)
class ResidualBlock {
public:
  ResidualBlock(int64_t in_channels, int64_t out_channels, int64_t cond_dim, bool attn);
  Tensor forward(const Tensor& x, const Tensor& cond) const;
  std::vector<Tensor*> parameters();

private:
  nn::Conv2d proj_; // 1x1 conv to match channels if input dim doesnt match output dim
  nn::AdaptiveGroupNorm agn1_;
  nn::Conv2d conv1_;
  nn::AdaptiveGroupNorm agn2_;
  nn::Conv2d conv2_;
  nn::SelfAttention2d attn_;
  bool should_proj_;
  bool has_attn_;
};

// ResidualBlocks: sequence of ResidualBlock with optional skip connection concatenation
class ResidualBlocks {
public:
  ResidualBlocks(std::vector<int64_t> in_channels, std::vector<int64_t> out_channels, int64_t cond_dim, bool attn);
  std::pair<Tensor, std::vector<Tensor>> forward(const Tensor& x, const Tensor& cond, const std::vector<Tensor>& to_cat = {}) const;
  std::vector<Tensor*> parameters();

private:
  std::vector<ResidualBlock> blocks_;
};

} // models
} // tl
