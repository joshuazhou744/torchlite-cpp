#pragma once

#include <tl/nn.h>
#include <cstdint>
#include <vector>

namespace tl {
namespace diamond {

// ResidualBlock: Conv1x1 -> AdaGN -> SiLU -> Conv -> AdaGN -> SiLU + skip connection
// cond_dim: size of conditioning vector (timestep and action embedding concatenated)
class ResidualBlock {
public:
  ResidualBlock(int64_t in_channels, int64_t out_channels, int64_t cond_dim, bool attn, int64_t gn_group_size = 32, int64_t attn_head_dim = 8);
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
  ResidualBlocks() = default;
  ResidualBlocks(std::vector<int64_t> in_channels, std::vector<int64_t> out_channels, int64_t cond_dim, bool attn, int64_t gn_group_size = 32, int64_t attn_head_dim = 8);
  std::pair<Tensor, std::vector<Tensor>> forward(const Tensor& x, const Tensor& cond, const std::vector<Tensor>& to_cat = {}) const;
  std::vector<Tensor*> parameters();

private:
  std::vector<ResidualBlock> blocks_;
};

// UNet: encoder-bottleneck-decoder with skip connections CNN
// encoder downsamples, decoder upsamples with residuals from encoder outputs
class UNet {
public:
  UNet(int64_t cond_channels, std::vector<int64_t> depths, std::vector<int64_t> channels, std::vector<int64_t> attn_depths, int64_t gn_group_size = 32, int64_t attn_head_dim = 8);
  Tensor forward(const Tensor& x, const Tensor& cond) const;
  std::vector<Tensor*> parameters();
private:
  std::vector<ResidualBlocks> d_blocks_; // encoder ResidualBlocks
  std::vector<ResidualBlocks> u_blocks_; // decoder ResidualBlocks
  ResidualBlocks mid_blocks_; // bottleneck with attention
  std::vector<nn::Downsample> downsamples_;
  std::vector<nn::Upsample> upsamples_;
  int64_t num_down_;
};

} // diamond
} // tl
