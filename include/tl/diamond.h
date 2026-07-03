#pragma once

#include <tl/nn.h>
#include <cstdint>
#include <vector>
#include <optional>

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
  UNet(int64_t cond_dim, std::vector<int64_t> depths, std::vector<int64_t> channels, std::vector<int64_t> attn_depths, int64_t gn_group_size = 32, int64_t attn_head_dim = 8);
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

struct InnerModelConfig {
  int64_t img_channels;
  int64_t num_steps_conditioning;
  int64_t cond_dim;
  std::vector<int64_t> depths;
  std::vector<int64_t> channels;
  std::vector<int64_t> attn_depths;
  std::optional<int64_t> num_actions;
};

// InnerModel: full DIAMOND denoising network
// wraps UNet with noise/action conditioning, input and output projections
class InnerModel {
public:
  InnerModel(InnerModelConfig cfg);
  Tensor forward(
      const Tensor& noisy_next_obs, // noisy version of next frame: [N, 3, H, W]
      const Tensor& c_noise, // scalar noise conditioning: [N]
      const Tensor& obs, // previous 4 frames: [N, 12, H, W]
      const Tensor& act // previous frame actions: [N, 4]
) const;
  std::vector<Tensor*> parameters();

private:
  nn::FourierFeatures noise_emb_; // encodes scalar noise level to a cond_dim shaped noise embedding
  nn::Embedding act_emb_; // looks up action embeddings
  nn::Flatten flatten_; // flatten action embedding to cond_dim
  nn::Linear cond_proj1_; // first linear projection of conditioning tensor
  nn::SiLU silu_; // activation beween linear layers that mix noise and action embeddings
  nn::Linear cond_proj2_; // second linear projection of conditioning tensor
  nn::Conv2d conv_in_; // project input frames to UNet channel dim
  UNet unet_; // encoder-bottleneck-decoder
  nn::GroupNorm norm_out_; // group norm before output projection
  nn::Conv2d conv_out_; // projects UNet output to pixel space, input channels -> img_channels
};

} // diamond
} // tl
