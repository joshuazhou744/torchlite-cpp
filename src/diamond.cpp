#include <tl/diamond.h>
#include <tl/activation.h>
#include <tl/ops.h>
#include <tl/factory.h>

#include <cstdint>
#include <stdexcept>
#include <algorithm>

namespace tl {
namespace diamond {

// Residual block with adaptive group norm
ResidualBlock::ResidualBlock(int64_t in_channels, int64_t out_channels, int64_t cond_dim, bool attn, int64_t gn_group_size, int64_t attn_head_dim)
  : proj_(in_channels, out_channels, 1, 1, 0),
    agn1_(std::max(int64_t(1), in_channels / gn_group_size), in_channels, cond_dim),
    conv1_(in_channels, out_channels, 3, 1, 1),
    agn2_(std::max(int64_t(1), out_channels / gn_group_size), out_channels, cond_dim),
    conv2_(out_channels, out_channels, 3, 1, 1),
    attn_(out_channels, std::max(int64_t(1), out_channels / attn_head_dim)),
    should_proj_(in_channels != out_channels),
    has_attn_(attn)
{
  // zero init conv2 weight
  conv2_.set_weight(zeros(conv2_.weight().sizes()));
}

std::vector<Tensor*> ResidualBlock::parameters() {
  std::vector<Tensor*> params;
  auto append = [&](auto& m) {
    auto p = m.parameters();
    params.insert(params.end(), p.begin(), p.end());
  };
  if (should_proj_) append(proj_);
  append(agn1_);
  append(conv1_);
  append(agn2_);
  append(conv2_);
  if (has_attn_) append(attn_);
  return params;
}

Tensor ResidualBlock::forward(const Tensor& x, const Tensor& cond) const {
  Tensor r = should_proj_ ? proj_.forward(x) : x;
  Tensor h = conv1_.forward(silu(agn1_.forward(x, cond)));
  h = conv2_.forward(silu(agn2_.forward(h, cond)));
  h = add(h, r);
  if (has_attn_) h = attn_.forward(h);
  return h;
}

ResidualBlocks::ResidualBlocks(std::vector<int64_t> in_channels, std::vector<int64_t> out_channels, int64_t cond_dim, bool attn, int64_t gn_group_size, int64_t attn_head_dim) {
  if (in_channels.size() != out_channels.size()) {
    throw std::invalid_argument("ResidualBlocks: in_channels must have same length as out_channels");
  }
  for (int64_t i = 0; i < (int64_t)in_channels.size(); ++i) {
    blocks_.emplace_back(in_channels[i], out_channels[i], cond_dim, attn, gn_group_size, attn_head_dim);
  }
}

std::vector<Tensor*> ResidualBlocks::parameters() {
  std::vector<Tensor*> params;
  for (auto& b: blocks_) {
    auto p = b.parameters();
    params.insert(params.end(), p.begin(), p.end());
  }
  return params;
}

std::pair<Tensor, std::vector<Tensor>> ResidualBlocks::forward(const Tensor& x, const Tensor& cond, const std::vector<Tensor>& to_cat) const {
  Tensor h = x;
  std::vector<Tensor> outputs;
  for (int64_t i = 0; i < (int64_t)blocks_.size(); ++i) {
    h = (i < (int64_t)to_cat.size() && !to_cat[i].empty()) ? cat({h, to_cat[i]}, 1) : h;
    h = blocks_[i].forward(h, cond);
    outputs.push_back(h);
  }
  return {h, outputs};
}

UNet::UNet(int64_t cond_dim, std::vector<int64_t> depths, std::vector<int64_t> channels, std::vector<int64_t> attn_depths, int64_t gn_group_size, int64_t attn_head_dim)
  : num_down_((int64_t)channels.size() - 1)
{
  if (depths.size() != channels.size() || depths.size() != attn_depths.size()) {
    throw std::invalid_argument("UNet: depths, channels, attn_depths must have the same length");
  }

  // down and up blocks
  for (int64_t i = 0; i < (int64_t)depths.size(); ++i) {
    int64_t n = depths[i];
    int64_t c1 = channels[std::max(int64_t(0), i - 1)];
    int64_t c2 = channels[i];
    bool attn = attn_depths[i];

    std::vector<int64_t> d_in, d_out;
    // list_in_channels=[c1] + [c2] * (n-1)
    d_in.push_back(c1);
    for (int64_t j = 0; j < n - 1; ++j) d_in.push_back(c2);
    // list_out_channels=[c2] * n
    for (int64_t j = 0; j < n; ++j) d_out.push_back(c2);
    d_blocks_.emplace_back(d_in, d_out, cond_dim, attn, gn_group_size, attn_head_dim);

    std::vector<int64_t> u_in, u_out;
    // list_in_channels=[2*c2] * n + [c1+c2]
    for (int64_t j = 0; j < n; ++j) u_in.push_back(c2 * 2);
    u_in.push_back(c1 + c2);
    // list_out_channels=[c2] * n + [c1]
    for (int64_t j = 0; j < n; ++j) u_out.push_back(c2);
    u_out.push_back(c1);
    u_blocks_.emplace_back(u_in, u_out, cond_dim, attn, gn_group_size, attn_head_dim);
  }

  // reverse u_blocks order
  std::reverse(u_blocks_.begin(), u_blocks_.end());

  // mid blocks
  int64_t ch = channels.back();
  // list_in_channels=[channels[-1]] * 2
  // list_out_channels=[channels[-1]] * 2
  mid_blocks_ = ResidualBlocks({ch, ch}, {ch, ch}, cond_dim, true, gn_group_size, attn_head_dim);

  // downsamples
  for (int64_t i = 0; i < num_down_; ++i) {
    downsamples_.emplace_back(channels[i]);
  }
  // upsamples
  for (int64_t i = num_down_ - 1; i >= 0; --i) {
    upsamples_.emplace_back(2, channels[i]); // scale_factor = 2
  }
}

std::vector<Tensor*> UNet::parameters() {
  std::vector<Tensor*> params;
  auto append = [&](auto& m) {
    auto p = m.parameters();
    params.insert(params.end(), p.begin(), p.end());
  };
  for (auto& b: d_blocks_) append(b);
  for (auto& b: u_blocks_) append(b);
  append(mid_blocks_);
  for (auto& d: downsamples_) append(d);
  for (auto& u: upsamples_) append(u);
  return params;
}



} // diamond
} // tl
