#include <tl/models.h>
#include <tl/activation.h>
#include <tl/ops.h>
#include <tl/factory.h>

#include <cstdint>

namespace tl {
namespace models {

static constexpr int64_t GN_GROUP_SIZE = 32;
static constexpr int64_t ATTN_HEAD_DIM = 8;

// Residual block with adaptive group norm
ResidualBlock::ResidualBlock(int64_t in_channels, int64_t out_channels, int64_t cond_dim, bool attn)
  : proj_(in_channels, out_channels, 1, 1, 0),
    agn1_(std::max(int64_t(1), in_channels / GN_GROUP_SIZE), in_channels, cond_dim),
    conv1_(in_channels, out_channels, 3, 1, 1),
    agn2_(std::max(int64_t(1), out_channels / GN_GROUP_SIZE), out_channels, cond_dim),
    conv2_(out_channels, out_channels, 3, 1, 1),
    attn_(out_channels, std::max(int64_t(1), out_channels / ATTN_HEAD_DIM)),
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


}
}
