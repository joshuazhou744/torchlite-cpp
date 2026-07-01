#include <tl/models.h>
#include <tl/activation.h>
#include <tl/ops.h>

#include <cstdint>

namespace tl {
namespace models {


// Residual block with adaptive group norm
ResidualBlock::ResidualBlock(int64_t in_channels, int64_t out_channels, int64_t num_groups, int64_t cond_dim)
  : conv1_(in_channels, out_channels, 3, 1, 1),
    conv2_(out_channels, out_channels, 3, 1, 1),
    agn1_(num_groups, in_channels, cond_dim),
    agn2_(num_groups, out_channels, cond_dim),
    skip_conv_(in_channels, out_channels, 1, 1, 0),
    needs_skip_conv_(in_channels != out_channels)
{}

std::vector<Tensor*> ResidualBlock::parameters() {
  std::vector<Tensor*> params;
  auto append = [&](auto& m) {
    auto p = m.parameters();
    params.insert(params.end(), p.begin(), p.end());
  };
  append(conv1_);
  append(conv2_);
  append(agn1_);
  append(agn2_);
  if (needs_skip_conv_) append(skip_conv_);
  return params;
}

Tensor ResidualBlock::forward(const Tensor& x, const Tensor& cond) const {
  Tensor h = agn1_.forward(x, cond);
  h = silu(h);
  h = conv1_.forward(h);
  h = agn2_.forward(h, cond);
  h = silu(h);
  h = conv2_.forward(h);

  Tensor skip = needs_skip_conv_ ? skip_conv_.forward(x) : x;
  return add(h, skip);
}


}
}
