#include <iostream>
#include <cassert>
#include <tl/models.h>
#include <tl/factory.h>
#include "test_utils.h"

void test_models() {
  // test ResidualBlock: same channels (no skip conv)
  {
    tl::models::ResidualBlock rb(4, 4, 2, 8); // in=out=4, 2 groups, cond_dim=8

    tl::Tensor x    = tl::randn({2, 4, 8, 8}); // [N=2, C=4, H=8, W=8]
    tl::Tensor cond = tl::randn({2, 8});         // [N=2, cond_dim=8]

    tl::Tensor out = rb.forward(x, cond);

    // output shape preserved
    assert(out.sizes().size() == 4);
    assert(out.sizes()[0] == 2);
    assert(out.sizes()[1] == 4);
    assert(out.sizes()[2] == 8);
    assert(out.sizes()[3] == 8);

    // all values finite
    for (int i = 0; i < out.numel(); ++i)
      assert(std::isfinite(out.data()[i]));
  }

  // test ResidualBlock: different channels (skip conv active)
  {
    tl::models::ResidualBlock rb(4, 8, 2, 16); // in=4, out=8

    tl::Tensor x    = tl::randn({2, 4, 8, 8});
    tl::Tensor cond = tl::randn({2, 16});

    tl::Tensor out = rb.forward(x, cond);

    // output has out_channels
    assert(out.sizes()[1] == 8);
    assert(out.sizes()[2] == 8);
    assert(out.sizes()[3] == 8);

    for (int i = 0; i < out.numel(); ++i)
      assert(std::isfinite(out.data()[i]));
  }

  // test ResidualBlock: different cond -> different output
  {
    tl::models::ResidualBlock rb(4, 4, 2, 8);

    tl::Tensor x     = tl::randn({1, 4, 4, 4});
    tl::Tensor cond1 = tl::randn({1, 8});
    tl::Tensor cond2 = tl::randn({1, 8});

    tl::Tensor out1 = rb.forward(x, cond1);
    tl::Tensor out2 = rb.forward(x, cond2);

    bool any_diff = false;
    for (int i = 0; i < out1.numel(); ++i) {
      if (!is_close(out1.data()[i], out2.data()[i])) { any_diff = true; break; }
    }
    assert(any_diff);
  }

  // test ResidualBlock: parameters() returns non-empty list
  {
    tl::models::ResidualBlock rb_same(4, 4, 2, 8);  // no skip conv
    tl::models::ResidualBlock rb_diff(4, 8, 2, 8);  // with skip conv

    auto p_same = rb_same.parameters();
    auto p_diff = rb_diff.parameters();

    assert(!p_same.empty());
    assert(p_diff.size() > p_same.size()); // skip conv adds extra params
  }

  std::cout << "models tests passed" << std::endl;
}
