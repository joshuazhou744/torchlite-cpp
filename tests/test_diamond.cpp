#include <iostream>
#include <cassert>
#include <tl/diamond.h>
#include <tl/factory.h>
#include "test_utils.h"

void test_diamond() {
  {
    tl::diamond::ResidualBlock rb(4, 4, 8, false);

    tl::Tensor x    = tl::randn({2, 4, 8, 8});
    tl::Tensor cond = tl::randn({2, 8});

    tl::Tensor out = rb.forward(x, cond);

    assert(out.sizes().size() == 4);
    assert(out.sizes()[0] == 2);
    assert(out.sizes()[1] == 4);
    assert(out.sizes()[2] == 8);
    assert(out.sizes()[3] == 8);

    for (int i = 0; i < out.numel(); ++i)
      assert(std::isfinite(out.data()[i]));
  }

  {
    tl::diamond::ResidualBlock rb(4, 8, 16, false);

    tl::Tensor x    = tl::randn({2, 4, 8, 8});
    tl::Tensor cond = tl::randn({2, 16});

    tl::Tensor out = rb.forward(x, cond);

    assert(out.sizes()[1] == 8);
    assert(out.sizes()[2] == 8);
    assert(out.sizes()[3] == 8);

    for (int i = 0; i < out.numel(); ++i)
      assert(std::isfinite(out.data()[i]));
  }

  {
    tl::diamond::ResidualBlock rb_same(4, 4, 8, false);
    tl::diamond::ResidualBlock rb_diff(4, 8, 8, false);

    auto p_same = rb_same.parameters();
    auto p_diff = rb_diff.parameters();

    assert(!p_same.empty());
    assert(p_diff.size() > p_same.size());
  }

  {
    tl::diamond::ResidualBlock rb(8, 8, 16, true);

    tl::Tensor x    = tl::randn({2, 8, 4, 4});
    tl::Tensor cond = tl::randn({2, 16});

    tl::Tensor out = rb.forward(x, cond);

    assert(out.sizes()[1] == 8);
    for (int i = 0; i < out.numel(); ++i)
      assert(std::isfinite(out.data()[i]));
  }

  // ResidualBlocks: encoder mode (no skip connections)
  {
    tl::diamond::ResidualBlocks rbs({4, 4}, {4, 8}, 8, false);

    tl::Tensor x    = tl::randn({2, 4, 8, 8});
    tl::Tensor cond = tl::randn({2, 8});

    auto [out, outputs] = rbs.forward(x, cond);

    assert(out.sizes()[1] == 8);
    assert(out.sizes()[2] == 8);
    assert((int64_t)outputs.size() == 2);
    assert(outputs[0].sizes()[1] == 4);
    assert(outputs[1].sizes()[1] == 8);
    for (int i = 0; i < out.numel(); ++i)
      assert(std::isfinite(out.data()[i]));
  }

  // ResidualBlocks: decoder mode (with skip connections)
  {
    tl::Tensor x    = tl::randn({2, 4, 8, 8});
    tl::Tensor cond = tl::randn({2, 8});

    tl::Tensor skip0 = tl::randn({2, 4, 8, 8});
    tl::Tensor skip1 = tl::randn({2, 4, 8, 8});

    tl::diamond::ResidualBlocks rbs({8, 8}, {4, 4}, 8, false);

    auto [out, outputs] = rbs.forward(x, cond, {skip0, skip1});

    assert(out.sizes()[1] == 4);
    assert((int64_t)outputs.size() == 2);
    for (int i = 0; i < out.numel(); ++i)
      assert(std::isfinite(out.data()[i]));
  }

  std::cout << "diamond tests passed" << std::endl;
}
