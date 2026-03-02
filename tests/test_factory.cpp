#include <iostream>
#include <tl/tensor.h>
#include <tl/factory.h>
#include <cassert>
#include <cmath>

void test_factory() {
  // test full: all elements should equal the given value
  tl::Tensor f = tl::full({2, 3}, 7.0f);
  assert(f.sizes()[0] == 2 && f.sizes()[1] == 3);
  for (int i = 0; i < f.numel(); ++i) {
    assert(f.data()[i] == 7.0f);
  }

  // test zeros: all elements should be 0
  tl::Tensor z = tl::zeros({3, 4});
  assert(z.numel() == 12);
  for (int i = 0; i < z.numel(); ++i) {
    assert(z.data()[i] == 0.0f);
  }

  // test ones: all elements should be 1
  tl::Tensor o = tl::ones({2, 2});
  assert(o.numel() == 4);
  for (int i = 0; i < o.numel(); ++i) {
    assert(o.data()[i] == 1.0f);
  }

  // test randn: correct shape, values are finite
  tl::Tensor r = tl::randn({100});
  assert(r.sizes()[0] == 100);
  for (int i = 0; i < r.numel(); ++i) {
    assert(std::isfinite(r.data()[i]));
  }

  // test randn: not all values are the same (extremely unlikely)
  bool has_different = false;
  for (int i = 1; i < r.numel(); ++i) {
    if (r.data()[i] != r.data()[0]) {
      has_different = true;
      break;
    }
  }
  assert(has_different);

  // test arange: [2, 3, 4, 5]
  tl::Tensor a = tl::arange(2, 6);
  assert(a.sizes().size() == 1);
  assert(a.sizes()[0] == 4);
  assert(a.data()[0] == 2.0f);
  assert(a.data()[1] == 3.0f);
  assert(a.data()[2] == 4.0f);
  assert(a.data()[3] == 5.0f);

  // test arange from 0: [0, 1, 2]
  tl::Tensor a2 = tl::arange(0, 3);
  assert(a2.sizes()[0] == 3);
  assert(a2.data()[0] == 0.0f);
  assert(a2.data()[2] == 2.0f);

  std::cout << "factory tests passed" << std::endl;
}
