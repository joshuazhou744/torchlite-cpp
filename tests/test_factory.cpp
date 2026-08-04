#include <iostream>
#include <tl/tensor.h>
#include <tl/factory.h>
#include <cassert>
#include <cmath>
#include <stdexcept>
#include "test_utils.h"

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

  // test linspace: 5 points from 0 to 1 inclusive -> [0, 0.25, 0.5, 0.75, 1]
  tl::Tensor ls = tl::linspace(0.0f, 1.0f, 5);
  assert(ls.sizes().size() == 1);
  assert(ls.sizes()[0] == 5);
  assert(is_close(ls.data()[0], 0.0f));
  assert(is_close(ls.data()[1], 0.25f));
  assert(is_close(ls.data()[2], 0.5f));
  assert(is_close(ls.data()[3], 0.75f));
  assert(is_close(ls.data()[4], 1.0f));

  // test linspace endpoints inclusive over a non-unit range: [2, 4, 6, 8]
  tl::Tensor ls2 = tl::linspace(2.0f, 8.0f, 4);
  assert(ls2.sizes()[0] == 4);
  assert(is_close(ls2.data()[0], 2.0f));
  assert(is_close(ls2.data()[3], 8.0f));

  // test linspace with steps == 1 returns just start
  tl::Tensor ls3 = tl::linspace(3.0f, 9.0f, 1);
  assert(ls3.sizes()[0] == 1);
  assert(is_close(ls3.data()[0], 3.0f));

  // test like functions
  tl::Tensor ref = tl::randn({2, 3, 4});
  assert(tl::zeros_like(ref).sizes() == ref.sizes());
  assert(tl::zeros_like(ref).data()[5] == 0.0f);
  assert(tl::ones_like(ref).data()[5] == 1.0f);
  assert(is_close(tl::full_like(ref, -2.5f).data()[5], -2.5f));
  assert(tl::randn_like(ref).sizes() == ref.sizes());

  // test tri_mask: [rows, cols] of 1s inside the triangle, 0s outside
  {
    auto check = [](const tl::Tensor& t, int64_t rows, int64_t cols,
                    const std::vector<float>& want) {
      assert(t.sizes().size() == 2);
      assert(t.sizes()[0] == rows && t.sizes()[1] == cols);
      assert((int64_t)want.size() == rows * cols);
      for (int64_t i = 0; i < rows * cols; ++i) assert(t.data()[i] == want[i]);
    };

    // square, defaults (lower, diagonal 0): keep col <= row
    check(tl::tri_mask(4, 4), 4, 4, {
      1, 0, 0, 0,
      1, 1, 0, 0,
      1, 1, 1, 0,
      1, 1, 1, 1});

    // upper: keep col >= row
    check(tl::tri_mask(4, 4, 0, false), 4, 4, {
      1, 1, 1, 1,
      0, 1, 1, 1,
      0, 0, 1, 1,
      0, 0, 0, 1});

    // the output holds only exact 0.0 and 1.0, which is what makes masked_fill's
    // `mask == 0` test and tril's `mul` by the mask exact rather than approximate
    tl::Tensor binary = tl::tri_mask(5, 3, 1);
    for (int64_t i = 0; i < binary.numel(); ++i) {
      assert(binary.data()[i] == 0.0f || binary.data()[i] == 1.0f);
    }

    // non-square WIDE (rows < cols). asymmetric shapes are the only way to catch
    // transposed indexing: op[c * rows + r] stays in bounds here and scrambles
    // the contents while leaving the shape correct
    check(tl::tri_mask(2, 5), 2, 5, {
      1, 0, 0, 0, 0,
      1, 1, 0, 0, 0});

    // non-square TALL (rows > cols)
    check(tl::tri_mask(5, 2), 5, 2, {
      1, 0,
      1, 1,
      1, 1,
      1, 1,
      1, 1});

    // non-square upper, both orientations
    check(tl::tri_mask(2, 5, 0, false), 2, 5, {
      1, 1, 1, 1, 1,
      0, 1, 1, 1, 1});
    check(tl::tri_mask(5, 2, 0, false), 5, 2, {
      1, 1,
      0, 1,
      0, 0,
      0, 0,
      0, 0});

    // the KV-cache case: q_len=2 queries against k_len=5 keys with
    // diagonal = k_len - q_len, so the newest query sees every key
    check(tl::tri_mask(2, 5, 3), 2, 5, {
      1, 1, 1, 1, 0,
      1, 1, 1, 1, 1});

    // diagonal shifts the boundary one element per row per unit
    check(tl::tri_mask(3, 3, 1), 3, 3, {
      1, 1, 0,
      1, 1, 1,
      1, 1, 1});
    check(tl::tri_mask(3, 3, -1), 3, 3, {
      0, 0, 0,
      1, 0, 0,
      1, 1, 0});

    // lower(d=0) and upper(d=1) partition every cell exactly once, so their
    // elementwise sum is all ones. holds for non-square too, and catches an
    // off-by-one at the boundary (<= vs <)
    tl::Tensor lo = tl::tri_mask(3, 5, 0, true);
    tl::Tensor hi = tl::tri_mask(3, 5, 1, false);
    for (int64_t i = 0; i < 15; ++i) assert(lo.data()[i] + hi.data()[i] == 1.0f);

    // saturating diagonals: far negative is all zeros, far positive is all ones
    tl::Tensor none = tl::tri_mask(3, 3, -3);
    for (int64_t i = 0; i < 9; ++i) assert(none.data()[i] == 0.0f);
    tl::Tensor all = tl::tri_mask(3, 3, 2);
    for (int64_t i = 0; i < 9; ++i) assert(all.data()[i] == 1.0f);

    // non-positive dimensions throw
    bool threw = false;
    try { tl::tri_mask(0, 3); }
    catch (const std::invalid_argument&) { threw = true; }
    assert(threw);

    threw = false;
    try { tl::tri_mask(3, 0); }
    catch (const std::invalid_argument&) { threw = true; }
    assert(threw);

    threw = false;
    try { tl::tri_mask(-1, 3); }
    catch (const std::invalid_argument&) { threw = true; }
    assert(threw);
  }

  std::cout << "factory tests passed" << std::endl;
}
