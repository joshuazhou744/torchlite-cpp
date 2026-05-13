#include <iostream>
#include <cstdio>
#include <cassert>
#include <cmath>

#include <tl/tensor.h>
#include <tl/nn.h>
#include <tl/factory.h>
#include <tl/model_io.h>

void test_model_io() {
  // build a small model
  tl::nn::Linear l1(4, 8);
  tl::nn::Linear l2(8, 2);
  tl::nn::Sequential model({&l1, &l2});

  // save
  const std::string path = "test_model_io.tl";
  tl::save_model(path, model.parameters());

  // build a fresh model with same architecture, load into it
  tl::nn::Linear l1_loaded(4, 8);
  tl::nn::Linear l2_loaded(8, 2);
  tl::nn::Sequential model_loaded({&l1_loaded, &l2_loaded});
  tl::load_model(path, model_loaded.parameters());

  // every param tensor should match value-for-value
  auto orig = model.parameters();
  auto loaded = model_loaded.parameters();
  assert(orig.size() == loaded.size());
  for (size_t i = 0; i < orig.size(); ++i) {
    assert(orig[i]->numel() == loaded[i]->numel());
    for (int64_t j = 0; j < orig[i]->numel(); ++j) {
      assert(orig[i]->data()[j] == loaded[i]->data()[j]);
    }
  }

  // forward pass on both models should yield identical output
  tl::Tensor x = tl::randn({3, 4});
  tl::Tensor y_orig = model.forward(x);
  tl::Tensor y_loaded = model_loaded.forward(x);
  for (int64_t i = 0; i < y_orig.numel(); ++i) {
    assert(y_orig.data()[i] == y_loaded.data()[i]);
  }

  // shape mismatch should throw on load
  tl::nn::Linear wrong(4, 16); // out=16 instead of 8
  tl::nn::Sequential model_wrong({&wrong, &l2_loaded});
  bool threw = false;
  try {
    tl::load_model(path, model_wrong.parameters());
  } catch (const std::runtime_error&) {
    threw = true;
  }
  assert(threw);

  // cleanup
  std::remove(path.c_str());

  std::cout << "model_io tests passed" << std::endl;
}
