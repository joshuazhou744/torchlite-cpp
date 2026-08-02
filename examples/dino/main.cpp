#include <tl/dino.h>
#include <tl/tensor.h>
#include <tl/factory.h>
#include <tl/model_io.h>

#include <cmath>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct Config {
  int64_t dim = 0;
  int64_t depth = 0;
  int64_t num_heads = 0;
  int64_t patch_size = 0;
  int64_t n_storage = 0;
  int64_t batch = 0;
  int64_t height = 0;
  int64_t width = 0;
  std::vector<int64_t> layers;
};

static constexpr float TOL = 1e-2f;

// parse key value lines into cfg
static void read_cfg(const std::string& path, Config& cfg) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open cfg: " + path);

  std::string line;
  while (std::getline(in, line)) {
    std::istringstream ls(line);
    std::string key;
    if (!(ls >> key)) continue; // blank line

    // layer line
    if (key == "layers") {
      int64_t v;
      while (ls >> v) cfg.layers.push_back(v);
      continue;
    }

    int64_t v;
    if (!(ls >> v)) throw std::runtime_error("cfg: no value for key '" + key + "'");
    if (key == "dim") cfg.dim = v;
    else if (key == "depth") cfg.depth = v;
    else if (key == "num_heads") cfg.num_heads = v;
    else if (key == "patch_size") cfg.patch_size = v;
    else if (key == "n_storage") cfg.n_storage = v;
    else if (key == "batch") cfg.batch = v;
    else if (key == "height") cfg.height = v;
    else if (key == "width") cfg.width = v;
    else throw std::runtime_error("cfg: unknown key " + key);
  }
}

static bool compare(const tl::Tensor& got, const tl::Tensor& exp, int64_t layer, float tol) {
  tl::Tensor a = got.contiguous(); // what .tl loaded gives
  tl::Tensor b = exp.contiguous(); // what we expected (from pytorch)
  if (a.sizes() != b.sizes()) {
    std::cerr << "layer " << layer << ": shape mismatch" << std::endl;
    return false;
  }

  const float* pa = a.data();
  const float* pb = b.data();
  int64_t n = a.numel();
  float max_abs = 0.0f, ref_max = 0.0f;
  double sum_abs = 0.0;
  bool finite = true;
  for (int64_t j = 0; j < n; ++j) {
    if (!std::isfinite(pa[j])) finite = false;
    float d = std::fabs(pa[j] - pb[j]);
    sum_abs += d;
    max_abs = std::max(max_abs, d);
    ref_max = std::max(ref_max, std::fabs(pb[j]));
  }
  float rel = (ref_max > 0.0f) ? max_abs / ref_max : max_abs;
  bool ok = finite && rel <= tol;

  std::cout << "layer " << layer
    << ": max_abs=" << max_abs
    << " mean_abs=" << (sum_abs / n)
    << " ref_max=" << ref_max << " ref=" << rel
    << (finite ? (ok ? " PASS" : " FAIL") : " NON-FINITE") << std::endl;
  return ok;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: " << argv[0] << " <model_prefix>" << std::endl;
    return 1;
  }
  std::string prefix = argv[1];

  Config cfg;
  read_cfg(prefix + ".cfg", cfg); // model config
  read_cfg(prefix + ".golden.cfg", cfg); // test image config

  int64_t h = cfg.height / cfg.patch_size;
  int64_t w = cfg.width / cfg.patch_size;

  std::cout << "config: dim=" << cfg.dim << " depth=" << cfg.depth
          << " heads=" << cfg.num_heads << " patch=" << cfg.patch_size
          << " storage=" << cfg.n_storage << std::endl;
  std::cout << "input:  [" << cfg.batch << ", 3, " << cfg.height << ", " << cfg.width
          << "] -> grid " << h << "x" << w << ", "
          << (1 + cfg.n_storage + h * w) << " tokens" << std::endl;

  tl::dino::DinoViT vit(cfg.dim, cfg.depth, cfg.num_heads, cfg.patch_size, cfg.n_storage);
  auto params = vit.parameters();
  tl::load_model(prefix + ".tl", params);
  std::cout << "loaded " << params.size() << " weight tensors" << std::endl;

  tl::Tensor x = tl::zeros({cfg.batch, 3, cfg.height, cfg.width});
  std::vector<tl::Tensor> expected;
  expected.reserve(cfg.layers.size());
  for (size_t i = 0; i < cfg.layers.size(); ++i) {
    expected.push_back(tl::zeros({cfg.batch, cfg.dim, h, w}));
  }

  std::vector<tl::Tensor*> golden = {&x};
  for (auto& e: expected) golden.push_back(&e);

  tl::load_model(prefix + ".golden", golden);
  std::cout << "loaded golden: input + " << expected.size() << " feature maps" << std::endl;

  std::vector<tl::Tensor> got = vit.forward(x, cfg.layers);

  bool pass = true;
  for (size_t i = 0; i < got.size(); ++i) {
    if (!compare(got[i], expected[i], cfg.layers[i], TOL)) pass = false;
  }

  std::cout << (pass ? "golden test PASSED" : "golden test FAILED") << std::endl;
  return 0;
}
