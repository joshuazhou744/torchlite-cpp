#include <tl/nn.h>
#include <tl/ops.h>
#include <tl/factory.h>

#include <chrono>
#include <cstdio>
#include <algorithm>
#include <fstream>
#include <string>
#include <vector>

namespace {

using clk = std::chrono::steady_clock;

template <typename Fn>
double time_best(Fn&& fn, int warmup, int iters) {
  for (int i = 0; i < warmup; ++i) fn();
  double best = 1e30;
  for (int i = 0; i < iters; ++i) {
    auto t0 = clk::now();
    fn();
    auto t1 = clk::now();
    best = std::min(best, std::chrono::duration<double>(t1 - t0).count());
  }
  return best;
}

// peak resident set size in MB from /proc/self/status (Linux)
long peak_rss_mb() {
  std::ifstream f("/proc/self/status");
  std::string line;
  while (std::getline(f, line)) {
    if (line.rfind("VmPeak:", 0) == 0) {
      long kb = 0;
      sscanf(line.c_str(), "VmPeak: %ld kB", &kb);
      return kb / 1024;
    }
  }
  return -1;
}

void bench(int64_t depth, int64_t width, int64_t batch) {
  // build depth Linear+ReLU layers shared by both runs
  std::vector<tl::nn::Linear*> linears;
  std::vector<tl::nn::ReLU*>   relus;
  std::vector<tl::nn::Module*> layer_ptrs;
  for (int64_t i = 0; i < depth; ++i) {
    linears.push_back(new tl::nn::Linear(width, width));
    relus.push_back(new tl::nn::ReLU());
    layer_ptrs.push_back(linears.back());
    layer_ptrs.push_back(relus.back());
  }
  tl::nn::Sequential block(layer_ptrs);

  // --- plain ---
  tl::Tensor x_plain = tl::randn({batch, width});
  x_plain.set_requires_grad(true);
  long rss_before_plain = peak_rss_mb();
  double t_plain = time_best([&] {
    x_plain.grad() = tl::Tensor();
    for (auto* p : block.parameters()) p->grad() = tl::Tensor();
    tl::Tensor y = block.forward(x_plain);
    y.backward();
  }, 1, 5);
  long rss_plain = peak_rss_mb() - rss_before_plain;

  // --- checkpointed ---
  tl::nn::Checkpoint ckpt(&block);
  tl::Tensor x_ckpt = tl::randn({batch, width});
  x_ckpt.set_requires_grad(true);
  long rss_before_ckpt = peak_rss_mb();
  double t_ckpt = time_best([&] {
    x_ckpt.grad() = tl::Tensor();
    for (auto* p : block.parameters()) p->grad() = tl::Tensor();
    tl::Tensor y = ckpt.forward(x_ckpt);
    y.backward();
  }, 1, 5);
  long rss_ckpt = peak_rss_mb() - rss_before_ckpt;

  printf("  depth=%-3lld  width=%-5lld  batch=%-5lld  |"
         "  plain: %6.1f ms  +%3ld MB  |"
         "  ckpt: %6.1f ms  +%3ld MB  |"
         "  slowdown: %.2fx\n",
         (long long)depth, (long long)width, (long long)batch,
         t_plain * 1e3, rss_plain,
         t_ckpt  * 1e3, rss_ckpt,
         t_ckpt / t_plain);

  for (auto* p : linears) delete p;
  for (auto* p : relus)   delete p;
}

} // anon namespace

int main() {
  printf("gradient checkpointing benchmark: plain vs checkpointed forward+backward\n");
  printf("  (ckpt trades compute for memory: expect ~2x slowdown, lower peak RSS)\n\n");
  printf("  config                          |  plain                  |  checkpointed           |  overhead\n");
  printf("  -----------------------------------------------------------------------------------------------------------------------\n");

  bench(8,  256,  128);
  bench(16, 256,  128);
  bench(32, 256,  128);
  bench(16, 512,  128);
  bench(16, 1024, 128);
  bench(16, 256,  512);
}
