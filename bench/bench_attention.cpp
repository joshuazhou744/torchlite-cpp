#include <tl/ops.h>
#include <tl/factory.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cmath>

namespace {

volatile float g_sink = 0.0f;
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

// reference attention: materializes the full [S,S] scores matrix
// out = softmax(Q·Kᵀ / sqrt(d)) · V
tl::Tensor reference_attention(const tl::Tensor& Q, const tl::Tensor& K,
                               const tl::Tensor& V, float sm_scale) {
  tl::Tensor scores = tl::matmul(Q, tl::transpose(K, 0, 1)); // [S,S]
  scores = tl::scale(scores, sm_scale);
  tl::Tensor attn   = tl::softmax(scores);                   // [S,S]
  return tl::matmul(attn, V);                                // [S,d]
}

void bench_size(int64_t S, int64_t d) {
  tl::Tensor Q = tl::randn({S, d});
  tl::Tensor K = tl::randn({S, d});
  tl::Tensor V = tl::randn({S, d});
  float sm_scale = 1.0f / std::sqrt((float)d);

  int iters = S <= 256 ? 20 : (S <= 1024 ? 5 : 3);

  double ref_s = time_best([&] {
    tl::Tensor o = reference_attention(Q, K, V, sm_scale);
    g_sink = o.data()[0];
  }, 2, iters);

  double flash_s = time_best([&] {
    tl::Tensor o = tl::flash_attention(Q, K, V, sm_scale);
    g_sink = o.data()[0];
  }, 2, iters);

  tl::Tensor a = reference_attention(Q, K, V, sm_scale);
  tl::Tensor b = tl::flash_attention(Q, K, V, sm_scale);
  float max_err = 0.0f;
  for (int64_t i = 0; i < S * d; ++i)
  max_err = std::max(max_err, std::fabs(a.data()[i] - b.data()[i]));

  double scores_mb = (double)S * S * 4 / (1024.0 * 1024.0);
  printf("  %6lld   ref %7.3f ms   flash %7.3f ms   %5.2fx   err %.1e   (%.1f MB saved)\n", (long long)S, ref_s * 1e3, flash_s * 1e3, ref_s / flash_s, max_err, scores_mb);
}

} // anon namespace

int main() {
  const int64_t d = 64; // head dim
  printf("attention bench (single head, head_dim=%lld)\n\n", (long long)d);
  printf("    seq      reference         flash       speedup    error     mem saved\n");
  printf("  -------------------------------------------------------------------------\n");
  for (int64_t S : {128, 256, 512, 1024, 2048}) bench_size(S, d);
}
