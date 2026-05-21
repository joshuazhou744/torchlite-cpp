#include <tl/ops.h>
#include <tl/factory.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>

// detect NEON for arm vs non-arm systems
#if defined(__aarch64__) || defined(__ARM_NEON)
  #include <arm_neon.h>
  #define TL_HAVE_NEON 1
#else
  #define TL_HAVE_NEON 0
#endif

namespace {

// volatile sink: reading a result into it stops the optimizer from deleting
// work whos output is otherwise unused
volatile float g_sink = 0.0f;

using clk = std::chrono::steady_clock;

// run fn(), return fastest (min) time
template <typename Fn>
double time_best(Fn&& fn, int warmup, int iters) {
  for (int i = 0; i < warmup; ++i) fn();
  double best = 1e30;
  for (int i = 0; i < iters; ++i) {
    auto t0 = clk::now();
    fn();
    auto t1 = clk::now();
    double secs = std::chrono::duration<double> (t1 - t0).count();
    best = std::min(best, secs);
  }
  return best;
}

// empirically measure the single core fp32 compute ceiling
double measure_peak_gflops() {
  const int64_t iters = 300'000'000;
#if TL_HAVE_NEON
  const int N = 16;                              // accumulators, 4 fp32 lanes each
  float32x4_t acc[N];
  for (int i = 0; i < N; ++i) acc[i] = vdupq_n_f32(0.01f * (i + 1));
  const float32x4_t a = vdupq_n_f32(0.9999999f); // <1 keeps the chain bounded
  const float32x4_t b = vdupq_n_f32(0.0000001f);

  auto t0 = clk::now();
  for (int64_t it = 0; it < iters; ++it)
    for (int i = 0; i < N; ++i) acc[i] = vfmaq_f32(b, a, acc[i]); // b + a*acc
  auto t1 = clk::now();

  float32x4_t s = acc[0];                         // reduce so nothing is dead code
  for (int i = 1; i < N; ++i) s = vaddq_f32(s, acc[i]);
  g_sink = vgetq_lane_f32(s, 0) + vgetq_lane_f32(s, 1)
         + vgetq_lane_f32(s, 2) + vgetq_lane_f32(s, 3);

  double secs  = std::chrono::duration<double>(t1 - t0).count();
  double flops = (double)iters * N * 4 /*lanes*/ * 2 /*mul+add*/;
  return flops / secs / 1e9;
#else
  const int N = 16;                               // scalar fallback (non-ARM)
  float acc[N];
  for (int i = 0; i < N; ++i) acc[i] = 0.01f * (i + 1);
  const float a = 0.9999999f, b = 0.0000001f;

  auto t0 = clk::now();
  for (int64_t it = 0; it < iters; ++it)
    for (int i = 0; i < N; ++i) acc[i] = a * acc[i] + b;
  auto t1 = clk::now();

  float s = 0; for (int i = 0; i < N; ++i) s += acc[i];
  g_sink = s;

  double secs  = std::chrono::duration<double>(t1 - t0).count();
  double flops = (double)iters * N * 2;
  return flops / secs / 1e9;
#endif
}

// time matmul on an n x n problem and print GFLOP/s + % of peak
void bench_size(int64_t n, double peak) {
  tl::Tensor a = tl::randn({n, n});
  tl::Tensor b = tl::randn({n, n});

  int iters = n <= 128 ? 50 : (n < 512 ? 10 : 3);
  double secs = time_best([&] {
      tl::Tensor c = tl::matmul(a, b);
      g_sink = c.data()[0];
  }, 2, iters);

  double flops = 2.0 * (double)n * n * n;
  double gflops = flops / secs / 1e9;
  printf("  %5lld   %9.3f ms   %9.2f GFLOP/s   %5.1f%%\n", (long long)n, secs * 1e3, gflops, 100.0 * gflops / peak);
}
} // anon namespace

int main() {
  printf("measuring single-core compute peak");
  double peak = measure_peak_gflops();
  printf("peak: %.2f GFLOP/s (single core, fp32)\n\n", peak);

  printf("   size        time          throughput      %%peak\n");
  printf("  ----------------------------------------------------\n");
  for (int64_t n : {64, 128, 256, 512, 1024}) bench_size(n, peak);
}
