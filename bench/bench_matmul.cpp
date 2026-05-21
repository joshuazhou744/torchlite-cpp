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
  const int N = 16; // accumulators, 4 fp32 lanes each
  float32x4_t c0=vdupq_n_f32(0.01f),  c1=vdupq_n_f32(0.02f),  c2=vdupq_n_f32(0.03f), c3=vdupq_n_f32(0.04f);
  float32x4_t c4=vdupq_n_f32(0.05f),  c5=vdupq_n_f32(0.06f),  c6=vdupq_n_f32(0.07f), c7=vdupq_n_f32(0.08f);
  float32x4_t c8=vdupq_n_f32(0.09f),  c9=vdupq_n_f32(0.10f),  c10=vdupq_n_f32(0.11f), c11=vdupq_n_f32(0.12f);
  float32x4_t c12=vdupq_n_f32(0.13f), c13=vdupq_n_f32(0.14f), c14=vdupq_n_f32(0.15f), c15=vdupq_n_f32(0.16f);
  const float32x4_t a = vdupq_n_f32(0.9999999f); // <1 keeps the chain bounded
  const float32x4_t b = vdupq_n_f32(0.0000001f);


  auto t0 = clk::now();
  for (int64_t it = 0; it < iters; ++it) {
    c0 =vfmaq_f32(b,a,c0);  c1 =vfmaq_f32(b,a,c1);  c2 =vfmaq_f32(b,a,c2);  c3 =vfmaq_f32(b,a,c3);
    c4 =vfmaq_f32(b,a,c4);  c5 =vfmaq_f32(b,a,c5);  c6 =vfmaq_f32(b,a,c6);  c7=vfmaq_f32(b,a,c7);
    c8 =vfmaq_f32(b,a,c8);  c9 =vfmaq_f32(b,a,c9);  c10=vfmaq_f32(b,a,c10); c11=vfmaq_f32(b,a,c11);
    c12=vfmaq_f32(b,a,c12); c13=vfmaq_f32(b,a,c13); c14=vfmaq_f32(b,a,c14); c15=vfmaq_f32(b,a,c15);
  }
  auto t1 = clk::now();

  float32x4_t s = vaddq_f32(vaddq_f32(vaddq_f32(c0,c1), vaddq_f32(c2,c3)), vaddq_f32(vaddq_f32(c4,c5), vaddq_f32(c6,c7)));
  s = vaddq_f32(s, vaddq_f32(vaddq_f32(vaddq_f32(c8,c9), vaddq_f32(c10,c11)), vaddq_f32(vaddq_f32(c12,c13), vaddq_f32(c14,c15))));
  g_sink = vgetq_lane_f32(s,0)+vgetq_lane_f32(s,1)+vgetq_lane_f32(s,2)+vgetq_lane_f32(s,3);

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
