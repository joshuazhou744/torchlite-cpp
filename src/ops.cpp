#include <tl/ops.h>
#include <tl/factory.h>
#include <tl/autograd.h>

#include <iostream>
#include <omp.h>
#include <cstdint> // for int64_t
#include <stdexcept>
#include <algorithm> // for max()
#include <cmath> // for sqrt() and exp()
#include <limits> // for infinity
#include <thread>

#if defined(__aarch64__) || defined(__ARM_NEON)
  #include <arm_neon.h>
  #define TL_HAVE_NEON 1
#else
  #define TL_HAVE_NEON 0
#endif

namespace tl {

// helper function to validate two tensors have same shape
//static void check_same_shape(const Tensor& a, const Tensor& b) {
  //if (a.sizes() != b.sizes()) {
  //throw std::invalid_argument("Tensor shapes must match");
//}
//}

// helper function to map a multidimensional index to a flat index considering broadcasting
static int64_t get_broadcast_index(int64_t linear_index,
                                  const std::vector<int64_t>& t_sizes,
                                  const std::vector<int64_t>& t_strides,
                                  const std::vector<int64_t>& target_sizes) {
  int64_t physical_index = 0;
  int ndim = target_sizes.size();
  int t_ndim = t_sizes.size();

  for (int i = ndim - 1; i >= 0; --i) {
    int64_t dim_index = linear_index % target_sizes[i];
    linear_index /= target_sizes[i];

    // if dimension exists int he smaller tensor and isn't size 1, use stride
    int t_dim_index = i - (ndim - t_ndim);
    if (t_dim_index >= 0 && t_sizes[t_dim_index] != 1) {
      physical_index += dim_index * t_strides[t_dim_index];
    }
  }

  return physical_index;
}

// helper function to find the final broadcasted shape
static std::vector<int64_t> compute_broadcast_shape(const std::vector<int64_t>& a, const std::vector<int64_t>& b) {
  int ndim_a = a.size();
  int ndim_b = b.size();
  int ndim_out = std::max(ndim_a, ndim_b);
  std::vector<int64_t> out_shape(ndim_out);

  for (int i = ndim_out - 1; i >= 0; --i) {
    int64_t size_a = (i < ndim_out - ndim_a) ? 1 : a[i - (ndim_out - ndim_a)];
    int64_t size_b = (i < ndim_out - ndim_b) ? 1 : b[i - (ndim_out - ndim_b)];

    if (size_a != size_b && size_a != 1 && size_b != 1) {
      throw std::invalid_argument("Incompatible dimensions for broadcasting");
    }

    out_shape[i] = std::max(size_a, size_b);
  }
  return out_shape;
}

// element-wise addition
Tensor add(const Tensor& a, const Tensor& b) {
  // determine resulting shape
  std::vector<int64_t> out_shape = compute_broadcast_shape(a.sizes(), b.sizes());
  Tensor out(out_shape);

  const auto& sizes_a = a.sizes();
  const auto& strides_a = a.strides();
  const auto& sizes_b = b.sizes();
  const auto& strides_b = b.strides();

  float* op = out.data();
  const float* ap = a.data();
  const float* bp = b.data();

  // iterate through the outputs flat memory
  const int64_t n = out.numel();
  for (int64_t i = 0; i < n; ++i) {
    // map outputs linear index to physical index
    int64_t index_a = get_broadcast_index(i, sizes_a, strides_a, out_shape);
    int64_t index_b = get_broadcast_index(i, sizes_b, strides_b, out_shape);

    op[i] = ap[index_a] + bp[index_b];
  }

  if (a.requires_grad || b.requires_grad) {
    track<AddBackward>(out, {&a, &b});
  }
  return out;
}

// element-wise subtraction
Tensor sub(const Tensor& a, const Tensor& b) {
  std::vector<int64_t> out_shape = compute_broadcast_shape(a.sizes(), b.sizes());
  Tensor out(out_shape);

  const auto& sizes_a = a.sizes();
  const auto& strides_a = a.strides();
  const auto& sizes_b = b.sizes();
  const auto& strides_b = b.strides();

  float* op = out.data();
  const float* ap = a.data();
  const float* bp = b.data();

  const int64_t n = out.numel();
  for (int64_t i = 0; i < n; ++i) {
    int64_t index_a = get_broadcast_index(i, sizes_a, strides_a, out_shape);
    int64_t index_b = get_broadcast_index(i, sizes_b, strides_b, out_shape);
    op[i] = ap[index_a] - bp[index_b];
  }

  if (a.requires_grad || b.requires_grad) {
    track<SubBackward>(out, {&a, &b});
  }

  return out;
}

// unary sqrt
Tensor sqrt(const Tensor& input) {
  Tensor a = input.contiguous();
  Tensor out(a.sizes());
  const float* ap = a.data();
  float* op = out.data();

  const int64_t n = a.numel();
  for (int64_t i = 0; i < n; ++i) {
    op[i] = std::sqrt(ap[i]);
  }

  if (input.requires_grad) {
    if (auto fn = track<SqrtBackward>(out, {&input})) {
      fn->output_cache = out.contiguous();
    }
  }

  return out;
}


Tensor mul(const Tensor& a, const Tensor& b) {
  // determine resulting shape
  std::vector<int64_t> out_shape = compute_broadcast_shape(a.sizes(), b.sizes());
  Tensor out(out_shape);

  const auto& sizes_a = a.sizes();
  const auto& strides_a = a.strides();
  const auto& sizes_b = b.sizes();
  const auto& strides_b = b.strides();

  float* op = out.data();
  const float* ap = a.data();
  const float* bp = b.data();

  // iterate through the outputs flat memory
  const int64_t n = out.numel();
  for (int64_t i = 0; i < n; ++i) {
    // map outputs linear index to physical index
    int64_t index_a = get_broadcast_index(i, sizes_a, strides_a, out_shape);
    int64_t index_b = get_broadcast_index(i, sizes_b, strides_b, out_shape);

    op[i] = ap[index_a] * bp[index_b];
  }

  if (a.requires_grad || b.requires_grad) {
    if (auto fn = track<MulBackward>(out, {&a, &b})) {
      fn->a_cache = a.contiguous();
      fn->b_cache = b.contiguous();
    }
  }

  return out;
}

Tensor div(const Tensor& a, const Tensor& b) {
  // determine resulting shape
  std::vector<int64_t> out_shape = compute_broadcast_shape(a.sizes(), b.sizes());
  Tensor out(out_shape);

  const auto& sizes_a = a.sizes();
  const auto& strides_a = a.strides();
  const auto& sizes_b = b.sizes();
  const auto& strides_b = b.strides();

  float* op = out.data();
  const float* ap = a.data();
  const float* bp = b.data();

  // iterate through the outputs flat memory
  const int64_t n = out.numel();
  for (int64_t i = 0; i < n; ++i) {
    // map outputs linear index to physical index
    int64_t index_a = get_broadcast_index(i, sizes_a, strides_a, out_shape);
    int64_t index_b = get_broadcast_index(i, sizes_b, strides_b, out_shape);

    op[i] = ap[index_a] / bp[index_b];
  }

  if (a.requires_grad || b.requires_grad) {
    if (auto fn = track<DivBackward>(out, {&a, &b})) {
      fn->a_cache = a.contiguous();
      fn->b_cache = b.contiguous();
    }
  }

  return out;
}

// GEMM kernels: out[M,N] = a[M,K] * b[K,N]
// Stages: scalar baseline -> cache blocking -> register tiling -> packing.

// naive triple loop
[[maybe_unused]] static void gemm_naive(const float* a, const float* b, float* out, int64_t M, int64_t N, int64_t K) {
  for (int64_t m = 0; m < M; ++m) {
    for (int64_t n = 0; n < N; ++n) {
      out[m * N + n] = 0.0f;
    }
    for (int64_t k = 0; k < K; ++k) {
      float a_val = a[m * K + k];
      for (int64_t n = 0; n < N; ++n) {
        out[m * N + n] += a_val * b[k * N + n];
      }
    }
  }
}

// 64x64 cache-blocked triple loop
static void gemm_blocked(const float* a, const float* b, float* out, int64_t M, int64_t N, int64_t K) {
  const int64_t T = 64;

  // zero up front: the kk loop accumulates into each tile across many steps
  for (int64_t i = 0; i < M * N; ++i) out[i] = 0.0f;

  // outer loops: step over output tiles and slices of the contraction dim, K
  for (int64_t mm = 0; mm < M; mm += T) {
    int64_t m_end = std::min(mm + T, M);
    for (int64_t nn = 0; nn < N; nn += T) {
      int64_t n_end = std::min(nn + T, N);
      for (int64_t kk = 0; kk < K; kk += T) {
        int64_t k_end = std::min(kk + T, K);

        // inner loops: regular matmul on tiles
        for (int64_t m = mm; m < m_end; ++m) {
          for (int64_t k = kk; k < k_end; ++k) {
            float a_val = a[m * K + k];
            for (int64_t n = nn; n < n_end; ++n) {
              out[m * N + n] += a_val * b[k * N + n];
            }
          }
        }
      }
    }
  }
}

// REGISTER TILING: 8x8 NEON micro-kernels (16 vector regs)
#if TL_HAVE_NEON
// micro_kernel_8x8: compute on full 8x8 tile of C = A_panel * B_panel in registers
// a: top left of 8-row A panel -> a[row*K + k]
// b: top left of 8-col B panel -> b[k*N + col]
// c: top left of the 8x8 C tile
static inline void micro_kernel_8x8(const float* a, const float* b, float* c, int64_t K, int64_t N) {
  // 8 rows x (8 cols = 2 NEON vectors) = 16 accumulator registers
  float32x4_t c0l = vdupq_n_f32(0.0f), c0r = vdupq_n_f32(0.0f);
  float32x4_t c1l = vdupq_n_f32(0.0f), c1r = vdupq_n_f32(0.0f);
  float32x4_t c2l = vdupq_n_f32(0.0f), c2r = vdupq_n_f32(0.0f);
  float32x4_t c3l = vdupq_n_f32(0.0f), c3r = vdupq_n_f32(0.0f);
  float32x4_t c4l = vdupq_n_f32(0.0f), c4r = vdupq_n_f32(0.0f);
  float32x4_t c5l = vdupq_n_f32(0.0f), c5r = vdupq_n_f32(0.0f);
  float32x4_t c6l = vdupq_n_f32(0.0f), c6r = vdupq_n_f32(0.0f);
  float32x4_t c7l = vdupq_n_f32(0.0f), c7r = vdupq_n_f32(0.0f);

  for (int64_t k = 0; k < K; ++k) {
    float32x4_t bl = vld1q_f32(b + k * N + 0);   // cols 0..3
    float32x4_t br = vld1q_f32(b + k * N + 4);   // cols 4..7
    float a0 = a[0*K + k]; c0l = vfmaq_n_f32(c0l, bl, a0); c0r = vfmaq_n_f32(c0r, br, a0);
    float a1 = a[1*K + k]; c1l = vfmaq_n_f32(c1l, bl, a1); c1r = vfmaq_n_f32(c1r, br, a1);
    float a2 = a[2*K + k]; c2l = vfmaq_n_f32(c2l, bl, a2); c2r = vfmaq_n_f32(c2r, br, a2);
    float a3 = a[3*K + k]; c3l = vfmaq_n_f32(c3l, bl, a3); c3r = vfmaq_n_f32(c3r, br, a3);
    float a4 = a[4*K + k]; c4l = vfmaq_n_f32(c4l, bl, a4); c4r = vfmaq_n_f32(c4r, br, a4);
    float a5 = a[5*K + k]; c5l = vfmaq_n_f32(c5l, bl, a5); c5r = vfmaq_n_f32(c5r, br, a5);
    float a6 = a[6*K + k]; c6l = vfmaq_n_f32(c6l, bl, a6); c6r = vfmaq_n_f32(c6r, br, a6);
    float a7 = a[7*K + k]; c7l = vfmaq_n_f32(c7l, bl, a7); c7r = vfmaq_n_f32(c7r, br, a7);
  }

  vst1q_f32(c + 0*N + 0, c0l); vst1q_f32(c + 0*N + 4, c0r);
  vst1q_f32(c + 1*N + 0, c1l); vst1q_f32(c + 1*N + 4, c1r);
  vst1q_f32(c + 2*N + 0, c2l); vst1q_f32(c + 2*N + 4, c2r);
  vst1q_f32(c + 3*N + 0, c3l); vst1q_f32(c + 3*N + 4, c3r);
  vst1q_f32(c + 4*N + 0, c4l); vst1q_f32(c + 4*N + 4, c4r);
  vst1q_f32(c + 5*N + 0, c5l); vst1q_f32(c + 5*N + 4, c5r);
  vst1q_f32(c + 6*N + 0, c6l); vst1q_f32(c + 6*N + 4, c6r);
  vst1q_f32(c + 7*N + 0, c7l); vst1q_f32(c + 7*N + 4, c7r);
}

// micro_kernel_8x8_acc: same 8x8 register tile, accumulates a kc-deep slab instead of entire row/col
// accumulate: load C, FMA a kc-deep slab, store back
static inline void micro_kernel_8x8_acc(const float* a, const float* b, float* c, int64_t kc, int64_t K, int64_t N) {
  // 8 rows x (8 cols = 2 NEON vectors) = 16 accumulator registers
  float32x4_t c0l = vld1q_f32(c + 0*N + 0), c0r = vld1q_f32(c + 0*N + 4);
  float32x4_t c1l = vld1q_f32(c + 1*N + 0), c1r = vld1q_f32(c + 1*N + 4);
  float32x4_t c2l = vld1q_f32(c + 2*N + 0), c2r = vld1q_f32(c + 2*N + 4);
  float32x4_t c3l = vld1q_f32(c + 3*N + 0), c3r = vld1q_f32(c + 3*N + 4);
  float32x4_t c4l = vld1q_f32(c + 4*N + 0), c4r = vld1q_f32(c + 4*N + 4);
  float32x4_t c5l = vld1q_f32(c + 5*N + 0), c5r = vld1q_f32(c + 5*N + 4);
  float32x4_t c6l = vld1q_f32(c + 6*N + 0), c6r = vld1q_f32(c + 6*N + 4);
  float32x4_t c7l = vld1q_f32(c + 7*N + 0), c7r = vld1q_f32(c + 7*N + 4);

  for (int64_t k = 0; k < kc; ++k) {
    // load 8 columns of B at row k: two contiguous 4-float vectors
    float32x4_t bl = vld1q_f32(b + k * N + 0);   // cols 0..3
    float32x4_t br = vld1q_f32(b + k * N + 4);   // cols 4..7

    // broadcast each row's A value and FMA into that row's accumulators
    float a0 = a[0*K + k]; c0l = vfmaq_n_f32(c0l, bl, a0); c0r = vfmaq_n_f32(c0r, br, a0);
    float a1 = a[1*K + k]; c1l = vfmaq_n_f32(c1l, bl, a1); c1r = vfmaq_n_f32(c1r, br, a1);
    float a2 = a[2*K + k]; c2l = vfmaq_n_f32(c2l, bl, a2); c2r = vfmaq_n_f32(c2r, br, a2);
    float a3 = a[3*K + k]; c3l = vfmaq_n_f32(c3l, bl, a3); c3r = vfmaq_n_f32(c3r, br, a3);
    float a4 = a[4*K + k]; c4l = vfmaq_n_f32(c4l, bl, a4); c4r = vfmaq_n_f32(c4r, br, a4);
    float a5 = a[5*K + k]; c5l = vfmaq_n_f32(c5l, bl, a5); c5r = vfmaq_n_f32(c5r, br, a5);
    float a6 = a[6*K + k]; c6l = vfmaq_n_f32(c6l, bl, a6); c6r = vfmaq_n_f32(c6r, br, a6);
    float a7 = a[7*K + k]; c7l = vfmaq_n_f32(c7l, bl, a7); c7r = vfmaq_n_f32(c7r, br, a7);
  }

  // write the 8x8 tile back to C
  vst1q_f32(c + 0*N + 0, c0l); vst1q_f32(c + 0*N + 4, c0r);
  vst1q_f32(c + 1*N + 0, c1l); vst1q_f32(c + 1*N + 4, c1r);
  vst1q_f32(c + 2*N + 0, c2l); vst1q_f32(c + 2*N + 4, c2r);
  vst1q_f32(c + 3*N + 0, c3l); vst1q_f32(c + 3*N + 4, c3r);
  vst1q_f32(c + 4*N + 0, c4l); vst1q_f32(c + 4*N + 4, c4r);
  vst1q_f32(c + 5*N + 0, c5l); vst1q_f32(c + 5*N + 4, c5r);
  vst1q_f32(c + 6*N + 0, c6l); vst1q_f32(c + 6*N + 4, c6r);
  vst1q_f32(c + 7*N + 0, c7l); vst1q_f32(c + 7*N + 4, c7r);
}
#endif

// register-tiled GEMM kernel (ACTIVE)
// 8x8 micro-tiles over full K + scalar edge remainders
static void gemm_tiled(const float* a, const float* b, float* out, int64_t M, int64_t N, int64_t K) {
#if TL_HAVE_NEON
  const int64_t MR = 8, NR = 8;

  // zero once: micro-kernel overwrites full tiles, the remainder code accumulates (+=)
  for (int64_t i = 0; i < M * N; ++i) out[i] = 0.0f;

  const int64_t M8 = (M / MR) * MR;   // full-tile region (largest multiples of 8)
  const int64_t N8 = (N / NR) * NR;

  for (int64_t m = 0; m < M8; m += MR)
    for (int64_t n = 0; n < N8; n += NR)
      micro_kernel_8x8(a + m * K, b + n, out + m * N + n, K, N);

  // right edge remainders (columns N8..N for rows 0..M8)
  for (int64_t mi = 0; mi < M8; ++mi)
    for (int64_t k = 0; k < K; ++k) {
      float a_val = a[mi * K + k];
      for (int64_t nj = N8; nj < N; ++nj)
        out[mi * N + nj] += a_val * b[k * N + nj];
    }

  // bottom edge remainders (rows M8..M, all columns)
  for (int64_t mi = M8; mi < M; ++mi)
    for (int64_t k = 0; k < K; ++k) {
      float a_val = a[mi * K + k];
      for (int64_t nj = 0; nj < N; ++nj)
        out[mi * N + nj] += a_val * b[k * N + nj];
    }
#else
  gemm_blocked(a, b, out, M, N, K);
#endif
}

// gemm_cache_blocked (INACTIVE): BLIS-style MC/NC/KC blocking, no packing.
[[maybe_unused]] static void gemm_cache_blocked(const float* a, const float* b, float* out, int64_t M, int64_t N, int64_t K) {
#if TL_HAVE_NEON
  const int64_t MR = 8, NR = 8; // register tiles
  const int64_t MC = 128, NC = 256, KC = 256; // cache blocks (NC is largest because its on the outer cache levels)

  // zero once
  for (int64_t i = 0; i < M * N; ++i) out[i] = 0.0f;

  // full tile region
  const int64_t M8 = (M / MR) * MR;
  const int64_t N8 = (N / NR) * NR;

  // cache-blocking loops
  for (int64_t jc = 0; jc < N8; jc += NC) {
    int64_t jc_end = std::min(jc + NC, N8);
    for (int64_t pc = 0; pc < K; pc += KC) {
      int64_t kc = std::min(pc + KC, K) - pc;
      for (int64_t ic = 0; ic < M8; ic += MC) {
        int64_t ic_end = std::min(ic + MC, M8);

        // register-tiling loops
        for (int64_t m = ic; m < ic_end; m += MR) {
          for (int64_t n = jc; n < jc_end; n += NR) {
            micro_kernel_8x8_acc(a + m * K + pc, b + pc * N + n, out + m * N + n, kc, K, N);
          }
        }
      }
    }
  }

  // right edge remainders (columns)
  for (int64_t mi = 0; mi < M8; ++mi) {
    for (int64_t k = 0; k < K; ++k) {
      float a_val = a[mi * K + k];
      for (int64_t nj = N8; nj < N; ++nj) {
        out[mi * N + nj] += a_val * b[k * N + nj];
      }
    }
  }

  // bottom edge remainders (rows)
  for (int64_t mi = M8; mi < M; ++mi) {
    for (int64_t k = 0; k < K; ++k) {
      float a_val = a[mi * K + k];
      for (int64_t nj = 0; nj < N; ++nj) {
        out[mi * N + nj] += a_val * b[k * N + nj];
      }
    }
  }

#else
    gemm_blocked(a, b, out, M, N, K);
#endif
}

// PACKING: copy A/B blocks into contiguous buffers in micro-kernel read order.
// matmul dispatches large matmuls to gemm_packed below
// pack a kc x nc block of B -> 8-col panels [k0:n0..n7]...
static void pack_B(const float* b, float* b_pack, int64_t pc, int64_t kc, int64_t jc, int64_t nc, int64_t N) {
  const int64_t NR = 8;
  int64_t dst = 0;
  for (int64_t p = 0; p < nc; p += NR) {
    for (int64_t k = 0; k < kc; ++k) {
      const float* src = b + (pc + k) * N + (jc + p);
      for (int64_t col = 0; col < NR; ++col) {
        b_pack[dst++] = src[col];
      }
    }
  }
}

// pack a mc x kc block of A -> 8-row panels [k0:m0..m7]...
static void pack_A(const float* a, float* a_pack, int64_t pc, int64_t kc, int64_t ic, int64_t mc, int64_t K) {
  const int64_t MR = 8;
  int64_t dst = 0;
  for (int64_t q = 0; q < mc; q += MR) {
    for (int64_t k = 0; k < kc; ++k) {
      for (int64_t row = 0; row < MR; ++row) {
        a_pack[dst++] = a[(ic + q + row) * K + (pc + k)];
      }
    }
  }

}

#if TL_HAVE_NEON
// like _acc but reads contiguous panels (a_pack[k*8+row], b_pack[k*8+col])
static inline void micro_kernel_8x8_packed(const float* a_pack, const float* b_pack, float* c, int64_t kc, int64_t N) {
  float32x4_t c0l = vld1q_f32(c + 0*N + 0), c0r = vld1q_f32(c + 0*N + 4);
  float32x4_t c1l = vld1q_f32(c + 1*N + 0), c1r = vld1q_f32(c + 1*N + 4);
  float32x4_t c2l = vld1q_f32(c + 2*N + 0), c2r = vld1q_f32(c + 2*N + 4);
  float32x4_t c3l = vld1q_f32(c + 3*N + 0), c3r = vld1q_f32(c + 3*N + 4);
  float32x4_t c4l = vld1q_f32(c + 4*N + 0), c4r = vld1q_f32(c + 4*N + 4);
  float32x4_t c5l = vld1q_f32(c + 5*N + 0), c5r = vld1q_f32(c + 5*N + 4);
  float32x4_t c6l = vld1q_f32(c + 6*N + 0), c6r = vld1q_f32(c + 6*N + 4);
  float32x4_t c7l = vld1q_f32(c + 7*N + 0), c7r = vld1q_f32(c + 7*N + 4);

  for (int64_t k = 0; k < kc; ++k) {
    float32x4_t bl = vld1q_f32(b_pack + k*8 + 0);   // cols 0..3, contiguous
    float32x4_t br = vld1q_f32(b_pack + k*8 + 4);   // cols 4..7, contiguous
    const float* ap = a_pack + k*8;                 // 8 row-values for this k
    c0l = vfmaq_n_f32(c0l, bl, ap[0]); c0r = vfmaq_n_f32(c0r, br, ap[0]);
    c1l = vfmaq_n_f32(c1l, bl, ap[1]); c1r = vfmaq_n_f32(c1r, br, ap[1]);
    c2l = vfmaq_n_f32(c2l, bl, ap[2]); c2r = vfmaq_n_f32(c2r, br, ap[2]);
    c3l = vfmaq_n_f32(c3l, bl, ap[3]); c3r = vfmaq_n_f32(c3r, br, ap[3]);
    c4l = vfmaq_n_f32(c4l, bl, ap[4]); c4r = vfmaq_n_f32(c4r, br, ap[4]);
    c5l = vfmaq_n_f32(c5l, bl, ap[5]); c5r = vfmaq_n_f32(c5r, br, ap[5]);
    c6l = vfmaq_n_f32(c6l, bl, ap[6]); c6r = vfmaq_n_f32(c6r, br, ap[6]);
    c7l = vfmaq_n_f32(c7l, bl, ap[7]); c7r = vfmaq_n_f32(c7r, br, ap[7]);
  }

  vst1q_f32(c + 0*N + 0, c0l); vst1q_f32(c + 0*N + 4, c0r);
  vst1q_f32(c + 1*N + 0, c1l); vst1q_f32(c + 1*N + 4, c1r);
  vst1q_f32(c + 2*N + 0, c2l); vst1q_f32(c + 2*N + 4, c2r);
  vst1q_f32(c + 3*N + 0, c3l); vst1q_f32(c + 3*N + 4, c3r);
  vst1q_f32(c + 4*N + 0, c4l); vst1q_f32(c + 4*N + 4, c4r);
  vst1q_f32(c + 5*N + 0, c5l); vst1q_f32(c + 5*N + 4, c5r);
  vst1q_f32(c + 6*N + 0, c6l); vst1q_f32(c + 6*N + 4, c6r);
  vst1q_f32(c + 7*N + 0, c7l); vst1q_f32(c + 7*N + 4, c7r);
}
#endif

// gemm_packed (ACTIVE for large matmuls): cache blocking + packing (full BLIS shape)
[[maybe_unused]] static void gemm_packed(const float* a, const float* b, float* out, int64_t M, int64_t N, int64_t K) {
#if TL_HAVE_NEON
  const int64_t MR = 8, NR = 8;
  const int64_t MC = 128, NC = 256, KC = 256;

  for (int64_t i = 0; i < M * N; ++i) out[i] = 0.0f;   // packed kernel accumulates

  const int64_t M8 = (M / MR) * MR;
  const int64_t N8 = (N / NR) * NR;

  std::vector<float> A_pack(MC * KC);
  std::vector<float> B_pack(KC * NC);

  for (int64_t jc = 0; jc < N8; jc += NC) {
    int64_t nc = std::min(NC, N8 - jc);
    for (int64_t pc = 0; pc < K; pc += KC) {
      int64_t kc = std::min(KC, K - pc);
      pack_B(b, B_pack.data(), pc, kc, jc, nc, N);          // pack B block once per (jc,pc)
      for (int64_t ic = 0; ic < M8; ic += MC) {
        int64_t mc = std::min(MC, M8 - ic);
        pack_A(a, A_pack.data(), pc, kc, ic, mc, K);        // pack A block once per (jc,pc,ic)

        for (int64_t m = ic; m < ic + mc; m += MR) {
          const float* a_panel = A_pack.data() + ((m - ic) / MR) * kc * MR;
          for (int64_t n = jc; n < jc + nc; n += NR) {
            const float* b_panel = B_pack.data() + ((n - jc) / NR) * kc * NR;
            micro_kernel_8x8_packed(a_panel, b_panel, out + m * N + n, kc, N);
          }
        }
      }
    }
  }

  // right edge remainders
  for (int64_t mi = 0; mi < M8; ++mi) {
    for (int64_t k = 0; k < K; ++k) {
      float a_val = a[mi * K + k];
      for (int64_t nj = N8; nj < N; ++nj) {
        out[mi * N + nj] += a_val * b[k * N + nj];
      }
    }
  }

  // bottom edge remainders
  for (int64_t mi = M8; mi < M; ++mi) {
    for (int64_t k = 0; k < K; ++k) {
      float a_val = a[mi * K + k];
      for (int64_t nj = 0; nj < N; ++nj) {
        out[mi * N + nj] += a_val * b[k * N + nj];
      }
    }
  }
#else
  gemm_blocked(a, b, out, M, N, K);
#endif
}

// gemm_mt: parallel driver for multithreading the gemm kernel
static void gemm_mt(const float* a, const float* b, float* out, int64_t M, int64_t N, int64_t K) {
  unsigned nthreads = std::thread::hardware_concurrency();
  if (nthreads == 0) nthreads = 1;

  int64_t rows_per = (M + nthreads - 1) / nthreads; // ceiling split

  std::vector<std::thread> pool;
  for (unsigned t = 0; t < nthreads; ++t) {
    int64_t m_start = (int64_t)t * rows_per;
    if (m_start >= M) break;
    int64_t m_end = std::min(m_start + rows_per, M);
    int64_t band = m_end - m_start;

    pool.emplace_back([=]() {
      gemm_packed(a + m_start * K, b, out + m_start * N, band, N, K);
    });
  }
  for (auto& th: pool) th.join();
}

// gemm_mt_omp (INACTIVE): parallel driver for multithreading the gemm kernel using OpenMP
static void gemm_mt_omp(const float* a, const float* b, float* out, int64_t M, int64_t N, int64_t K) {
  unsigned nthreads = omp_get_max_threads();
  int64_t rows_per = (M + nthreads - 1) / nthreads;

  #pragma omp parallel for
  for (int t = 0; t < nthreads; ++t) {
    int64_t m_start = (int64_t)t * rows_per;
    if (m_start >= M) continue;
    int64_t m_end = std::min(m_start + rows_per, M);
    int64_t band = m_end - m_start;
    gemm_packed(a + m_start * K, b, out + m_start * N, band, N, K);
  }
}

// matrix multiplication
Tensor matmul(const Tensor& a_in, const Tensor& b_in) {
  Tensor a = a_in.contiguous();
  Tensor b = b_in.contiguous();

  // squeeze and track 1D matrices
  bool squeeze_a = false, squeeze_b = false;
  if (a.sizes().size() == 1) {
    a = reshape(a, {1, a.sizes()[0]}); // [K] -> [1, K]
    squeeze_a = true;
  }
  if (b.sizes().size() == 1) {
    b = reshape(b, {b.sizes()[0], 1}); // [K] -> [K, 1]
    squeeze_b = true;
  }

  // handle 0D tensors
  if (a.sizes().size() < 2 || b.sizes().size() < 2) {
    throw std::invalid_argument("matmul: requires at least 1D tensors");
  }

  const auto& s_a = a.sizes();
  const auto& s_b = b.sizes();

  // identify matrix dims
  int64_t M = s_a[s_a.size() - 2];
  int64_t K = s_a[s_a.size() - 1];
  int64_t K2 = s_b[s_b.size() - 2];
  int64_t N = s_b[s_b.size() - 1];

  if (K != K2) {
    throw std::invalid_argument("Incompatible dimensions for matmul");
  }

  // determine batch dimensions
  std::vector<int64_t> batch_a(s_a.begin(), s_a.end() - 2);
  std::vector<int64_t> batch_b(s_b.begin(), s_b.end() - 2);
  std::vector<int64_t> batch_out = compute_broadcast_shape(batch_a, batch_b);

  const auto& strides_a = a.strides();
  const auto& strides_b = b.strides();
  std::vector<int64_t> strides_a_batch(strides_a.begin(), strides_a.end() - 2);
  std::vector<int64_t> strides_b_batch(strides_b.begin(), strides_b.end() - 2);

  // construct final output shape
  std::vector<int64_t> out_shape = batch_out;
  out_shape.push_back(M);
  out_shape.push_back(N);
  Tensor out(out_shape);

  // batched execution
  int64_t num_batches = 1;
  for (int64_t dim: batch_out) {
    num_batches *= dim;
  }

  for (int64_t i = 0; i < num_batches; ++i) {
    int64_t index_a = get_broadcast_index(i, batch_a, strides_a_batch, batch_out);
    int64_t index_b = get_broadcast_index(i, batch_b, strides_b_batch, batch_out);

    const float* ap = a.data() + index_a;
    const float* bp = b.data() + index_b;
    float* op = out.data() + (i * M * N);

    if (M >= 256 && N >= 256 && K >= 256) {
      gemm_mt(ap, bp, op, M, N, K);
      // gemm_mt_omp(ap, bp, op, M, N, K);
    } else {
      gemm_tiled(ap, bp, op, M, N, K);
    }
  }

  Tensor result = out;
  if (squeeze_a && squeeze_b) { // both squeezed
    result = reshape(out, {}); // [1, K] @ [K, 1] -> [1, 1] -> scalar
  } else if (squeeze_a) { // only first input matrix squeezed
    std::vector<int64_t> s(out.sizes().begin(), out.sizes().end());
    s.erase(s.end() - 2); // [..., 1, N] @ [..., N] remove fake row dim
    result = reshape(out, s);
  } else if (squeeze_b) { // only second input matrix squeezed
    std::vector<int64_t> s(out.sizes().begin(), out.sizes().end());
    s.erase(s.end() - 1); // [..., M, 1] @ [..., M], remove fake col dim
    result = reshape(out, s);
  }

  if (a_in.requires_grad || b_in.requires_grad) {
    if (auto fn = track<MatmulBackward>(result, {&a_in, &b_in})) {
      fn->a_cache = a_in.contiguous();
      fn->b_cache = b_in.contiguous();
      fn->squeeze_a = squeeze_a;
      fn->squeeze_b = squeeze_b;
    }
  }

  return result;
}

// matrix transpose
Tensor transpose(const Tensor& a, int64_t dim0, int64_t dim1) {
  int64_t ndim = a.sizes().size();

  // negative dim wrapping
  if (dim0 < 0) dim0 += ndim;
  if (dim1 < 0) dim1 += ndim;

  if (dim0 < 0 || dim0 >= ndim || dim1 < 0 || dim1 >= ndim) {
    throw std::invalid_argument("tranpose: dimension out of range");
  }
  auto out_sizes = a.sizes();
  auto out_strides = a.strides();

  // simply swap the metadata
  std::swap(out_sizes[dim0], out_sizes[dim1]);
  std::swap(out_strides[dim0], out_strides[dim1]);

  Tensor out(a.data_, out_sizes, out_strides, a.offset_);

  if (a.requires_grad) {
    if (auto fn = track<TransposeBackward>(out, {&a})) {
      fn->dim0 = dim0;
      fn->dim1 = dim1;
    }
  }

  // use private constructor to create new view
  return out;
}

Tensor reshape(const Tensor& a, const std::vector<int64_t>& new_sizes) {
  // verify total elements match
  int64_t new_numel = 1;
  for (int64_t s: new_sizes) {
    if (s < 0) {
      throw std::invalid_argument("reshape: negative dims not allowed");
    }
    new_numel *= s;
  }

  if (new_numel != a.numel()) {
    throw std::invalid_argument("reshape: new shape must have same number of elements");
  }

  // make sure tensor is contiguous
  Tensor c = a.is_contiguous() ? a : a.contiguous();

  // compute new strides of new shape
  std::vector<int64_t> new_strides(new_sizes.size());
  int64_t stride = 1;
  for (int i = static_cast<int>(new_sizes.size()) - 1; i >= 0; --i) {
    new_strides[i] = stride;
    stride *= new_sizes[i];
  }

  Tensor out(c.data_, new_sizes, new_strides, c.offset_);

  if (a.requires_grad) {
    if (auto fn = track<ReshapeBackward>(out, {&a})) {
      fn->input_shape = a.sizes();
    }
  }

  // create new view using private constructor
  return out;
}

// tensor concatenation along existing dimension
Tensor cat(const std::vector<Tensor>& tensors, int64_t dim) {
  if (tensors.empty()) {
    throw std::invalid_argument("cat: empty tensor list");
  }

  int64_t ndim = tensors[0].sizes().size();
  if (dim < 0) dim += ndim;
  if (dim < 0 || dim >= ndim) {
    throw std::invalid_argument("cat: dimension out of range");
  }

  // validate shapes match except along given dim
  for (size_t t = 1; t < tensors.size(); ++t) {
    if ((int64_t)tensors[t].sizes().size() != ndim) {
      throw std::invalid_argument("cat: all tensors must have same number of dimensions");
    }

    for (int64_t d = 0; d < ndim; ++d) {
      if (d != dim && tensors[t].sizes()[d] != tensors[0].sizes()[d]) {
        throw std::invalid_argument("cat: shapes must match except in cat dimension");
      }
    }
  }

  // compute output shape
  std::vector<int64_t> out_sizes(tensors[0].sizes().begin(), tensors[0].sizes().end());
  int64_t total_cat_size = 0;
  for (const auto& t: tensors) {
    total_cat_size += t.sizes()[dim];
  }
  out_sizes[dim] = total_cat_size;

  Tensor out(out_sizes);
  float* op = out.data();

  // outer = product of dimensions before dim
  // inner = product of dimensions after dim
  int64_t outer = 1, inner = 1;
  for (int64_t i = 0; i < dim; ++i) outer *= out_sizes[i];
  for (int64_t i = dim+1; i < ndim; ++i) inner *= out_sizes[i];

  int64_t cat_offset = 0;
  for (const auto& t: tensors) {
    Tensor tc = t.contiguous();
    const float* tp = tc.data();
    int64_t t_dim_size = tc.sizes()[dim];

    for (int64_t o = 0; o < outer; ++o) {
      for (int64_t d = 0; d < t_dim_size; ++d) {
        for (int64_t n = 0; n < inner; ++n) {
          op[o * total_cat_size * inner + (cat_offset + d) * inner + n] = tp[o * t_dim_size * inner + d * inner + n];
        }
      }
    }
    cat_offset += t_dim_size;
  }
  return out;
}

// tensor stacking along a new dimension
Tensor stack(const std::vector<Tensor>& tensors, int64_t dim) {
  if (tensors.empty()) {
    throw std::invalid_argument("stack: empty tensor list");
  }

  int64_t ndim = tensors[0].sizes().size();
  if (dim < 0) dim += ndim + 1;
  if (dim < 0 || dim > ndim) {
    throw std::invalid_argument("stack: dimension out of range");
  }

  // validate all tensors have the same shape
  for (size_t t = 1; t < tensors.size(); ++t) {
    if (tensors[t].sizes() != tensors[0].sizes()) {
      throw std::invalid_argument("stack: all tensors must have same shape");
    }
  }

  // unsqueeze each tensor at given dimension, then concatenate
  std::vector<Tensor> unsqueezed;
  for (const auto& t: tensors) {
    std::vector<int64_t> new_sizes(t.sizes().begin(), t.sizes().end());
    new_sizes.insert(new_sizes.begin() + dim, 1);
    unsqueezed.push_back(reshape(t, new_sizes));
  }

  return cat(unsqueezed, dim);
}

// tensor slicing along a given dimension
Tensor slice(const Tensor& input, int64_t dim, int64_t start, int64_t end) {
  Tensor a = input.contiguous();
  const auto& sizes = a.sizes();
  int64_t ndim = sizes.size();

  if (dim < 0) dim += ndim;
  if (dim < 0 || dim >= ndim) {
    throw std::invalid_argument("slice: dimension out of range");
  }
  if (start < 0 || end > sizes[dim] || start >= end) {
    throw std::invalid_argument("slice: invalid start/end range");
  }

  // build output shape
  std::vector<int64_t> out_sizes(sizes.begin(), sizes.end());
  out_sizes[dim] = end - start;

  Tensor out(out_sizes);
  float* op = out.data();
  const float* ap = a.data();

  int64_t outer = 1, inner = 1;
  int64_t D = sizes[dim];
  int64_t slice_size = end - start;
  for (int64_t i = 0; i < dim; ++i) outer *= sizes[i];
  for (int64_t i = dim+1; i < ndim; ++i) inner *= sizes[i];

  for (int64_t o = 0; o < outer; ++o) {
    for (int64_t d = 0; d < slice_size; ++d) {
      for (int64_t n = 0; n < inner; ++n) {
        op[o * slice_size * inner + d * inner + n] = ap[o * D * inner + (start + d) * inner + n];
      }
    }
  }
  return out;
}

// scale a tensor by a factor (scalar)
Tensor scale(const Tensor& input, float scalar) {
  Tensor a = input.contiguous();
  Tensor out(a.sizes());

  const float* ap = a.data();
  float* op = out.data();

  const int64_t n = a.numel();
  for (int64_t i = 0; i < n; ++i) {
    op[i] = ap[i] * scalar;
  }

  if (input.requires_grad) {
    if (auto fn = track<ScaleBackward>(out, {&input})) {
      fn->scalar = scalar;
    }
  }

  return out;
}

// softmax, squeeze everything between 0-1 (into probabilities) in the last dim
Tensor softmax(const Tensor& input) {
  Tensor a = input.contiguous();
  const auto& sizes = a.sizes();
  if (sizes.empty()) {
    throw std::invalid_argument("Softmax requires at least 1D tensor");
  }

  if (sizes.back() == 0) {
    return Tensor(sizes); // no last dim to normalize
  }

  Tensor out(sizes);

  const float* ap = a.data();
  float* op = out.data();

  const int64_t D = sizes.back();
  const int64_t outer = a.numel() / D;

  for (int64_t i = 0; i < outer; ++i) {
    const float* row = ap + i * D;
    float* out_row = op + i * D;

    // find maximum
    float max_val = row[0];
    for (int64_t j = 1; j < D; ++j) {
      if (row[j] > max_val) {
        max_val = row[j];
      }
    }

    // exp(x - max) and sum
    float sum = 0.0f;
    for (int64_t j = 0; j < D; ++j) {
      out_row[j] = std::exp(row[j] - max_val);
      sum += out_row[j];
    }

    // normalize
    for (int64_t j = 0; j < D; ++j) {
      out_row[j] /= sum;
    }

  }

  if (input.requires_grad) {
    if (auto fn = track<SoftmaxBackward>(out, {&input})) {
      fn->output_cache = out.contiguous();
    }
  }

  return out;
}

// flash attention forward: tiled Q @ K^T -> softmax -> scores @ V
Tensor flash_attention(const Tensor& Q, const Tensor& K, const Tensor& V, float sm_scale) {
  const int64_t S = Q.sizes()[0];
  const int64_t d = Q.sizes()[1];

  Tensor Qc = Q.contiguous(), Kc = K.contiguous(), Vc = V.contiguous();
  const float* Qp = Qc.data();
  const float* Kp = Kc.data();
  const float* Vp = Vc.data();

  Tensor O({S, d});
  float* Op = O.data();
  Tensor L({S});
  float* Lp = L.data();

  const int64_t Bq = 64, Bk = 64; // query and key block sizes

  std::vector<float> T(Bq * Bk); // score tile
  std::vector<float> m(Bq), l(Bq); // running max (m), running sum (l)
  std::vector<float> Oacc(Bq * d); // output accumulator

  for (int64_t i0 = 0; i0 < S; i0 += Bq) { // query blocks
    int64_t qrows = std::min(Bq, S - i0);

    // reset running state
    for (int64_t qi = 0; qi < qrows; ++qi) {
      m[qi] = -std::numeric_limits<float>::infinity();
      l[qi] = 0.0f;
      for (int64_t c = 0; c < d; ++c) Oacc[qi * d + c] = 0.0f;
    }

    // key and value blocks
    for (int64_t j0 = 0; j0 < S; j0 += Bk) {
      int64_t krows = std::min(Bk, S - j0);

      // T = Q_i @ K_j^T * sm_scale -> tile[qrows, krows]
      for (int64_t qi = 0; qi < qrows; ++qi) {
        const float* qrow = Qp + (i0 + qi) * d;
        for (int64_t kj = 0; kj < krows; ++kj) {
          const float* krow = Kp + (j0 + kj) * d;
          float dot = 0.0f;
          for (int64_t c = 0; c < d; ++c) dot += qrow[c] * krow[c];
          T[qi * Bk + kj] = dot * sm_scale;
        }
      }

      // online softmax
      for (int64_t qi = 0; qi < qrows; ++qi) {
        float m_block = -std::numeric_limits<float>::infinity();
        for (int64_t kj = 0; kj < krows; ++kj) {
          m_block = std::max(m_block, T[qi * Bk + kj]);
        }

        float m_new = std::max(m[qi], m_block);
        float corr = std::exp(m[qi] - m_new);

        float row_sum = 0.0f;
        for (int64_t kj = 0; kj < krows; ++kj) {
          float p = std::exp(T[qi * Bk + kj] - m_new);
          T[qi * Bk + kj] = p;
          row_sum += p;
        }
        l[qi] = l[qi] * corr + row_sum;

        // update output
        float* oacc = Oacc.data() + qi * d;
        for (int64_t c = 0; c < d; ++c) oacc[c] *= corr;
        for (int64_t kj = 0; kj < krows; ++kj) {
          float p = T[qi * Bk + kj];
          const float* vrow = Vp + (j0 + kj) * d;
          for (int64_t c = 0; c < d; ++c) oacc[c] += p * vrow[c];
        }

        m[qi] = m_new;
      }
    }

    // normalize by running sum
    for (int64_t qi = 0; qi < qrows; ++qi) {
      const float* oacc = Oacc.data() + qi * d;
      float inv = 1.0f / l[qi];
      float* orow = Op + (i0 + qi) * d;
      for (int64_t c = 0; c < d; ++c) orow[c] = oacc[c] * inv;
      Lp[i0 + qi] = m[qi] + std::log(l[qi]);
    }
  }

  if (Q.requires_grad || K.requires_grad || V.requires_grad) {
    if (auto fn = track<FlashAttentionBackward>(O, {&Q, &K, &V})) {
      fn->Q_cache = Qc;
      fn->K_cache = Kc;
      fn->V_cache = Vc;
      fn->O_cache = O;
      fn->L_cache = L;
      fn->sm_scale = sm_scale;
    }
  }
  return O;
}

// argmax, get index of maximum value along a given dimension
Tensor argmax(const Tensor& input, int64_t dim) {
  if (dim < 0) dim += input.sizes().size(); // dimension wrapping

  const auto& sizes = input.sizes();
  int64_t ndim = sizes.size();

  // output shape is input shape with target dim collapsed
  std::vector<int64_t> out_sizes;
  for (int64_t i = 0; i < ndim; ++i) {
    if (i != dim) out_sizes.push_back(sizes[i]);
  }
  if (out_sizes.empty()) out_sizes.push_back(1); // scalar case

  Tensor out(out_sizes);
  Tensor c = input.contiguous();
  const float* ap = c.data();
  float* op = out.data();

  int64_t outer = 1, inner = 1, D = sizes[dim];
  for (int64_t i = 0; i < dim; ++i) outer *= sizes[i];
  for (int64_t i = dim + 1; i < ndim; ++i) inner *= sizes[i];

  for (int64_t o = 0; o < outer; ++o) {
    for (int64_t i = 0; i < inner; ++i) {
      float best = ap[o * D * inner + i];
      int64_t best_index = 0;
      for (int64_t d = 1; d < D; ++d) {
        float val = ap[o * D * inner + d * inner + i];
        if (val > best) {
          best = val;
          best_index = d;
        }
      }
      op[o * inner + i] = static_cast<float>(best_index);
    }
  }

  return out;
}

// sum along the dimension of a tensor
Tensor sum(const Tensor& input, int64_t dim, bool keepdim) {
  Tensor a = input.contiguous();
  const auto& sizes = a.sizes();
  int64_t ndim = sizes.size();

  if (dim < 0) dim += ndim; // negative dim wrapping

  if (dim < 0 || dim >= ndim) {
    throw std::invalid_argument("sum: dimension out of range");
  }

  // build output shape
  std::vector<int64_t> out_sizes;
  for (int64_t i = 0; i < ndim; ++i) {
    if (i == dim) {
      if (keepdim) out_sizes.push_back(1);
    } else {
      out_sizes.push_back(sizes[i]);
    }
  }

  Tensor out(out_sizes);
  float* op = out.data();
  const float* ap = a.data();

  // outer = product of dims before dim
  // D = size of dim being reduced
  // inner = product of dims after dim
  int64_t outer = 1, inner = 1;
  int64_t D = sizes[dim];
  for (int64_t i = 0; i < dim; ++i) outer *= sizes[i];
  for (int64_t i = dim + 1; i < ndim; ++i) inner *= sizes[i];

  for (int64_t o = 0; o < outer; ++o) {
    for (int64_t n = 0; n < inner; ++n) {
      float acc = 0.0f;
      for (int64_t d = 0; d < D; ++d) {
        acc += ap[o * D * inner + d * inner + n];
      }
      op[o * inner + n] = acc;
    }
  }

  if (input.requires_grad) {
    if (auto fn = track<SumBackward>(out, {&input})) {
      fn->input_shape = input.sizes();
      fn->dim_ = dim;
      fn->keepdim_ = keepdim;
    }
  }

  return out;
}

// calculate mean of a dimension
Tensor mean(const Tensor& input, int64_t dim, bool keepdim) {
  // negative dim wrapping
  int64_t ndim = input.sizes().size();
  if (dim < 0) dim += ndim;

  Tensor s = sum(input, dim, keepdim);
  int64_t D = input.sizes()[dim];
  if (D == 0) {
    throw std::invalid_argument("mean: cannot reduce a zero-sized dimension");
  }

  return scale(s, 1.0f / static_cast<float>(D));
}

// element-wise absolute value
Tensor abs(const Tensor& input) {
  Tensor a = input.contiguous();
  Tensor out(a.sizes());
  const float* ap = a.data();
  float* op = out.data();

  for (int64_t i = 0; i < a.numel(); ++i) {
    op[i] = std::abs(ap[i]);
  }

  if (input.requires_grad) {
    if (auto fn = track<AbsBackward>(out, {&input})) {
      fn->input_cache = input.contiguous();
    }
  }

  return out;
}

// calculate the variance of a dimension
// var = mean((x - mean)^2)
Tensor variance(const Tensor& input, int64_t dim, bool keepdim) {
  int64_t ndim = input.sizes().size();
  if (dim < 0) dim += ndim; // dim wrapping

  Tensor m = mean(input, dim, true);
  Tensor diff = sub(input, m);
  Tensor sq = mul(diff, diff);
  Tensor var = mean(sq, dim, keepdim);
  return var;
}

Tensor neg(const Tensor& input) {
  Tensor a = input.contiguous();
  Tensor out(a.sizes());
  const float* ap = a.data();
  float* op = out.data();

  const int64_t n = a.numel();
  for (int64_t i = 0; i < n; ++i) {
    op[i] = -ap[i];
  }

  if (input.requires_grad) {
    track<NegBackward>(out, {&input});
  }

  return out;
}

Tensor exp(const Tensor& input) {
  Tensor a = input.contiguous();
  Tensor out(a.sizes());
  const float* ap = a.data();
  float* op = out.data();

  const int64_t n = a.numel();
  for (int64_t i = 0; i < n; ++i) {
    op[i] = std::exp(ap[i]);
  }

  if (input.requires_grad) {
    if (auto fn = track<ExpBackward>(out, {&input})) {
      fn->output_cache = out.contiguous();
    }
  }

  return out;
}

Tensor pow(const Tensor& input, float x) {
  Tensor a = input.contiguous();
  Tensor out(a.sizes());
  const float* ap = a.data();
  float* op = out.data();

  const int64_t n = a.numel();
  for (int64_t i = 0; i < n; ++i) {
    op[i] = std::pow(ap[i], x);
  }

  if (input.requires_grad) {
    if (auto fn = track<PowBackward>(out, {&input})) {
      fn->input_cache = input.contiguous();
      fn->exponent = x;
    }
  }

  return out;
}

Tensor log(const Tensor& input) {
  Tensor a = input.contiguous();
  Tensor out(a.sizes());
  const float* ap = a.data();
  float* op = out.data();

  const int64_t n = a.numel();
  for (int64_t i = 0; i < n; ++i) {
    op[i] = std::log(ap[i]);
  }

  if (input.requires_grad) {
    if (auto fn = track<LogBackward>(out, {&input})) {
      fn->input_cache = input.contiguous();
    }
  }

  return out;
}

Tensor clamp(const Tensor& input, float min_val, float max_val) {
  Tensor a = input.contiguous();
  Tensor out(a.sizes());
  const float* ap = a.data();
  float* op = out.data();

  const int64_t n = a.numel();
  for (int64_t i = 0; i < n; ++i) {
    op[i] = std::max(min_val, std::min(max_val, ap[i]));
  }

  if (input.requires_grad) {
    if (auto fn = track<ClampBackward>(out, {&input})) {
      fn->input_cache = input.contiguous();
      fn->min_val = min_val;
      fn->max_val = max_val;
    }
  }

  return out;
}

// pad a dimension with a given value to a target length
Tensor pad(const Tensor& input, int64_t dim, int64_t target_len, float value) {
  if (dim < 0) dim += input.sizes().size();

  const auto& sizes = input.sizes();
  int64_t ndim = sizes.size();
  int64_t current = sizes[dim];

  if (target_len <= current) return input.contiguous();

  // build output tensor shape
  std::vector<int64_t> out_sizes(sizes.begin(), sizes.end());
  out_sizes[dim] = target_len;

  Tensor out = full(out_sizes, value);
  Tensor c = input.contiguous();
  const float* ap = c.data();
  float* op = out.data();

  int64_t outer = 1, inner = 1;
  for (int64_t i = 0; i < dim; ++i) outer *= sizes[i];
  for (int64_t i = dim + 1; i < ndim; ++i) inner *= sizes[i];

  for (int64_t o = 0; o < outer; ++o) {
    for (int64_t d = 0; d < current; ++d) { // copy up to original len, rest remains as pad value
      for (int64_t i = 0; i < inner; ++i) {
        op[o * target_len * inner + d * inner + i] = ap[o * current * inner + d * inner + i];
      }
    }
  }

  return out;
}

// convolution of a filter tensor over an input tensor using im2col (unroll inputs into one big matmul)
Tensor conv2d(const Tensor& input, const Tensor& weight, const Tensor& bias, int64_t stride, int64_t padding, int64_t groups) {
  const int64_t N = input.sizes()[0];
  const int64_t C_in = input.sizes()[1];
  const int64_t H = input.sizes()[2];
  const int64_t W = input.sizes()[3];

  const int64_t C_out = weight.sizes()[0];
  const int64_t kH = weight.sizes()[2];
  const int64_t kW = weight.sizes()[3];

  // validate groups
  if (groups <= 0) throw std::invalid_argument("conv2d: groups must be > 0");
  if (C_in % groups != 0) throw std::invalid_argument("conv2d: groups doesn't divide C_in");
  if (C_out % groups != 0) throw std::invalid_argument("conv2d: groups doesn't divide C_out");
  if (weight.sizes()[1] != C_in / groups) throw std::invalid_argument("conv2d: weight.sizes()[1] must equal C_in / groups");

  // per-group channel counts
  const int64_t C_in_g = C_in / groups;
  const int64_t C_out_g = C_out / groups;

  // remove edge pixels with padding in consideration
  const int64_t H_out = (H + 2 * padding - kH) / stride + 1;
  const int64_t W_out = (W + 2 * padding - kW) / stride + 1;

  // weight: (C_out, C_in_g, kH, kW)
  // col: (C_in_g * kH * kW, H_out * W_out)
  Tensor w = reshape(weight, {C_out, C_in_g*kH*kW});
  std::vector<Tensor> w_groups;
  // pre-slice weight rows by groups
  w_groups.reserve(groups);
  for (int64_t g = 0; g < groups; ++g) {
    w_groups.push_back(slice(w, 0, g * C_out_g, (g + 1) * C_out_g));
  }
  Tensor out({N, C_out, H_out*W_out});
  const float* ip = input.data();

  for (int64_t n = 0; n < N; ++n) { // iterate over images in a batch
    // im2col for image n only: (C_in*kH*kW, H_out*W_out)
    for (int64_t g = 0; g < groups; ++g) {
      Tensor col_ng({C_in_g * kH * kW, H_out * W_out});
      float* cp = col_ng.data();
      for (int64_t c = 0; c < C_in_g; ++c) { // iterate over each input channel
        int64_t c_actual = g * C_in_g + c;
        const float* in_ch = ip + n * (C_in * H * W) + c_actual * (H * W);
        for (int64_t kh = 0; kh < kH; ++kh) { // iterate over each kernel row
          for (int64_t kw = 0; kw < kW; ++kw) { // iterate over each kernel col
            int64_t col_row = c * kH * kW + kh * kW + kw;
            float* col_row_ptr = cp + col_row * (H_out * W_out);
            int64_t ow_start = 0;
            while (ow_start < W_out && (ow_start * stride - padding + kw) < 0) ++ow_start;
            int64_t ow_end = W_out;
            while (ow_end > 0 && ((ow_end - 1) * stride - padding + kw) >= W) --ow_end;
            for (int64_t oh = 0; oh < H_out; ++oh) { // iterate over output height
              int64_t ih = oh * stride - padding + kh;
              float* out_row = col_row_ptr + oh * W_out;
              if (ih < 0 || ih >= H) {
                std::fill(out_row, out_row + W_out, 0.0f);
                continue;
              }
              const float* in_row = in_ch + ih * W;

              // zero the left/right padding edges
              std::fill(out_row, out_row + ow_start, 0.0f);
              std::fill(out_row + ow_end, out_row + W_out, 0.0f);

              for (int64_t ow = ow_start; ow < ow_end; ++ow) {
                int64_t iw = ow * stride - padding + kw;
                out_row[ow] = in_row[iw];
              }
            }
          }
        }
      }

      // matmul per image: w @ col[n] -> (C_out, H_out * W_out)
      Tensor out_ng = matmul(w_groups[g], col_ng);
      // copy into out[n]
      float* op = out.data() + n * C_out * H_out * W_out + g * C_out_g * H_out * W_out;
      const float* onp = out_ng.data();
      for (int64_t i = 0; i < C_out_g * H_out * W_out; ++i) op[i] = onp[i];
    }
  }
  Tensor result = reshape(out, {N, C_out, H_out, W_out});

  // add bias
  if (!bias.empty()) {
    result = add(result, reshape(bias, {1, C_out, 1, 1}));
  }

  if (input.requires_grad || weight.requires_grad) {
    if (auto fn = track<Conv2dBackward>(result, {&input, &weight, &bias})) {
      fn->weight_cache = weight.contiguous();
      fn->input_cache = input.contiguous();
      fn->stride = stride;
      fn->padding = padding;
      fn->groups = groups;
      fn->N = N;
      fn->C_in = C_in;
      fn->H = H;
      fn->W = W;
    }
  }

  return result;
}

// max pooling: slide a square window over the last 2 dims of the tensor and take maximum in the window
Tensor max_pool2d(const Tensor& input, int64_t kernel_size, int64_t stride, int64_t padding) {
  if (stride == 0) stride = kernel_size;

  const int64_t N = input.sizes()[0];
  const int64_t C = input.sizes()[1];
  const int64_t H = input.sizes()[2];
  const int64_t W = input.sizes()[3];

  const int64_t H_out = (H + 2 * padding - kernel_size) / stride + 1;
  const int64_t W_out = (W + 2 * padding - kernel_size) / stride + 1;

  Tensor out({N, C, H_out, W_out});
  const float* ip = input.data();
  float* op = out.data();

  std::vector<int64_t> argmax_indices(N * C * H_out * W_out, -1);

  for (int64_t n = 0; n < N; ++n) {
    for (int64_t c = 0; c < C; ++c) {
      for (int64_t oh = 0; oh < H_out; ++oh) {
        for (int64_t ow = 0; ow < W_out; ++ow) {
          int64_t ih_start = oh * stride - padding;
          int64_t iw_start = ow * stride - padding;

          float max_val = -std::numeric_limits<float>::infinity();
          int64_t arg = -1;
          for (int64_t kh = 0; kh < kernel_size; ++kh) {
            for (int64_t kw = 0; kw < kernel_size; ++kw) {
              int64_t ih = ih_start + kh;
              int64_t iw = iw_start + kw;
              if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;
              int64_t in_i = n*(C*H*W) + c*(H*W) + ih*W + iw;
              float v = ip[in_i];
              if (v > max_val) {
                max_val = v;
                arg = in_i;
              }
            }
          }

          int64_t out_i = n*(C*H_out*W_out) + c*(H_out*W_out) + oh*W_out + ow;
          op[out_i] = (arg >= 0) ? max_val : 0.0f;
          argmax_indices[out_i] = arg;
        }
      }
    }
  }

  if (input.requires_grad) {
    if (auto fn = track<MaxPool2dBackward>(out, {&input})) {
      fn->argmax_indices = std::move(argmax_indices);
      fn->N = N;
      fn->C = C;
      fn->H = H;
      fn->W = W;
    }
  }

  return out;
}


// avg pooling: slide a square window over the last 2 dims of the tensor and take average of the window
Tensor avg_pool2d(const Tensor& input, int64_t kernel_size, int64_t stride, int64_t padding) {
  if (stride == 0) stride = kernel_size;

  const int64_t N = input.sizes()[0];
  const int64_t C = input.sizes()[1];
  const int64_t H = input.sizes()[2];
  const int64_t W = input.sizes()[3];

  const int64_t H_out = (H + 2 * padding - kernel_size) / stride + 1;
  const int64_t W_out = (W + 2 * padding - kernel_size) / stride + 1;
  const float window_area = static_cast<float> (kernel_size * kernel_size);

  Tensor out({N, C, H_out, W_out});
  const float* ip = input.data();
  float* op = out.data();

  for (int64_t n = 0; n < N; ++n) {
    for (int64_t c = 0; c < C; ++c) {
      for (int64_t oh = 0; oh < H_out; ++oh) {
        for (int64_t ow = 0; ow < W_out; ++ow) {
          int64_t ih_start = oh * stride - padding;
          int64_t iw_start = ow * stride - padding;

          float sum = 0.0f;
          for (int64_t kh = 0; kh < kernel_size; ++kh) {
            for (int64_t kw = 0; kw < kernel_size; ++kw) {
              int64_t ih = ih_start + kh;
              int64_t iw = iw_start + kw;
              if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;
              sum += ip[n*(C*H*W) + c*(H*W) + ih*W + iw];
            }
          }

          op[n*(C*H_out*W_out) + c*(H_out*W_out) + oh*W_out + ow] = sum / window_area;
        }
      }
    }
  }


  if (input.requires_grad) {
    if (auto fn = track<AvgPool2dBackward>(out, {&input})) {
      fn->kernel_size = kernel_size;
      fn->stride = stride;
      fn->padding = padding;
      fn->N = N;
      fn->C = C;
      fn->H = H;
      fn->W = W;
    }
  }

  return out;
}


}
