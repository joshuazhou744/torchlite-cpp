#include <iostream>
#include <tl/tensor.h>
#include <tl/ops.h>
#include <tl/nn.h>
#include <tl/factory.h>
#include <cassert>
#include "test_utils.h"

void test_ops() {
  // test element wise addition
  tl::Tensor a({2, 2});
  tl::Tensor b({2, 2});
  a.data()[0] = 1.0f; a.data()[1] = 2.0f; // [[1.0, 2.0], [0.0, 0.0]]
  b.data()[0] = 3.0f; b.data()[1] = 4.0f; // [[3.0, 4.0], [0.0, 0.0]]

  tl::Tensor res_add = tl::add(a, b);
  assert(res_add.data()[0] == 4.0f);
  assert(res_add.data()[1] == 6.0f);

  // test matmul (indexing logic check)
  tl::Tensor A({2, 3}); // [[1, 2, 3], [4, 5, 6]]
  tl::Tensor B({3, 2}); // [[7, 8], [9, 10], [11, 12]]

  for(int i = 0; i < A.numel(); ++i) A.data()[i] = i + 1.0f;
  for(int i = 0; i < B.numel(); ++i) B.data()[i] = i + 7.0f;

  tl::Tensor res_matmul = tl::matmul(A, B);
  // Row 0, Col 0: (1*7) + (2*9) + (3*11) = 58
  assert(res_matmul.data()[0] == 58.0f);
  // Row 1, Col 1: (4*8) + (5*10) + (6*12) = 154
  assert(res_matmul.data()[3] == 154.0f);

  // test transpose
  tl::Tensor transpose_in({2, 3});
  for (int i = 0; i < 6; ++i) transpose_in.data()[i] = (float) i;
  tl::Tensor transpose_out = tl::transpose(transpose_in, 0, 1).contiguous();

  assert(transpose_out.sizes()[0] == 3);
  assert(transpose_out.sizes()[1] == 2);
  assert(transpose_out.data()[1] == 3.0f);


  // test broadcasting on a (2, 3) matrix + (1, 3) vector
  tl::Tensor mat({2, 3});
  tl::Tensor row({1, 3});
  for (int i = 0; i < 6; ++i) mat.data()[i] = 10.0f;
  for (int i = 0; i < 3; ++i) row.data()[i] = (float) i;

  tl::Tensor broadcast_res = tl::add(mat, row);
  assert(broadcast_res.sizes()[0] == 2 && broadcast_res.sizes()[1] == 3);
  assert(broadcast_res.data()[0] == 10.0f); // 10 + 0
  assert(broadcast_res.data()[2] == 12.0f); // 10 + 2
  assert(broadcast_res.data()[5] == 12.0f); // 10 + 2

  // test batched matmul: (2, 3, 2, 2) @ (2, 4) -> (2, 3, 2, 4)
  // simulate a real multi-head attention batched matmul
  tl::Tensor input({2, 3, 2, 2}); // 4D input matrix
  for (int i = 0; i < input.numel(); ++i) input.data()[i] = 1.0f;

  tl::Tensor weights({2, 4}); // 2D weight matrix
  for (int i = 0; i < weights.numel(); ++i) weights.data()[i] = 0.5f;

  // execute matmul and assert tests
  tl::Tensor output = tl::matmul(input, weights);

  assert(output.sizes().size() == 4);
  assert(output.sizes()[0] == 2); // batch dim
  assert(output.sizes()[1] == 3); // heads
  assert(output.sizes()[2] == 2); // seq
  assert(output.sizes()[3] == 4); // out dimension

  // value check
  for (int i = 0; i < output.numel(); ++i) {
    assert(is_close(output.data()[i], 1.0f));
  }

  // heavy matmul: (256, 512) @ (512, 256) — exercises gemm_blocked tile loops
  // fill a with all 1s, b with all 1s -> every output element = sum over K = 512
  tl::Tensor heavy_a({256, 512});
  tl::Tensor heavy_b({512, 256});
  for (int64_t i = 0; i < heavy_a.numel(); ++i) heavy_a.data()[i] = 1.0f;
  for (int64_t i = 0; i < heavy_b.numel(); ++i) heavy_b.data()[i] = 1.0f;
  tl::Tensor heavy_out = tl::matmul(heavy_a, heavy_b);
  assert(heavy_out.sizes()[0] == 256);
  assert(heavy_out.sizes()[1] == 256);
  for (int64_t i = 0; i < heavy_out.numel(); ++i) {
    assert(is_close(heavy_out.data()[i], 512.0f, 1e-2f));
  }

  // large ragged matmul: (259, 300) @ (300, 277) -> (259, 277)
  // dims >= 256 route through the packed kernel; non-multiples of 8 hit the
  // scalar remainders and K=300 (not a multiple of KC) hits the K-slab tail.
  // integer-valued inputs keep the fp32 sums exact, so we can check every
  // element against a naive triple-loop reference (catches index/pack bugs).
  {
    const int64_t LM = 259, LK = 300, LN = 277;
    tl::Tensor la({LM, LK});
    tl::Tensor lb({LK, LN});
    for (int64_t i = 0; i < la.numel(); ++i) la.data()[i] = (float)((i * 3 + 1) % 11 - 5);
    for (int64_t i = 0; i < lb.numel(); ++i) lb.data()[i] = (float)((i * 7 + 2) % 13 - 6);

    tl::Tensor lo = tl::matmul(la, lb);
    assert(lo.sizes()[0] == LM && lo.sizes()[1] == LN);

    for (int64_t i = 0; i < LM; ++i) {
      for (int64_t j = 0; j < LN; ++j) {
        float ref = 0.0f;
        for (int64_t k = 0; k < LK; ++k)
          ref += la.data()[i * LK + k] * lb.data()[k * LN + j];
        assert(is_close(lo.data()[i * LN + j], ref, 1e-3f));
      }
    }
  }

  // test softmax normalization and order
  tl::Tensor softmax_tensor({3});
  softmax_tensor.data()[0] = 1.0f;
  softmax_tensor.data()[1] = 2.0f;
  softmax_tensor.data()[2] = 3.0f;

  tl::Tensor softmax_out = tl::softmax(softmax_tensor);

  float sum = softmax_out.data()[0] + softmax_out.data()[1] + softmax_out.data()[2];
  assert(is_close(sum, 1.0f));
  assert(softmax_out.data()[2] > softmax_out.data()[1] && softmax_out.data()[1] > softmax_out.data()[0]);

  // test scale
  tl::Tensor scale_tensor({2});
  scale_tensor.data()[0] = 2.0f;
  scale_tensor.data()[1] = 4.0f;

  tl::Tensor scale_out = tl::scale(scale_tensor, 0.5f);
  assert(is_close(scale_out.data()[0], 1.0f));
  assert(is_close(scale_out.data()[1], 2.0f));


  // test sum along dim 0: [[1,2,3],[4,5,6]] -> [5,7,9]
  tl::Tensor sum_in({2, 3});
  sum_in.data()[0] = 1.0f; sum_in.data()[1] = 2.0f; sum_in.data()[2] = 3.0f;
  sum_in.data()[3] = 4.0f; sum_in.data()[4] = 5.0f; sum_in.data()[5] = 6.0f;

  tl::Tensor sum_d0 = tl::sum(sum_in, 0);
  assert(sum_d0.sizes().size() == 1);
  assert(sum_d0.sizes()[0] == 3);
  assert(is_close(sum_d0.data()[0], 5.0f));
  assert(is_close(sum_d0.data()[1], 7.0f));
  assert(is_close(sum_d0.data()[2], 9.0f));

  // test sum along dim 1: [[1,2,3],[4,5,6]] -> [6,15]
  tl::Tensor sum_d1 = tl::sum(sum_in, 1);
  assert(sum_d1.sizes().size() == 1);
  assert(sum_d1.sizes()[0] == 2);
  assert(is_close(sum_d1.data()[0], 6.0f));
  assert(is_close(sum_d1.data()[1], 15.0f));

  // test sum keepdim: [[1,2,3],[4,5,6]] -> [[5,7,9]]
  tl::Tensor sum_kd = tl::sum(sum_in, 0, true);
  assert(sum_kd.sizes().size() == 2);
  assert(sum_kd.sizes()[0] == 1);
  assert(sum_kd.sizes()[1] == 3);
  assert(is_close(sum_kd.data()[0], 5.0f));

  // test mean along dim 1: [[1,2,3],[4,5,6]] -> [2,5]
  tl::Tensor mean_d1 = tl::mean(sum_in, 1);
  assert(mean_d1.sizes().size() == 1);
  assert(mean_d1.sizes()[0] == 2);
  assert(is_close(mean_d1.data()[0], 2.0f));
  assert(is_close(mean_d1.data()[1], 5.0f));

  // test sum on 3D: (2,3,2) along dim 1
  tl::Tensor sum3d({2, 3, 2});
  for (int i = 0; i < 12; ++i) sum3d.data()[i] = (float)i;
  // [[[0,1],[2,3],[4,5]], [[6,7],[8,9],[10,11]]]
  // sum along dim 1 -> [[6,9],[24,27]]
  tl::Tensor sum3d_out = tl::sum(sum3d, 1);
  assert(sum3d_out.sizes().size() == 2);
  assert(sum3d_out.sizes()[0] == 2);
  assert(sum3d_out.sizes()[1] == 2);
  assert(is_close(sum3d_out.data()[0], 6.0f));
  assert(is_close(sum3d_out.data()[1], 9.0f));
  assert(is_close(sum3d_out.data()[2], 24.0f));
  assert(is_close(sum3d_out.data()[3], 27.0f));

  // test sub with broadcasting: [[10,10,10],[10,10,10]] - [1,2,3]
  tl::Tensor sub_mat({2, 3});
  tl::Tensor sub_row({1, 3});
  for (int i = 0; i < 6; ++i) sub_mat.data()[i] = 10.0f;
  sub_row.data()[0] = 1.0f; sub_row.data()[1] = 2.0f; sub_row.data()[2] = 3.0f;

  tl::Tensor sub_out = tl::sub(sub_mat, sub_row);
  assert(is_close(sub_out.data()[0], 9.0f));
  assert(is_close(sub_out.data()[1], 8.0f));
  assert(is_close(sub_out.data()[2], 7.0f));
  assert(is_close(sub_out.data()[3], 9.0f));

  // test sqrt
  tl::Tensor sqrt_in({4});
  sqrt_in.data()[0] = 0.0f;
  sqrt_in.data()[1] = 1.0f;
  sqrt_in.data()[2] = 4.0f;
  sqrt_in.data()[3] = 9.0f;

  tl::Tensor sqrt_out = tl::sqrt(sqrt_in);
  assert(is_close(sqrt_out.data()[0], 0.0f));
  assert(is_close(sqrt_out.data()[1], 1.0f));
  assert(is_close(sqrt_out.data()[2], 2.0f));
  assert(is_close(sqrt_out.data()[3], 3.0f));

  // test div with broadcasting: [[10,20,30],[40,50,60]] / [2,5,10]
  tl::Tensor div_mat({2, 3});
  tl::Tensor div_row({1, 3});
  div_mat.data()[0] = 10.0f; div_mat.data()[1] = 20.0f; div_mat.data()[2] = 30.0f;
  div_mat.data()[3] = 40.0f; div_mat.data()[4] = 50.0f; div_mat.data()[5] = 60.0f;
  div_row.data()[0] = 2.0f; div_row.data()[1] = 5.0f; div_row.data()[2] = 10.0f;

  tl::Tensor div_out = tl::div(div_mat, div_row);
  assert(is_close(div_out.data()[0], 5.0f));
  assert(is_close(div_out.data()[1], 4.0f));
  assert(is_close(div_out.data()[2], 3.0f));
  assert(is_close(div_out.data()[3], 20.0f));

  // test neg
  tl::Tensor neg_in({3});
  neg_in.data()[0] = 1.0f; neg_in.data()[1] = -2.0f; neg_in.data()[2] = 0.0f;

  tl::Tensor neg_out = tl::neg(neg_in);
  assert(neg_out.data()[0] == -1.0f);
  assert(neg_out.data()[1] == 2.0f);
  assert(neg_out.data()[2] == 0.0f);

  // test exp: e^0 = 1, e^1 ~ 2.71828
  tl::Tensor exp_in({2});
  exp_in.data()[0] = 0.0f; exp_in.data()[1] = 1.0f;

  tl::Tensor exp_out = tl::exp(exp_in);
  assert(is_close(exp_out.data()[0], 1.0f));
  assert(is_close(exp_out.data()[1], 2.71828f, 1e-4));

  // test log: ln(1) = 0, ln(e) = 1
  tl::Tensor log_in({2});
  log_in.data()[0] = 1.0f; log_in.data()[1] = std::exp(1.0f);

  tl::Tensor log_out = tl::log(log_in);
  assert(is_close(log_out.data()[0], 0.0f));
  assert(is_close(log_out.data()[1], 1.0f));

  // test pow: 2^3 = 8, 9^0.5 = 3
  tl::Tensor pow_in({2});
  pow_in.data()[0] = 2.0f; pow_in.data()[1] = 9.0f;

  tl::Tensor pow_out = tl::pow(pow_in, 3.0f);
  assert(is_close(pow_out.data()[0], 8.0f));

  tl::Tensor pow_sqrt = tl::pow(pow_in, 0.5f);
  assert(is_close(pow_sqrt.data()[1], 3.0f));

  // test clamp: clamp [-5, 0, 5, 10] to [0, 7]
  tl::Tensor clamp_in({4});
  clamp_in.data()[0] = -5.0f; clamp_in.data()[1] = 0.0f;
  clamp_in.data()[2] = 5.0f; clamp_in.data()[3] = 10.0f;

  tl::Tensor clamp_out = tl::clamp(clamp_in, 0.0f, 7.0f);
  assert(clamp_out.data()[0] == 0.0f);
  assert(clamp_out.data()[1] == 0.0f);
  assert(clamp_out.data()[2] == 5.0f);
  assert(clamp_out.data()[3] == 7.0f);

  // test variance: [[1,2,3],[4,5,6]] along dim 1
  // row 0: mean=2, var=((1-2)^2 + (2-2)^2 + (3-2)^2) / 3 = 2/3
  // row 1: mean=5, var=((4-5)^2 + (5-5)^2 + (6-5)^2) / 3 = 2/3
  tl::Tensor var_in({2, 3});
  var_in.data()[0] = 1.0f; var_in.data()[1] = 2.0f; var_in.data()[2] = 3.0f;
  var_in.data()[3] = 4.0f; var_in.data()[4] = 5.0f; var_in.data()[5] = 6.0f;

  tl::Tensor var_out = tl::variance(var_in, 1);
  assert(var_out.sizes().size() == 1);
  assert(var_out.sizes()[0] == 2);
  assert(is_close(var_out.data()[0], 2.0f / 3.0f));
  assert(is_close(var_out.data()[1], 2.0f / 3.0f));

  // test cat: join [[1,2],[3,4]] and [[5,6],[7,8]] along dim 0
  tl::Tensor cat_a({2, 2});
  tl::Tensor cat_b({2, 2});
  cat_a.data()[0] = 1.0f; cat_a.data()[1] = 2.0f;
  cat_a.data()[2] = 3.0f; cat_a.data()[3] = 4.0f;
  cat_b.data()[0] = 5.0f; cat_b.data()[1] = 6.0f;
  cat_b.data()[2] = 7.0f; cat_b.data()[3] = 8.0f;

  tl::Tensor cat_out = tl::cat({cat_a, cat_b}, 0);
  assert(cat_out.sizes()[0] == 4);
  assert(cat_out.sizes()[1] == 2);
  assert(cat_out.data()[0] == 1.0f);
  assert(cat_out.data()[3] == 4.0f);
  assert(cat_out.data()[4] == 5.0f);
  assert(cat_out.data()[7] == 8.0f);

  // test cat along dim 1
  tl::Tensor cat_d1 = tl::cat({cat_a, cat_b}, 1);
  assert(cat_d1.sizes()[0] == 2);
  assert(cat_d1.sizes()[1] == 4);
  assert(cat_d1.data()[0] == 1.0f);
  assert(cat_d1.data()[1] == 2.0f);
  assert(cat_d1.data()[2] == 5.0f);
  assert(cat_d1.data()[3] == 6.0f);

  // test stack: two [3] tensors at dim 0 -> [2, 3]
  tl::Tensor stack_a({3});
  tl::Tensor stack_b({3});
  stack_a.data()[0] = 1.0f; stack_a.data()[1] = 2.0f; stack_a.data()[2] = 3.0f;
  stack_b.data()[0] = 4.0f; stack_b.data()[1] = 5.0f; stack_b.data()[2] = 6.0f;

  tl::Tensor stack_out = tl::stack({stack_a, stack_b}, 0);
  assert(stack_out.sizes().size() == 2);
  assert(stack_out.sizes()[0] == 2);
  assert(stack_out.sizes()[1] == 3);
  assert(stack_out.data()[0] == 1.0f);
  assert(stack_out.data()[3] == 4.0f);

  // test slice: [[1,2,3,4],[5,6,7,8]] slice dim 1 from 1 to 3
  tl::Tensor slice_in({2, 4});
  for (int i = 0; i < 8; ++i) slice_in.data()[i] = (float)(i + 1);

  tl::Tensor slice_out = tl::slice(slice_in, 1, 1, 3);
  assert(slice_out.sizes()[0] == 2);
  assert(slice_out.sizes()[1] == 2);
  assert(slice_out.data()[0] == 2.0f);
  assert(slice_out.data()[1] == 3.0f);
  assert(slice_out.data()[2] == 6.0f);
  assert(slice_out.data()[3] == 7.0f);

  // test conv2d: (1,1,4,4) input, (1,1,3,3) all-ones kernel, stride=1, padding=0
  // output shape should be (1,1,2,2)
  // each output value = sum of 3x3 patch
  // input:
  //  1  2  3  4
  //  5  6  7  8
  //  9 10 11 12
  // 13 14 15 16
  // patch at (0,0): 1+2+3+5+6+7+9+10+11 = 54
  // patch at (0,1): 2+3+4+6+7+8+10+11+12 = 63
  // patch at (1,0): 5+6+7+9+10+11+13+14+15 = 90
  // patch at (1,1): 6+7+8+10+11+12+14+15+16 = 99
  tl::Tensor conv_in({1, 1, 4, 4});
  for (int i = 0; i < 16; ++i) conv_in.data()[i] = (float)(i + 1);

  tl::Tensor conv_w({1, 1, 3, 3});
  for (int i = 0; i < 9; ++i) conv_w.data()[i] = 1.0f;

  tl::Tensor conv_bias; // empty — no bias
  tl::Tensor conv_out = tl::conv2d(conv_in, conv_w, conv_bias, 1, 0);

  assert(conv_out.sizes()[0] == 1);
  assert(conv_out.sizes()[1] == 1);
  assert(conv_out.sizes()[2] == 2);
  assert(conv_out.sizes()[3] == 2);
  assert(is_close(conv_out.data()[0], 54.0f));
  assert(is_close(conv_out.data()[1], 63.0f));
  assert(is_close(conv_out.data()[2], 90.0f));
  assert(is_close(conv_out.data()[3], 99.0f));

  // test conv2d with groups=2 (depthwise): each output channel must only see its own
  // input channel — if any cross-channel mixing leaks through, the asserts below will fail.
  //
  // input (1, 2, 3, 3):  channel 0 all 1.0,  channel 1 all 10.0
  // weight (2, 1, 3, 3): filter 0 all 1.0,   filter 1 all 2.0   (C_in/groups = 1)
  // padding=0, stride=1 -> output (1, 2, 1, 1):
  //   out[0,0,0,0] = sum(ch0 patch) * filter0 = 9*1 * 1 = 9
  //   out[0,1,0,0] = sum(ch1 patch) * filter1 = 9*10 * 2 = 180
  // if channels were mixed, ch0 output would also pick up ch1's huge values.
  tl::Tensor dwconv_in({1, 2, 3, 3});
  for (int i = 0; i < 9;  ++i) dwconv_in.data()[i] = 1.0f;   // channel 0
  for (int i = 9; i < 18; ++i) dwconv_in.data()[i] = 10.0f;  // channel 1

  tl::Tensor dwconv_w({2, 1, 3, 3});
  for (int i = 0; i < 9;  ++i) dwconv_w.data()[i] = 1.0f;    // filter for ch 0
  for (int i = 9; i < 18; ++i) dwconv_w.data()[i] = 2.0f;    // filter for ch 1

  tl::Tensor dwconv_bias; // empty
  tl::Tensor dwconv_out = tl::conv2d(dwconv_in, dwconv_w, dwconv_bias,
                                     /*stride=*/1, /*padding=*/0, /*groups=*/2);

  assert(dwconv_out.sizes().size() == 4);
  assert(dwconv_out.sizes()[0] == 1);
  assert(dwconv_out.sizes()[1] == 2);
  assert(dwconv_out.sizes()[2] == 1);
  assert(dwconv_out.sizes()[3] == 1);
  assert(is_close(dwconv_out.data()[0], 9.0f));    // ch 0 isolated -> small value
  assert(is_close(dwconv_out.data()[1], 180.0f));  // ch 1 isolated, filter1 applied -> 180

  // also exercise the nn::Conv2d module path with groups, verify per-group weight shape
  // (C_out, C_in/groups, k, k) = (4, 1, 3, 3) for a depthwise layer with C=4
  tl::nn::Conv2d dwmod(/*in=*/4, /*out=*/4, /*k=*/3,
                       /*stride=*/1, /*padding=*/1, /*groups=*/4, /*bias=*/false);
  assert(dwmod.weight().sizes().size() == 4);
  assert(dwmod.weight().sizes()[0] == 4);  // C_out
  assert(dwmod.weight().sizes()[1] == 1);  // C_in / groups
  assert(dwmod.weight().sizes()[2] == 3);
  assert(dwmod.weight().sizes()[3] == 3);
  tl::Tensor dwmod_in = tl::ones({2, 4, 5, 5});
  tl::Tensor dwmod_out = dwmod.forward(dwmod_in);
  assert(dwmod_out.sizes()[0] == 2);
  assert(dwmod_out.sizes()[1] == 4);  // depthwise preserves channel count
  assert(dwmod_out.sizes()[2] == 5);  // padding=1 preserves spatial dims
  assert(dwmod_out.sizes()[3] == 5);

  // test max_pool2d: (1,1,4,4) input, kernel=2, stride=2 -> (1,1,2,2)
  // input (filled 1..16):
  //  1  2 | 3  4
  //  5  6 | 7  8
  // ----- | -----
  //  9 10 | 11 12
  // 13 14 | 15 16
  // expected: max of each 2x2 tile -> 6, 8, 14, 16
  tl::Tensor pool_in({1, 1, 4, 4});
  for (int i = 0; i < 16; ++i) pool_in.data()[i] = (float)(i + 1);

  tl::Tensor mp_out = tl::max_pool2d(pool_in, 2, 2, 0);
  assert(mp_out.sizes()[0] == 1);
  assert(mp_out.sizes()[1] == 1);
  assert(mp_out.sizes()[2] == 2);
  assert(mp_out.sizes()[3] == 2);
  assert(is_close(mp_out.data()[0], 6.0f));
  assert(is_close(mp_out.data()[1], 8.0f));
  assert(is_close(mp_out.data()[2], 14.0f));
  assert(is_close(mp_out.data()[3], 16.0f));

  // test avg_pool2d: same input, kernel=2, stride=2 -> (1,1,2,2)
  // expected: avg of each 2x2 tile -> 3.5, 5.5, 11.5, 13.5
  tl::Tensor ap_out = tl::avg_pool2d(pool_in, 2, 2, 0);
  assert(ap_out.sizes()[0] == 1);
  assert(ap_out.sizes()[1] == 1);
  assert(ap_out.sizes()[2] == 2);
  assert(ap_out.sizes()[3] == 2);
  assert(is_close(ap_out.data()[0], 3.5f));
  assert(is_close(ap_out.data()[1], 5.5f));
  assert(is_close(ap_out.data()[2], 11.5f));
  assert(is_close(ap_out.data()[3], 13.5f));

  // test max_pool2d default stride (stride=0 -> uses kernel_size)
  tl::Tensor mp_default = tl::max_pool2d(pool_in, 2);
  assert(mp_default.sizes()[2] == 2 && mp_default.sizes()[3] == 2);
  assert(is_close(mp_default.data()[0], 6.0f));

  // ---- flash attention forward: must match the materialized reference ----
  // reference: O = softmax(Q @ K^T * scale) @ V, computed with the standard ops.
  // S=80 deliberately crosses the 64-row tile boundary to exercise the tiling.
  {
    const int64_t S = 80, d = 16;
    tl::Tensor Q = tl::randn({S, d});
    tl::Tensor K = tl::randn({S, d});
    tl::Tensor V = tl::randn({S, d});
    float sm_scale = 1.0f / std::sqrt((float)d);

    tl::Tensor scores = tl::scale(tl::matmul(Q, tl::transpose(K, 0, 1)), sm_scale);
    tl::Tensor ref = tl::matmul(tl::softmax(scores), V);   // [S, d]

    tl::Tensor flash = tl::flash_attention(Q, K, V, sm_scale);

    assert(flash.sizes()[0] == S && flash.sizes()[1] == d);
    for (int64_t i = 0; i < S * d; ++i)
      assert(is_close(flash.data()[i], ref.data()[i], 1e-4f));
  }

  // ---- flash attention backward: grads must match autograd through the reference ----
  // ground truth = backprop through the materialized softmax-attention path.
  // backward() seeds dO = ones, so both paths get the same upstream gradient.
  {
    const int64_t S = 80, d = 16;
    tl::Tensor Q = tl::randn({S, d});
    tl::Tensor K = tl::randn({S, d});
    tl::Tensor V = tl::randn({S, d});
    Q.set_requires_grad(true);
    K.set_requires_grad(true);
    V.set_requires_grad(true);
    float sm_scale = 1.0f / std::sqrt((float)d);

    // reference backward (ground truth)
    tl::Tensor scores = tl::scale(tl::matmul(Q, tl::transpose(K, 0, 1)), sm_scale);
    tl::Tensor ref = tl::matmul(tl::softmax(scores), V);
    ref.backward();

    std::vector<float> dQ_ref(Q.grad().data(), Q.grad().data() + S * d);
    std::vector<float> dK_ref(K.grad().data(), K.grad().data() + S * d);
    std::vector<float> dV_ref(V.grad().data(), V.grad().data() + S * d);

    // clear grads before the flash pass (accumulate_grad would otherwise add on top)
    Q.grad() = tl::Tensor();
    K.grad() = tl::Tensor();
    V.grad() = tl::Tensor();

    // flash backward
    tl::Tensor flash = tl::flash_attention(Q, K, V, sm_scale);
    flash.backward();

    for (int64_t i = 0; i < S * d; ++i) {
      assert(is_close(Q.grad().data()[i], dQ_ref[i], 1e-4f));
      assert(is_close(K.grad().data()[i], dK_ref[i], 1e-4f));
      assert(is_close(V.grad().data()[i], dV_ref[i], 1e-4f));
    }
  }

  // test cos: cos(0)=1, cos(pi)=-1, cos(pi/2)~0
  {
    tl::Tensor t({3});
    t.data()[0] = 0.0f;
    t.data()[1] = (float)M_PI;
    t.data()[2] = (float)M_PI / 2.0f;
    tl::Tensor c = tl::cos(t);
    assert(is_close(c.data()[0],  1.0f, 1e-6f));
    assert(is_close(c.data()[1], -1.0f, 1e-6f));
    assert(is_close(c.data()[2],  0.0f, 1e-6f));
  }

  // test sin: sin(0)=0, sin(pi)~0, sin(pi/2)=1
  {
    tl::Tensor t({3});
    t.data()[0] = 0.0f;
    t.data()[1] = (float)M_PI;
    t.data()[2] = (float)M_PI / 2.0f;
    tl::Tensor s = tl::sin(t);
    assert(is_close(s.data()[0], 0.0f, 1e-6f));
    assert(is_close(s.data()[1], 0.0f, 1e-5f));
    assert(is_close(s.data()[2], 1.0f, 1e-6f));
  }

  // test rope_cos_sin: exact table values, interleaved pairs
  // positions [0,1,2], dim=4, theta=100 -> inv_freq = [1.0, 0.1]
  {
    tl::Tensor pos({3});
    pos.data()[0] = 0.0f; pos.data()[1] = 1.0f; pos.data()[2] = 2.0f;
    auto [c, s] = tl::rope_cos_sin(pos, 4, 100.0f);

    assert(c.sizes()[0] == 3 && c.sizes()[1] == 4);
    assert(s.sizes()[0] == 3 && s.sizes()[1] == 4);

    // t=0: no rotation -> cos=1, sin=0 everywhere
    for (int i = 0; i < 4; ++i) {
      assert(is_close(c.data()[i], 1.0f, 1e-5f));
      assert(is_close(s.data()[i], 0.0f, 1e-5f));
    }
    // t=1: pair 0 angle 1.0, pair 1 angle 0.1, each value doubled (interleaved)
    assert(is_close(c.data()[4], 0.5403f, 1e-4f));
    assert(is_close(c.data()[5], 0.5403f, 1e-4f));
    assert(is_close(c.data()[6], 0.9950f, 1e-4f));
    assert(is_close(c.data()[7], 0.9950f, 1e-4f));
    assert(is_close(s.data()[4], 0.8415f, 1e-4f));
    assert(is_close(s.data()[5], 0.8415f, 1e-4f));
    assert(is_close(s.data()[6], 0.0998f, 1e-4f));
    assert(is_close(s.data()[7], 0.0998f, 1e-4f));
    // t=2: angles 2.0 and 0.2
    assert(is_close(c.data()[8], -0.4161f, 1e-4f));
    assert(is_close(s.data()[8], 0.9093f, 1e-4f));
    assert(is_close(c.data()[10], 0.9801f, 1e-4f));
    assert(is_close(s.data()[10], 0.1987f, 1e-4f));
  }

  // test apply_rotary: position 0 is identity, 90 degrees maps (a,b) -> (-b,a)
  {
    tl::Tensor pos({2});
    pos.data()[0] = 0.0f;
    pos.data()[1] = (float)M_PI / 2.0f; // dim=2 -> single pair, inv_freq=1 -> angle = position
    auto [c, s] = tl::rope_cos_sin(pos, 2, 100.0f);

    tl::Tensor x({2, 2});
    x.data()[0] = 1.0f; x.data()[1] = 2.0f; // t=0
    x.data()[2] = 3.0f; x.data()[3] = 4.0f; // t=1

    tl::Tensor out = tl::apply_rotary(x, c, s);
    // t=0: angle 0 -> unchanged
    assert(is_close(out.data()[0], 1.0f, 1e-5f));
    assert(is_close(out.data()[1], 2.0f, 1e-5f));
    // t=1: quarter turn -> (3,4) -> (-4,3)
    assert(is_close(out.data()[2], -4.0f, 1e-5f));
    assert(is_close(out.data()[3], 3.0f, 1e-5f));
  }

  // test apply_rotary: rotation preserves each pair's norm (cos^2 + sin^2 = 1)
  // and broadcasts over leading dims: x [N=2, T=3, dim=4] with tables [3, 4]
  {
    tl::Tensor pos({3});
    pos.data()[0] = 0.0f; pos.data()[1] = 1.0f; pos.data()[2] = 2.0f;
    auto [c, s] = tl::rope_cos_sin(pos, 4, 100.0f);

    tl::Tensor x = tl::randn({2, 3, 4});
    tl::Tensor out = tl::apply_rotary(x, c, s);
    assert(out.sizes()[0] == 2 && out.sizes()[1] == 3 && out.sizes()[2] == 4);

    for (int64_t i = 0; i < 2 * 3 * 4; i += 2) { // one iteration per pair
      float in_norm = x.data()[i] * x.data()[i] + x.data()[i + 1] * x.data()[i + 1];
      float out_norm = out.data()[i] * out.data()[i] + out.data()[i + 1] * out.data()[i + 1];
      assert(is_close(in_norm, out_norm, 1e-4f));
    }
  }

  // test rope_cos_sin_2d: h=2, w=3, dim=4 -> one pair per axis, angle = grid index
  // token r*w+c: first half rotates with row r, second half with col c
  {
    auto [c, s] = tl::rope_cos_sin_2d(2, 3, 4, 100.0f);
    assert(c.sizes()[0] == 6 && c.sizes()[1] == 4);

    // token 0 = (r0, c0): no rotation anywhere
    for (int i = 0; i < 4; ++i) {
      assert(is_close(c.data()[i], 1.0f, 1e-5f));
      assert(is_close(s.data()[i], 0.0f, 1e-5f));
    }
    // token 5 = (r1, c2): row half angle 1, col half angle 2
    assert(is_close(c.data()[5 * 4 + 0], 0.5403f, 1e-4f));
    assert(is_close(c.data()[5 * 4 + 2], -0.4161f, 1e-4f));
    assert(is_close(s.data()[5 * 4 + 0], 0.8415f, 1e-4f));
    assert(is_close(s.data()[5 * 4 + 2], 0.9093f, 1e-4f));
    // tokens 1 = (r0, c1) and 3 = (r1, c0): same values, opposite halves
    assert(is_close(c.data()[1 * 4 + 2], c.data()[3 * 4 + 0], 1e-5f));
    assert(is_close(c.data()[1 * 4 + 0], c.data()[3 * 4 + 2], 1e-5f));
  }

  // test RoPE validation: wrong table length, odd dim, dim not divisible by 4
  {
    tl::Tensor pos({2});
    pos.data()[0] = 0.0f; pos.data()[1] = 1.0f;
    auto [c, s] = tl::rope_cos_sin(pos, 4, 100.0f);

    bool threw = false;
    try {
      tl::Tensor x = tl::randn({2, 3, 4}); // T=3 but tables are [2, 4]
      tl::apply_rotary(x, c, s);
    } catch (const std::invalid_argument&) { threw = true; }
    assert(threw);

    threw = false;
    try { tl::rope_cos_sin(pos, 3, 100.0f); } // odd dim
    catch (const std::invalid_argument&) { threw = true; }
    assert(threw);

    threw = false;
    try { tl::rope_cos_sin_2d(2, 3, 6, 100.0f); } // dim % 4 != 0
    catch (const std::invalid_argument&) { threw = true; }
    assert(threw);
  }

  // test repeat_kv: consecutive copies so Q head i maps to KV head i / n_rep
  {
    tl::Tensor x({1, 2, 1, 2}); // [N, kv_heads=2, T=1, head_dim=2]
    x.data()[0] = 10.0f; x.data()[1] = 11.0f; // kv0
    x.data()[2] = 20.0f; x.data()[3] = 21.0f; // kv1

    tl::Tensor out = tl::repeat_kv(x, 2);
    assert(out.sizes()[0] == 1 && out.sizes()[1] == 4);
    assert(out.sizes()[2] == 1 && out.sizes()[3] == 2);

    // heads 0,1 = copies of kv0; heads 2,3 = copies of kv1 (NOT interleaved)
    float expected[8] = {10, 11, 10, 11, 20, 21, 20, 21};
    for (int i = 0; i < 8; ++i)
      assert(is_close(out.data()[i], expected[i]));

    // n_rep = 1: MHA passthrough, values unchanged
    tl::Tensor same = tl::repeat_kv(x, 1);
    for (int i = 0; i < 4; ++i)
      assert(is_close(same.data()[i], x.data()[i]));

    // 3D input throws
    bool threw = false;
    try { tl::repeat_kv(tl::randn({2, 2, 2}), 2); }
    catch (const std::invalid_argument&) { threw = true; }
    assert(threw);
  }

  // test dino_rope_cos_sin_2d: h=1, w=3, dim=8 -> periods {1, 10},
  // row coord 0 (identity), col coords -2/3, 0, +2/3, layout [row|col] tiled twice
  {
    auto [c, s] = tl::dino_rope_cos_sin_2d(1, 3, 8, 100.0f);
    assert(c.sizes()[0] == 3 && c.sizes()[1] == 8);
    assert(s.sizes()[0] == 3 && s.sizes()[1] == 8);

    // token 0: golden values from python reference of the DINO builder
    float c0[8] = {1.0f, 1.0f, -0.5f, 0.9135455f, 1.0f, 1.0f, -0.5f, 0.9135455f};
    float s0[8] = {0.0f, 0.0f, 0.8660254f, -0.4067366f, 0.0f, 0.0f, 0.8660254f, -0.4067366f};
    for (int i = 0; i < 8; ++i) {
      assert(is_close(c.data()[i], c0[i], 1e-5));
      assert(is_close(s.data()[i], s0[i], 1e-5));
    }

    // token 1 sits at the center of both axes -> zero angles -> identity rotation
    for (int i = 0; i < 8; ++i) {
      assert(is_close(c.data()[8 + i], 1.0f, 1e-5));
      assert(is_close(s.data()[8 + i], 0.0f, 1e-5));
    }

    // token 2 mirrors token 0 about the center: cos equal, sin negated
    for (int i = 0; i < 8; ++i) {
      assert(is_close(c.data()[16 + i], c0[i], 1e-5));
      assert(is_close(s.data()[16 + i], -s0[i], 1e-5));
    }

    // tile(2) layout: channels i and i + dim/2 share an angle
    for (int t = 0; t < 3; ++t) {
      for (int i = 0; i < 4; ++i) {
        assert(is_close(c.data()[t * 8 + i], c.data()[t * 8 + 4 + i]));
        assert(is_close(s.data()[t * 8 + i], s.data()[t * 8 + 4 + i]));
      }
    }

    // dim % 4 != 0 throws
    bool threw = false;
    try { tl::dino_rope_cos_sin_2d(2, 2, 6, 100.0f); }
    catch (const std::invalid_argument&) { threw = true; }
    assert(threw);
  }

  // test apply_rotary_half: pairing is (i, i + dim/2), NOT interleaved
  {
    // needle 0 = channels (0,2) at angle pi/2, needle 1 = channels (1,3) at angle 0
    tl::Tensor x({1, 4});
    x.data()[0] = 1.0f; x.data()[1] = 2.0f; x.data()[2] = 3.0f; x.data()[3] = 4.0f;
    tl::Tensor c({1, 4});
    tl::Tensor s({1, 4});
    c.data()[0] = 0.0f; c.data()[1] = 1.0f; c.data()[2] = 0.0f; c.data()[3] = 1.0f;
    s.data()[0] = 1.0f; s.data()[1] = 0.0f; s.data()[2] = 1.0f; s.data()[3] = 0.0f;

    tl::Tensor out = tl::apply_rotary_half(x, c, s);
    // rot = [-x2|x1] = [-3,-4,1,2]; out = x*cos + rot*sin = [-3, 2, 1, 4]
    // (interleaved rotate_half would give rot [-2,1,-4,3] -> out [-2, 2, -4, 4])
    assert(is_close(out.data()[0], -3.0f));
    assert(is_close(out.data()[1], 2.0f));
    assert(is_close(out.data()[2], 1.0f));
    assert(is_close(out.data()[3], 4.0f));
  }

  // apply_rotary_half preserves per-token norm (rotations are orthogonal),
  // tables broadcast over leading dims
  {
    auto [c, s] = tl::dino_rope_cos_sin_2d(1, 3, 8, 100.0f); // [3, 8]
    tl::Tensor x = tl::randn({2, 3, 8});
    tl::Tensor out = tl::apply_rotary_half(x, c, s);
    for (int64_t t = 0; t < 6; ++t) {
      float n_in = 0.0f, n_out = 0.0f;
      for (int64_t i = 0; i < 8; ++i) {
        n_in += x.data()[t * 8 + i] * x.data()[t * 8 + i];
        n_out += out.data()[t * 8 + i] * out.data()[t * 8 + i];
      }
      assert(is_close(std::sqrt(n_in), std::sqrt(n_out), 1e-4));
    }

    // mismatched table shape throws
    bool threw = false;
    try { tl::apply_rotary_half(x, c, tl::randn({3, 4})); }
    catch (const std::invalid_argument&) { threw = true; }
    assert(threw);
  }

  std::cout << "ops tests passed" << std::endl;
}
