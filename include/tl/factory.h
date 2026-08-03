#pragma once

#include <tl/tensor.h>
#include <cstdint>
#include <string>

namespace tl {

// tensor of a given value
Tensor full(const std::vector<int64_t>& sizes, float value);

// tensor of zeros
Tensor zeros(const std::vector<int64_t>& sizes);

// tensor of ones
Tensor ones(const std::vector<int64_t>& sizes);

// tensor of given size from normal distribution
Tensor randn(const std::vector<int64_t>& sizes);

// 1D tensor in the given range of values
Tensor arange(int start, int end);

// steps values evenly spaced from start to end (inclusive)
Tensor linspace(float start, float end, int64_t steps);

// load PyTorch tensors
Tensor load(const std::string& path, const std::vector<int64_t>& sizes);

// random tensor like
inline Tensor randn_like(const Tensor& t) { return randn(t.sizes()); }

// zeros tensor like
inline Tensor zeros_like(const Tensor& t) { return zeros(t.sizes()); }

// ones tensor like
inline Tensor ones_like(const Tensor& t) { return ones(t.sizes()); }

// full tensor like
inline Tensor full_like(const Tensor& t, float value) { return full(t.sizes(), value); }

// triangular mask
Tensor tri_mask(int64_t rows, int64_t cols, int64_t diagonal = 0, bool lower = true);

}
