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

// load PyTorch tensors
Tensor load(const std::string& path, const std::vector<int64_t>& sizes);

}
