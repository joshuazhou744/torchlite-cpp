// tensor.cpp
// lightweight implementation of a tensor with features
// - cpu-only
// - float32
// - continuous storage
// - simple shape tracking


#include <tl/tensor.h>
#include <numeric> // for std::accumulate
#include <stdexcept> // for std::invalid_argument exception
#include <limits> // for std::numeric_limits
#include <cstdint> // for int64_t

namespace tl {

// helper to compute number of elements from sizes
static int64_t compute_numel(const std::vector<int64_t>& sizes) {
    if (sizes.empty()) return 1; // 0D scalar tensors have numel = 1

    int64_t n = 1; // count number of elements
    for (int64_t d: sizes) {
        if (d < 0) {
            throw std::invalid_argument("Tensor size must be positive integer");
        }

        if (d == 0) return 0; // if a dimension is 0, tensor has no elements

        // overflow guard
        if (n > std::numeric_limits<int64_t>::max() / d) {
            throw std::overflow_error("Tensor numel overflow");
        }
        // multiple elements by the dimensions
        n *= d;
    }
    return n;
}

// constructor to allocate contiguous storage based on sizes
Tensor::Tensor(const std::vector<int64_t>& sizes)
        : sizes_(sizes) // initialize private member to sizes
{
    // use helper to get total numel
    const int64_t n = compute_numel(sizes_);

    // allocate or resize the contiguous buffer
    // vector<int64_t> owns the memory and keeps it contiguous
    data_.resize(static_cast<size_t>(n)); // resize initializes to 0.0f (float)

    // calcuate strides for dimension hopping (row-major)
    strides_.resize(sizes.size());
    int64_t current_stride = 1;
    for (int i = static_cast<int>(sizes.size()) - 1; i >= 0; --i) {
      strides_[i] = current_stride;
      current_stride *= sizes[i];
    }
}

// mutable raw data access
float* Tensor::data() {
    return data_.data();
}

// inmutable raw data access
const float* Tensor::data() const {
    return data_.data();
}

// sizes accessor
const std::vector<int64_t>& Tensor::sizes() const {
    return sizes_;
}

// number of elements (numel) accessor
int64_t Tensor::numel() const {
    return static_cast<int64_t>(data_.size());
}

}
