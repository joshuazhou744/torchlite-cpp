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
#include <ostream>

namespace tl {

// helper to print out the tensor
  std::ostream& operator<<(std::ostream& os, const Tensor& t) {
    os << "Tensor(shape[";

    for (size_t i = 0; i < t.sizes().size(); ++i) {
      os << t.sizes()[i];
      if (i + 1 < t.sizes().size()) os << ", ";
    }
    os << "], data=[";

    for (int64_t i = 0; i < t.numel(); ++i) {
      os << t.data()[i];
      if (i + 1 < t.numel()) os << ", ";
    }
    os << "])";
    return os;
  }

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
        : sizes_(sizes), // initialize private member to sizes
        offset_(0)
{
    // use helper to get total numel
    const int64_t n = compute_numel(sizes_);

    // allocate or resize the contiguous buffer
    // vector<int64_t> owns the memory and keeps it contiguous
    data_ = std::make_shared<std::vector<float>>(static_cast<size_t>(n));

    // calcuate strides for dimension hopping (row-major)
    strides_.resize(sizes.size());
    int64_t current_stride = 1;
    for (int i = static_cast<int>(sizes.size()) - 1; i >= 0; --i) {
      strides_[i] = current_stride;
      current_stride *= sizes[i];
    }
}

// create view of existing data
Tensor::Tensor(std::shared_ptr<std::vector<float>> data,
    const std::vector<int64_t>& sizes,
    const std::vector<int64_t>& strides,
    int64_t offset)
  : data_(data),
    sizes_(sizes),
    strides_(strides),
    offset_(offset)
{

}

// mutable raw data access
float* Tensor::data() {
    return data_->data() + offset_;
}

// inmutable raw data access
const float* Tensor::data() const {
    return data_->data() + offset_;
}

// sizes accessor
const std::vector<int64_t>& Tensor::sizes() const {
    return sizes_;
}

// number of elements (numel) accessor
int64_t Tensor::numel() const {
  return compute_numel(sizes_);
}

// strides accessor for higher dim tensors
const std::vector<int64_t>& Tensor::strides() const{
  return strides_;
}

// offset into storage buffer
int64_t Tensor::storage_offset() const {
  return offset_;
}

// check if tensor is contiguous
bool Tensor::is_contiguous() const {
  if (sizes_.empty()) return true;
  int64_t expected_stride = 1;
  for (int i = static_cast<int>(sizes_.size()) - 1; i >= 0; --i) {
    if (strides_[i] != expected_stride) {
      return false;
    }
    expected_stride *= sizes_[i];
  }

  return true;
}

}
