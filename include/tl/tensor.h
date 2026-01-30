#pragma once // tell compiler to only include this file once per translation unit (fully preprocessed cpp file, includes are processed)

#include <vector>
#include <cstdint> // fixed width integer types library

namespace tl { // defines that Tensor is in the tl library

// Tensor implementation state and methods
class Tensor {
public:
    // construct a contiguous float tensor
    explicit Tensor(const std::vector<int64_t>& sizes);

    // access raw data
    float* data();
    const float* data() const;

    // shape info
    const std::vector<int64_t>& sizes() const;
    const std::vector<int64_t>& strides() const;
    int64_t numel() const;

private:
    std::vector<int64_t> sizes_;
    std::vector<int64_t> strides_;
    std::vector<float> data_;
};

}
