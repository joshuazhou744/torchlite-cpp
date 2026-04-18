#pragma once // tell compiler to only include this file once per translation unit (fully preprocessed cpp file, includes are processed)

#include <vector>
#include <cstdint> // fixed width integer types library
#include <ostream>
#include <memory>

namespace tl { // defines that Tensor is in the tl library

class GradFunction; // forward declaration

// Tensor implementation state and methods
class Tensor {
public:
    // default tensor constructor (empty)
    Tensor();

    // construct a contiguous float tensor
    explicit Tensor(const std::vector<int64_t>& sizes);

    // access raw data
    float* data();
    const float* data() const;
    float& at(int64_t index);
    const float& at(int64_t index) const;
    Tensor contiguous() const;
    Tensor clone() const;

    // shape info
    const std::vector<int64_t>& sizes() const;
    const std::vector<int64_t>& strides() const;
    int64_t numel() const;

    // data info
    bool is_contiguous() const;
    int64_t storage_offset() const;
    bool empty() const;

    // autograd public methods
    void set_requires_grad(bool val);
    Tensor& grad();
    void backward();
    bool requires_grad = false; // should we track gradient of this tensor
    std::shared_ptr<GradFunction> grad_fn; // operation that created this tensor

private:
    Tensor(std::shared_ptr<std::vector<float>> data,
        const std::vector<int64_t>& sizes,
        const std::vector<int64_t>& strides,
        int64_t offset);

    std::shared_ptr<std::vector<float>> data_;

    std::vector<int64_t> sizes_;
    std::vector<int64_t> strides_;
    int64_t numel_;
    int64_t offset_;

    // grant access to ops that need to create views
    friend Tensor transpose(const Tensor& a, int64_t dim0, int64_t dim1);
    friend Tensor reshape(const Tensor& a, const std::vector<int64_t>& new_sizes);


    // autograd
    std::shared_ptr<Tensor> grad_; // accumulated gradients
};

  std::ostream& operator<<(std::ostream& os, const Tensor& t);

}
