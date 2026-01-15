#include <tl/ops.h>
#include <cstdint> // for int64_t
#include <stdexcept>
#include <algorithm> // for max()

namespace tl {

// helper function to validate two tensors have same shape
static void check_same_shape(const Tensor& a, const Tensor& b) {
    if (a.sizes() != b.sizes()) {
        throw std::invalid_argument("Tensor shapes must match");
    }
}

// element-wise addition
Tensor add(const Tensor& a, const Tensor& b) {
    check_same_shape(a, b);

    Tensor out(a.sizes());

    const float* ap = a.data();
    const float* bp = b.data();
    float* op = out.data();

    const int64_t n = a.numel();
    for (int64_t i = 0; i < n; ++i) {
        op[i] = ap[i] + bp[i];
    }

    return out;
}

// element-wise multiply
Tensor mul(const Tensor& a, const Tensor& b) {
    check_same_shape(a, b);

    Tensor out(a.sizes());

    const float* ap = a.data();
    const float* bp = b.data();
    float* op = out.data();

    const int64_t n = a.numel();
    for (int64_t i = 0; i < n; ++i) {
        op[i] = ap[i] * bp[i];
    }

    return out;
}

// unary ReLU
Tensor relu(const Tensor& a) {
    Tensor out(a.sizes());

    const float* ap = a.data();
    float* op = out.data();

    const int64_t n = a.numel();
    for (int64_t i = 0; i < n; ++i) {
        op[i] = std::max(0.0f, ap[i]);
    }

    return out;
}
}
