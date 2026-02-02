#include <iostream>
#include <tl/tensor.h>
#include <cassert>

int main() {
    // construct a 2 x 3 tensor
    tl::Tensor t1({2, 3});

    // check shape
    assert(t1.sizes().size() == 2);
    assert(t1.sizes()[0] == 2);
    assert(t1.sizes()[1] == 3);

    // check numel
    assert(t1.numel() == 6);

    // check data access
    float* data = t1.data();
    data[0] = 42.0f;
    assert(t1.data()[0] == 42.0f);

    // shape size invariant test
    tl::Tensor t2({2, 3, 4});
    assert(t2.numel() == 24);

    // check stides
    tl::Tensor t3({2, 3, 4});
    // expected strides: (12, 4, 1)
    assert(t3.strides()[0] == 12);
    assert(t3.strides()[1] == 4);
    assert(t3.strides()[2] == 1);

    std::cout << "tests passed" << std::endl;

    return 0;
}
