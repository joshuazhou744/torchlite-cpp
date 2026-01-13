#include <iostream>
#include <tl/tensor.h>
#include <cassert>

int main() {
    // construct a 2 x 3 tensor
    tl::Tensor t({2, 3});
   
    // check shape
    assert(t.sizes().size() == 2);
    assert(t.sizes()[0] == 2);
    assert(t.sizes()[1] == 3);

    // check numel
    assert(t.numel() == 6);

    // check data access
    float* data = t.data();
    data[0] = 42.0f;
    assert(t.data()[0] == 42.0f);

    std::cout << "tests passed" << std::endl;

    return 0;
}
