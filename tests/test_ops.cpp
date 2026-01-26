#include <iostream>
#include <tl/tensor.h>
#include <tl/ops.h>
#include <cassert>
#include <cmath>

// helper function for float comparison
bool is_close(float a, float b, float e = 1e-5) {
  return std::abs( a - b ) < e;
}

int main() {
  // test element wise addition
  tl::Tensor a({2, 2});
  tl::Tensor b({2, 2});
  a.data()[0] = 1.0f; a.data()[1] = 2.0f; // [[1.0, 2.0], [0.0, 0.0]]
  b.data()[0] = 3.0f; b.data()[1] = 4.0f; // [[3.0, 4.0], [0.0, 0.0]]
  
  tl::Tensor res_add = tl::add(a, b);
  assert(res_add.data()[0] == 4.0f);
  assert(res_add.data()[1] == 6.0f);
  
  // test ReLU edge cases (0 and negative)
  tl::Tensor c({1, 3});
  c.data()[0] = -5.0f; c.data()[1] = 0.0f; c.data()[2] = 5.0f;

  tl::Tensor res_relu = tl::relu(c);
  assert(res_relu.data()[0] == 0.0f);
  assert(res_relu.data()[1] == 0.0f);
  assert(res_relu.data()[2] == 5.0f);
  
  // test matmul (indexing logic check)
  tl::Tensor A({2, 3}); // [[1, 2, 3], [4, 5, 6]]
  tl::Tensor B({3, 2}); // [[7, 8], [9, 10], [11, 12]]
  
  for(int i = 0; i < A.numel(); ++i) A.data()[i] = i + 1.0f;
  for(int i = 0; i < B.numel(); ++i) B.data()[i] = i + 7.0f;

  tl::Tensor res_matmul = tl::matmul(A, B);
  // Row 0, Col 0: (1*7) + (2*9) + (3*11) = 58
  assert(res_matmul.data()[0] == 58.0f);
  // Row 1, Col 1: (4*8) + (5*10) + (6*12) = 154
  assert(res_matmul.data()[1] == 154.0f);

  // test sigmoid
  tl::Tensor d({1, 1});
  d.data()[0] = 0.0f;
  tl::Tensor res_sigmoid = tl::sigmoid(d);
  // sigmoid(0) = 0.5
  assert(is_close(res_sigmoid.data()[0], 0.5f));

  std::cout << "tests passed" << std::endl;
  return 0;
}
