#include <tl/tensor.h>
#include <tl/ops.h>
#include <tl/nn.h>
#include <tl/activation.h>
#include <tl/loss.h>
#include <tl/optim.h>
#include <tl/factory.h>

#include <iostream>
#include <vector>
#include <random>
#include <cmath>


// generate N samples of 8x8 stripe images
// horizontal stripes: class 0
// vertical stripes: class 1
// x: (N, 1, 8, 8), labels: (N,)
void create_patterns(tl::Tensor& x, std::vector<int>& labels, int n_per_class, std::mt19937& rng);
void print_image(const tl::Tensor& x, int sample_i);

int main() {
  std::mt19937 rng(std::random_device{}());

  // synthetic data

  // cnn
  tl::nn::Conv2d conv1(1, 8, 3, 1, 1); // (N, 1, 8, 8) -> (N, 8, 8, 8)
  tl::nn::Conv2d conv2(8, 16, 3, 1, 1); // (N, 8, 8, 8) -> (N, 16, 8, 8)
  tl::nn::Linear fc(16 * 8 * 8, 2); // fully connected classification head

  // optim

  // training loop


  // eval


  return 0;
}
