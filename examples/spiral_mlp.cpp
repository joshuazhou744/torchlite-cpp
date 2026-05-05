#include <tl/tensor.h>

#include <iostream>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

static const float PI = 3.1415926535897f;

void create_spiral(tl::Tensor& x, std::vector<int>& labels, int n_per_class, std::mt19937 rng) {
  std::normal_distribution<float> noise(0.0f, 0.1f);
  int N = n_per_class * 2;
  x = tl::Tensor({N, 2});
  labels.resize(N);

  for (int c = 0; c < 2; ++c) {
    for (int i = 0; i < n_per_class; ++i) {
      float t = static_cast<float>(i) / n_per_class;
      float angle = t * 4.0f * PI + c * PI; // angle: a(t) = t*4(pi) + c(pi)
      float r = t; // radius grows proportional to t
      int index = c * n_per_class + i; // arm 1 or arm 2
      x.data()[index * 2 + 0] = r * std::cos(angle) + noise(rng); // x value
      x.data()[index * 2 + 1] = r * std::sin(angle) + noise(rng); // y value
      labels[index] = c; // class of the arm (0 or 1)
    }
  }
}

int main() {
  std::mt19937 rng(42);

  tl::Tensor x_train;
  std::vector<int> y_train;
  create_spiral(x_train, y_train, 100, rng);

  tl::Tensor x_test;
  std::vector<int> y_test;
  create_spiral(x_test, y_test, 50, rng);

  std::cout << "train: " << x_train.sizes()[0] << " points\n";
  std::cout << "test: " << x_test.sizes()[0] << " points\n";

  std::cout << "first 4 x-values of arm 0:\n";
  for (int i = 0; i < 4; ++i) {
    std::cout << " x=(" << x_train.data()[i*2] << ", " << x_train.data()[i*2+1] << ") label=" << y_train[i] << "\n";
  }

  return 0;
}
