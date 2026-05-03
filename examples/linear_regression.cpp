#include <tl/tensor.h>
#include <tl/ops.h>
#include <tl/optim.h>
#include <tl/loss.h>
#include <tl/factory.h>

#include <iostream>
#include <cmath>
#include <cstdint>
#include <random>

int main() {
  // generate synthetic data
  std::normal_distribution<float> noise(0.0f, 0.05f);
  std::mt19937 rng(std::random_device{}());
  const int64_t N = 100;
  tl::Tensor x({N, 1});
  tl::Tensor y({N, 1});
  for (int64_t i = 0; i < N; ++i) {
    float xi = static_cast<float>(i) / N;
    x.data()[i] = xi;
    y.data()[i] = 2.0f * xi + 3.0f + noise(rng);
  }

  // params: pred = x @ w + b, w shape (1, 1), b shape (1,)
  tl::Tensor w = tl::randn({1,1});
  tl::Tensor b = tl::zeros({1});
  w.set_requires_grad(true);
  b.set_requires_grad(true);

  tl::SGD opt({&w, &b}, 0.3f);

  // training loop
  for (int step = 0; step < 1000; ++step) {
    tl::Tensor pred = tl::add(tl::matmul(x, w), b);
    tl::Tensor loss = tl::mse_loss(pred, y);

    opt.zero_grad();
    loss.backward();
    opt.step();

    // print logs every 20 steps
    if (step % 100 == 0) {
      std::cout << "step " << step
        << " loss=" << loss.data()[0]
        << " w=" << w.data()[0]
        << " b=" << b.data()[0] << "\n";
    }
  }

  std::cout << "\nfinal:\nw=" << w.data()[0] << " (expected 2.0)\n"
    << "b=" << b.data()[0] << " (expected 3.0)\n";

  return 0;
}
