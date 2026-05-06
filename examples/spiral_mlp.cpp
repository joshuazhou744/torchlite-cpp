#include <tl/tensor.h>
#include <tl/activation.h>
#include <tl/nn.h>
#include <tl/optim.h>
#include <tl/ops.h>
#include <tl/loss.h>

#include <iostream>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

static const float PI = 3.1415926535897f;

void create_spiral(tl::Tensor& x, std::vector<int>& labels, int n_per_class, std::mt19937& rng) {
  std::normal_distribution<float> noise(0.0f, 0.1f);
  int N = n_per_class * 2;
  x = tl::Tensor({N, 2});
  labels.resize(N);

  for (int64_t c = 0; c < 2; ++c) {
    for (int64_t i = 0; i < n_per_class; ++i) {
      float t = static_cast<float>(i) / n_per_class;
      float angle = t * 4.0f * PI + c * PI; // angle: a(t) = t*4(pi) + c(pi)
      float r = t; // radius grows proportional to t
      int64_t index = c * n_per_class + i; // arm 0 or arm 1
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
  create_spiral(x_train, y_train, 200, rng);

  tl::Tensor x_test;
  std::vector<int> y_test;
  create_spiral(x_test, y_test, 50, rng);

  std::cout << "train: " << x_train.sizes()[0] << " points\n";
  std::cout << "test: " << x_test.sizes()[0] << " points\n";

  // std::cout << "first 4 x-values of arm 0:\n";
  // for (int64_t i = 0; i < 4; ++i) {
  //   std::cout << " x=(" << x_train.data()[i*2] << ", " << x_train.data()[i*2+1] << ") label=" << y_train[i] << "\n";
  // }

  tl::nn::Linear l1(2, 32); // project 2 input features (x, y) to hidden dim of 16
  tl::nn::Linear l2(32, 32); // hidden layer -> hidden layer
  tl::nn::Linear l3(32, 2); // project hidden dim back to 2 class logits (classification head)

  std::vector<tl::Tensor*> params;
  for (auto *p: l1.parameters()) params.push_back(p);
  for (auto *p: l2.parameters()) params.push_back(p);
  for (auto *p: l3.parameters()) params.push_back(p);

  tl::Adam opt(params, 0.005f);

  // training loop
  for (int64_t step = 0; step < 3000; ++step) {
    tl::Tensor h1 = tl::relu(l1.forward(x_train)); // activation after first layer
    tl::Tensor h2 = tl::relu(l2.forward(h1)); // activation after second layer
    tl::Tensor logits = l3.forward(h2); // classification head
    tl::Tensor loss = tl::cross_entropy_loss(logits, y_train); // calculate loss

    opt.zero_grad();
    loss.backward();
    opt.step();

    if (step % 100 == 0 ) {
      std::cout << "step " << step << " loss =" << loss.data()[0] << "\n";
    }

  }

  // training accuracy
  // overfit check
  {
      tl::Tensor h1 = tl::relu(l1.forward(x_train));
      tl::Tensor h2 = tl::relu(l2.forward(h1));
      tl::Tensor logits = l3.forward(h2);
      int64_t correct = 0;
      for (int64_t i = 0; i < (int64_t)y_train.size(); ++i) {
          float s0 = logits.data()[i*2], s1 = logits.data()[i*2+1];
          if (((s1 > s0) ? 1 : 0) == y_train[i]) ++correct;
      }
      std::cout << "train accuracy: " << correct << "/" << y_train.size() << "\n";
  }

  // evaluate on test step (inference only, no loss)
  {
    tl::Tensor h1 = tl::relu(l1.forward(x_test)); // activation after first layer
    tl::Tensor h2 = tl::relu(l2.forward(h1)); // activation after second layer
    tl::Tensor logits = l3.forward(h2); // classification head

    int64_t correct = 0;
    for (int64_t i = 0; i < (int64_t)y_test.size(); ++i) {
      float score0 = logits.data()[i * 2];
      float score1 = logits.data()[i * 2 + 1];
      int pred = (score1 > score0) ? 1 : 0;
      if (pred == y_test[i]) ++correct;
    }

    std::cout << "test accuracy: " << correct << "/" << y_test.size() << " (" << 100.0f * correct / y_test.size() << "%)\n";
  }

  return 0;
}
