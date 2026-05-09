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
void create_patterns(tl::Tensor& x, std::vector<int>& labels, int n_per_class, std::mt19937& rng) {
  int N = n_per_class * 2;
  x = tl::zeros({N, 1, 8, 8});
  labels.resize(N);

  std::uniform_int_distribution<int> period_dist(2, 4);
  std::normal_distribution<float> noise(0.0f, 0.1f);

  for (int c = 0; c < 2; ++c) {
    for (int i = 0; i < n_per_class; ++i) {
      int index = c * n_per_class + i; // index of each pattern in x
      labels[index] = c; // class label
      int period = period_dist(rng); // get random period (2-4)
      // build out the pattern of an 8x8 grid
      for (int h = 0; h < 8; ++h) {
        for (int w = 0; w < 8; ++w) {
          // each period is (period/2) bright, followed by (period/2) dark stripes
          float val = (c == 0)
            ? (h % period < period / 2 ? 1.0f : 0.0f) // horizontal: vary by row
            : (w % period < period / 2 ? 1.0f : 0.0f); // vertical: vary by col
          val += noise(rng); // add noise
          x.data()[index * 64 + h * 8 + w] = val; // write to the flat data of x
        }
      }
    }
  }
}


void print_image(const tl::Tensor& x, int sample_i) {
  const float* data = x.data() + sample_i * 64; // each sample is 8x8 = 64 values
  for (int h = 0; h < 8; ++h) {
    for (int w = 0; w < 8; ++w) {
      std::cout << (data[h * 8 + w] > 0.5f ? "#" : ".") << " ";
    }
    std::cout << "\n";
  }
}

int main() {
  std::mt19937 rng(std::random_device{}());

  // generate data
  tl::Tensor x_train, x_test;
  std::vector<int> y_train, y_test;
  create_patterns(x_train, y_train, 200, rng);
  create_patterns(x_test, y_test, 50, rng);

  // cnn
  tl::nn::Conv2d conv1(1, 8, 3, 1, 1); // (N, 1, 8, 8) -> (N, 8, 8, 8)
  tl::nn::Conv2d conv2(8, 16, 3, 1, 1); // (N, 8, 8, 8) -> (N, 16, 8, 8)
  tl::nn::Linear fc(16 * 8 * 8, 2); // fully connected classification head

  // extract params from cnn
  std::vector<tl::Tensor*> params;
  auto p1 = conv1.parameters();
  auto p2 = conv2.parameters();
  auto p3 = fc.parameters();
  params.insert(params.end(), p1.begin(), p1.end());
  params.insert(params.end(), p2.begin(), p2.end());
  params.insert(params.end(), p3.begin(), p3.end());

  // optim
  tl::Adam opt(params, 0.0001f);

  // training loop
  for (int step = 0; step < 500; ++step) {
    // forward pass
    tl::Tensor out1 = tl::relu(conv1.forward(x_train));
    tl::Tensor out2 = tl::relu(conv2.forward(out1));
    tl::Tensor flat = tl::reshape(out2, {out2.sizes()[0], 16 * 8 * 8});
    tl::Tensor logits = fc.forward(flat);

    // loss function
    tl::Tensor loss = tl::cross_entropy_loss(logits, y_train);

    // backpropagation
    opt.zero_grad();
    loss.backward();
    opt.step();

    if (step % 100 == 0) {
        std::cout << "step " << step << " loss: " << loss.data()[0] << "\n";
    }
  }



  // eval


  return 0;
}
