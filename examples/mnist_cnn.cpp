// mnist example with depthwise-separable architecture
#include <tl/tensor.h>
#include <tl/ops.h>
#include <tl/nn.h>
#include <tl/factory.h>
#include <tl/loss.h>
#include <tl/optim.h>
#include <tl/autograd.h>

#include <iostream>
#include <fstream>
#include <cstdint>
#include <vector>
#include <string>
#include <numeric>
#include <stdexcept>
#include <random>
#include <algorithm>

// read big endian header
static uint32_t read_be32(std::ifstream& f) {
  uint8_t b[4]; // buffer for first 4 bytes
  f.read(reinterpret_cast<char*>(b), 4);
  if (!f) throw std::runtime_error("read_be32: stream error");
  // shift first 4 raw bytes to a single 32 bit value
  return (uint32_t(b[0]) << 24) | (uint32_t(b[1]) << 16) | (uint32_t(b[2]) << 8) | uint32_t(b[3]);
}

// load MNIST dataset
// max = 0 means to load all
void load_mnist(const std::string& images_path, const std::string& labels_path, int max_count, tl::Tensor& x, std::vector<int>& labels) {
  std::ifstream fi(images_path, std::ios::binary);
  if (!fi) throw std::runtime_error("could not open " + images_path);
  std::ifstream fl(labels_path, std::ios::binary);
  if (!fl) throw std::runtime_error("could not open " + labels_path);

  // image file header: magic, count, rows, cols
  uint32_t imag = read_be32(fi);
  if (imag != 0x00000803) throw std::runtime_error("bad image magic in " + images_path);
  int64_t N = read_be32(fi);
  int64_t rows = read_be32(fi);
  int64_t cols = read_be32(fi);

  // label file header: magic, count
  uint32_t lmag = read_be32(fl);
  if (lmag != 0x00000801) throw std::runtime_error("bad label magic in " + labels_path);
  int64_t NL = read_be32(fl);
  if (NL != N) throw std::runtime_error("image/label count mismatch");

  if (max_count > 0 && (int64_t)max_count < N) N = max_count;

  x = tl::zeros({N, 1, rows, cols});
  labels.resize(N);

  std::vector<uint8_t> img_buf(rows * cols);
  for (int64_t i = 0; i < N; ++i) {
    fi.read(reinterpret_cast<char*>(img_buf.data()), rows*cols);
    float* dst = x.data() + i * rows * cols;
    for (int64_t p = 0; p < rows * cols; ++p) {
      dst[p] = img_buf[p] / 255.0f; // normalize uint8 [0, 255] -> float [0, 1]
    }
    uint8_t lab;
    fl.read(reinterpret_cast<char*>(&lab), 1);
    labels[i] = static_cast<int>(lab);
  }
}

static void make_batch(
    const tl::Tensor& x_full,
    const std::vector<int>& y_full,
    const std::vector<int>& index,
    int start,
    int batch_size,
    tl::Tensor& x_batch,
    std::vector<int>& y_batch
  )
{
  const int64_t S = 1 * 28 * 28; // one sample has 784 floats
  for (int b = 0; b < batch_size; ++b) {
    int src = index[start + b];
    const float* sp = x_full.data() + src * S;
    float* dp = x_batch.data() + b * S;
    for (int64_t i = 0; i < S; ++i) dp[i] = sp[i];
    y_batch[b] = y_full[src];
  }
}

int main() {
  const int max_train = 5000;
  const int max_test = 1000;

  const std::string data_dir = "examples/data/mnist/";
  tl::Tensor x_train, x_test;
  std::vector<int> y_train, y_test;

  std::cout << "loading mnist\n";
  load_mnist(data_dir + "train-images-idx3-ubyte", data_dir + "train-labels-idx1-ubyte", max_train, x_train, y_train);
  load_mnist(data_dir + "t10k-images-idx3-ubyte", data_dir + "t10k-labels-idx1-ubyte", max_test, x_test, y_test);
  std::cout << "train: " << y_train.size() << " samples\n";
  std::cout << "test: " << y_test.size() << " samples\n";

  // build model
  // first conv block: 1 -> 16 channels
  tl::nn::Conv2d conv1(1, 16, 3, 1, 1);
  tl::nn::BatchNorm2d bn1(16);
  tl::nn::ReLU r1;
  tl::nn::MaxPool2d pool1(2, 2);

  // depthwise-separable block: 16 -> 32 channels, then pool to 7x7
  tl::nn::Conv2d conv_dw(16, 16, 3, 1, 1, 16); // depthwise
  tl::nn::BatchNorm2d bn_dw(16);
  tl::nn::ReLU r2;
  tl::nn::Conv2d conv_pw(16, 32, 1); // pointwise
  tl::nn::BatchNorm2d bn_pw(32);
  tl::nn::ReLU r3;
  tl::nn::MaxPool2d pool2(2, 2);

  // classification head
  tl::nn::Flatten flat;
  tl::nn::Linear fc1(32 * 7 * 7, 64); // 32 channels * 7x7 = 1568 features per image
  tl::nn::ReLU r4;
  tl::nn::Dropout drop(0.5f); // dropout higher because overfitting occurs mostly at classifier head
  tl::nn::Linear fc2(64, 10); // project to 10 MNIST classes

  // build the Sequential model
  tl::nn::Sequential model({
    &conv1, &bn1, &r1, &pool1,
    &conv_dw, &bn_dw, &r2,
    &conv_pw, &bn_pw, &r3, &pool2,
    &flat, &fc1, &r4, &drop, &fc2
  });

  // print model size (sanity check)
  auto params = model.parameters();
  int64_t total = 0;
  for (auto* p: params) total += p->numel();
  std::cout <<"model params: " << total << "\n";

  // optimizer
  tl::Adam opt(params, 2e-3f);

  // pre training setup
  const int batch_size = 32;
  const int epochs = 5;
  const int N_train = (int)y_train.size();
  const int N_test = (int)y_test.size();

  std::mt19937 rng(0xC0FFEE);
  std::vector<int> index(N_train);

  tl::Tensor x_batch = tl::zeros({batch_size, 1, 28, 28});
  std::vector<int> y_batch(batch_size);

  for (int epoch = 0; epoch < epochs; ++epoch) {
    // fresh shuffle
    std::iota(index.begin(), index.end(), 0);
    std::shuffle(index.begin(), index.end(), rng);

    drop.set_training(true); // activate dropout

    float running_loss = 0.0f;
    int batches = 0;
    const int total_batches = N_train / batch_size;
    for (int start = 0; start + batch_size <= N_train; start += batch_size) {
      make_batch(x_train, y_train, index, start, batch_size, x_batch, y_batch);

      tl::Tensor logits = model.forward(x_batch);
      tl::Tensor loss = tl::cross_entropy_loss(logits, y_batch);

      opt.zero_grad();
      loss.backward();
      opt.step();
      tl::release_graph(loss);

      running_loss += loss.data()[0];
      ++batches;
      if (batches % 25 == 0) {
        std::cout << "epoch " << (epoch+1)
                  << " batch " << batches << "/" << total_batches
                  << " loss= " << (running_loss / 25.0f) << "\n";
        running_loss = 0.0f;
      }
    }
    std::cout << "epoch " << (epoch + 1) << " done\n";
  }

  // evaluation
  drop.set_training(false);

  std::vector<int> eval_index(N_test);
  std::iota(eval_index.begin(), eval_index.end(), 0);

  int correct = 0;
  int seen = 0;
  for (int start = 0; start + batch_size <= N_test; start += batch_size) {
    make_batch(x_test, y_test, eval_index, start, batch_size, x_batch, y_batch);

    tl::Tensor logits = model.forward(x_batch);
    for (int b = 0; b < batch_size; ++b) {
      int pred = 0;
      for (int k = 1; k < 10; ++k) {
        if (logits.data()[b * 10 + k] > logits.data()[b * 10 + pred]) pred = k;
      }
      if (pred == y_batch[b]) ++correct; // correct prediction
      ++seen;
    }
  }
  std::cout << "test accuracy: " << correct << "/" << seen
            << " = " << (100.0f * correct / seen) << "%\n";

  return 0;
}
