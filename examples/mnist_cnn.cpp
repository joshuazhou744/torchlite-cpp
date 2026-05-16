// mnist example with depthwise-separable architecture
#include <tl/tensor.h>
#include <tl/ops.h>
#include <tl/nn.h>
#include <tl/factory.h>

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
  return (uint32_t(b[0] << 24) | (uint32_t(b[1] << 16) | (uint32_t(b[2]) << 8) | (uint32_t(b[3]);
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
  if (imag != 0x00000803) throw std::runtime_error("bad image magic in " + image_path);
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

  }
}
