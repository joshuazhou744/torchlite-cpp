#include <tl/factory.h>

#include <cstdint> // int64_t
#include <random> // randn
#include <stdexcept>
#include <fstream>
#include <string>

namespace tl {

// tensor full of a value
Tensor full(const std::vector<int64_t>& sizes, float value) {
  Tensor out(sizes);
  float *op = out.data();
  const int64_t n = out.numel();
  for (int64_t i = 0; i < n; ++i) {
    op[i] = value;
  }
  return out;
}
// tensor of zeros
Tensor zeros(const std::vector<int64_t>& sizes) {
  return full(sizes, 0.0f);
}

// tensor of ones
Tensor ones(const std::vector<int64_t>& sizes) {
  return full(sizes, 1.0f);
}

// tensor of random values from standard distribution
Tensor randn(const std::vector<int64_t>& sizes) {
  Tensor out(sizes);
  float *op = out.data();
  const int64_t n = out.numel();

  std::mt19937 gen(std::random_device{}()); // random engine
  std::normal_distribution<float> dist(0.0f, 1.0f);
  for (int64_t i = 0; i < n; ++i) {
    op[i] = dist(gen);
  }
  return out;
}

// 1D tensor of sequential integers
Tensor arange(int start, int end) {
  if (end <= start) {
    throw std::invalid_argument("arange: end must be greater than start");
  }
  int64_t size = end - start;
  Tensor out({size});
  float* op = out.data();
  for (int64_t i = 0; i < size; ++i) {
    op[i] = static_cast<float>(start + i);
  }
  return out;
}

// load PyTorch tensors
Tensor load(const std::string& path, const std::vector<int64_t>& sizes) {
  Tensor out(sizes);
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    throw std::invalid_argument("load: cannot open file " + path);
  }
  file.read(reinterpret_cast<char*>(out.data()), out.numel() * sizeof(float));
  return out;
}

}
