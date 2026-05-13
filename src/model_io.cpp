#include <tl/model_io.h>

#include <cstdint>
#include <stdexcept>
#include <fstream>

// File format:
// [magic: 4 bytes = "TLMD"]
// [version: int32_t = 1]
// [num_tensors: int64_t]
// for each tensor:
//  [ndim: int64_t]
//  [shape: int64_t * ndim]
//  [data: float * numel]

namespace tl {

static constexpr char MAGIC[4] = {'T', 'L', 'M', 'D'}; // torchlite model
static constexpr int32_t VERSION = 1;

void save_model(const std::string& path, const std::vector<Tensor*>& params) {
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    throw std::runtime_error("save_model: cannot open file: " + path);
  }

  // file header: magic + version
  out.write(MAGIC, 4);
  out.write(reinterpret_cast<const char*> (&VERSION), sizeof(int32_t));

  // number of tensors
  int64_t num_tensors = static_cast<int64_t> (params.size());
  out.write(reinterpret_cast<const char*> (&num_tensors), sizeof(int64_t));

  // each tensor is shape: ndim, shape, data
  for (Tensor* p: params) {
    Tensor t = p->contiguous();
    int64_t ndim = static_cast<int64_t> (t.sizes().size());
    out.write(reinterpret_cast<const char*> (&ndim), sizeof(int64_t));
    out.write(reinterpret_cast<const char*> (t.sizes().data()), ndim * sizeof(int64_t));
    out.write(reinterpret_cast<const char*> (t.data()), t.numel() * sizeof(float));
  }
}

void load_model(const std::string& path, const std::vector<Tensor*>& params) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("load_model: cannot open file: " + path);
  }

  // verify magic (TLMD)
  char magic[4];
  in.read(magic, 4);
  for (int i = 0; i < 4; ++i) {
    if (magic[i] != MAGIC[i]) {
      throw std::runtime_error("load_model: no TLMD magic header in " + path);
    }
  }

  // verify version
  int32_t version;
  in.read(reinterpret_cast<char*> (&version), sizeof(int32_t));
  if (version != VERSION) {
      throw std::runtime_error("load_model: unsupported version");
  }

  // verify tensor count
  int64_t num_tensors;
  in.read(reinterpret_cast<char*> (&num_tensors), sizeof(int64_t));
  if (num_tensors != static_cast<int64_t> (params.size())) {
      throw std::runtime_error("load_model: tensor count mismatch");
  }

  // verify stored tensors: ndim, shape, data
  for (Tensor* p: params) {
    int64_t ndim;
    in.read(reinterpret_cast<char*> (&ndim), sizeof(int64_t));
    if (ndim != static_cast<int64_t> (p->sizes().size())) {
        throw std::runtime_error("load_model: ndim mismatch");
    }

    std::vector<int64_t> shape(ndim);
    in.read(reinterpret_cast<char*> (shape.data()), ndim * sizeof(int64_t));
    for (int64_t i = 0; i < ndim; ++i) {
      if (shape[i] != p->sizes()[i]) {
        throw std::runtime_error("load_model: shape mismatch");
      }
    }

    in.read(reinterpret_cast<char*> (p->data()), p->numel() * sizeof(float));
  }
}

}
