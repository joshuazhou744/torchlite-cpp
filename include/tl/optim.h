#pragma once

#include <tl/tensor.h>

#include <vector>
#include <unordered_map>
#include <cstdint>

namespace tl {

void zero_grad(const std::vector<Tensor*>& params);

class SGD {
public:
  SGD(
      const std::vector<Tensor*>& params,
      float lr,
      float momentum = 0.0f,
      float weight_decay = 0.0f
  );
  void step();
  void zero_grad();

private:
  std::vector<Tensor*> params_;
  float lr_, momentum_, weight_decay_;
  std::unordered_map<Tensor*, Tensor> velocity_;
};

class Adam {
public:
  Adam(
      const std::vector<Tensor*>& params,
      float lr = 1e-3f,
      float beta1 = 0.9f,
      float beta2 = 0.999f,
      float eps = 1e-8f,
      float weight_decay = 0.0f
  );
  void step();
  void zero_grad();

private:
  std::vector<Tensor*> params_;
  float lr_, beta1_, beta2_, eps_, weight_decay_;
  int64_t t_ = 0;
  std::unordered_map<Tensor*, Tensor> m_, v_;
};

class AdamW {
public:
  AdamW(
      const std::vector<Tensor*>& params,
      float lr = 1e-3f,
      float beta1 = 0.9f,
      float beta2 = 0.999f,
      float eps = 1e-8f,
      float weight_decay = 1e-2f
  );
  void step();
  void zero_grad();

private:
  std::vector<Tensor*> params_;
  float lr_, beta1_, beta2_, eps_, weight_decay_;
  int64_t t_ = 0;
  std::unordered_map<Tensor*, Tensor> m_, v_;
};

} // namespace tl
