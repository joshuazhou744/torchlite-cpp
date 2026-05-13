#pragma once

#include <tl/tensor.h>
#include <vector>
#include <string>

namespace tl {
  void save_model(const std::string& path, const std::vector<Tensor*>& params);
  void load_model(const std::string& path, const std::vector<Tensor*>& params);
}
