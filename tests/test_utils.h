#pragma once
#include <iostream>
#include <cstdlib>
#include <cmath>

#define CHECK(cond) do { \
  if (!(cond)) { \
    std::cerr << "FAIL: " #cond " at " __FILE__ ":" << __LINE__ << "\n"; \
    std::abort(); \
  } \
} while(0)

inline bool is_close(float a, float b, float e = 1e-5f) {
  return std::abs(a - b) < e;
}
