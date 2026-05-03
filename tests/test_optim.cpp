#include <tl/tensor.h>
#include <tl/loss.h>

#include <iostream>
#include <cassert>
#include <cmath>

static bool close(float a, float b, float e = 1e-4f) {
  return std::abs(a - b) < e;
}

