#pragma once
#include <iostream>
#include <cstdlib>

#define CHECK(cond) do { \
  if (!(cond)) { \
    std::cerr << "FAIL: " #cond " at " __FILE__ ":" << __LINE__ << "\n"; \
    std::abort(); \
  } \
} while(0)
