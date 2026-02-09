cmake_minimum_required(VERSION 3.16)
project(torchlite_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# library components
add_library(torchlite
    src/tensor.cpp
    src/ops.cpp
)

target_include_directories(torchlite
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# create single test executable
add_executable(run_tests
  tests/run_all_tests.cpp
  tests/test_tensor.cpp
  tests/test_ops.cpp
)
target_link_libraries(run_tests PRIVATE torchlite)
