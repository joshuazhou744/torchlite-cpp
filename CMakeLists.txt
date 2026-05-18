cmake_minimum_required(VERSION 3.16)
project(torchlite_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# library components
add_library(torchlite
    src/tensor.cpp
    src/ops.cpp
    src/activation.cpp
    src/factory.cpp
    src/nn.cpp
    src/autograd.cpp
    src/loss.cpp
    src/optim.cpp
    src/model_io.cpp
)

# Eigen library link
find_package(Eigen3 REQUIRED)
target_link_libraries(torchlite PUBLIC Eigen3::Eigen)


target_compile_options(torchlite PRIVATE
  $<$<CONFIG:Release>:-O3 -march=native -DNDEBUG>
  $<$<CONFIG:Debug>:-O0 -g>
)


target_include_directories(torchlite
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# build tests/examples only when torchlite is the top-level project by default
# (so downstream consumers via FetchContent/add_subdirectory don't build them)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(TORCHLITE_TOP_LEVEL ON)
else()
  set(TORCHLITE_TOP_LEVEL OFF)
endif()

option(BUILD_TESTS "Build tests" ${TORCHLITE_TOP_LEVEL})
option(BUILD_EXAMPLES "Build examples" ${TORCHLITE_TOP_LEVEL})

if(BUILD_TESTS)
  # create single test executable
  add_executable(run_tests
    tests/run_all_tests.cpp
    tests/test_tensor.cpp
    tests/test_ops.cpp
    tests/test_activation.cpp
    tests/test_factory.cpp
    tests/test_nn.cpp
    tests/test_autograd.cpp
    tests/test_loss.cpp
    tests/test_optim.cpp
    tests/test_model_io.cpp
  )
  target_link_libraries(run_tests PRIVATE torchlite)
endif()

if(BUILD_EXAMPLES)
  add_executable(linear_regression examples/linear_regression.cpp)
  target_link_libraries(linear_regression PRIVATE torchlite)

  add_executable(spiral_mlp examples/spiral_mlp.cpp)
  target_link_libraries(spiral_mlp PRIVATE torchlite)

  add_executable(pattern_cnn examples/pattern_cnn.cpp)
  target_link_libraries(pattern_cnn PRIVATE torchlite)

  add_executable(mnist_cnn examples/mnist_cnn.cpp)
  target_link_libraries(mnist_cnn PRIVATE torchlite)
endif()
