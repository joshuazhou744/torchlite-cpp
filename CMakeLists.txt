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

# build tests flag
option(BUILD_TESTS "Build tests" ON)

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
  )
  target_link_libraries(run_tests PRIVATE torchlite)
endif()


# create linear regression example
add_executable(linear_regression examples/linear_regression.cpp)
target_link_libraries(linear_regression PRIVATE torchlite)

# create spiral mlp example
add_executable(spiral_mlp examples/spiral_mlp.cpp)
target_link_libraries(spiral_mlp PRIVATE torchlite)
