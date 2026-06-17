# torchlite-cpp

A lightweight C++ tensor library implementing core PyTorch-like operations for CPU. Designed as a minimal foundation for building neural networks without the complexity of full frameworks.

## Core Features

**Tensor Operations**
- N-dimensional tensor data structure with automatic stride calculation
- Element-wise operations (add, sub, mul, div, neg, exp, log, pow, sqrt, abs, clamp) with NumPy-style broadcasting
- Matrix multiplication with batch dimension support; flash attention
- Convolution and pooling: conv2d, max_pool2d, avg_pool2d
- Reductions: sum, mean, variance, argmax, softmax
- Reshape, transpose, cat, stack, slice, pad

**Neural Network Modules**
- Linear, Conv2d, MaxPool2d, AvgPool2d, Flatten
- LayerNorm, BatchNorm2d, InputNormalize, Dropout
- MultiHeadAttention, TransformerEncoderLayer, TransformerEncoder
- PositionalEncoding
- Activation functions: ReLU, GELU, Sigmoid

**Training**
- Reverse-mode autograd (backward computation graph) over all operations
- Optimizers: SGD, Adam, AdamW
- Loss functions: MSE, MAE, BCE, NLL, Cross-entropy

**Design**
- CPU-only, float32 operations
- Weight loading from raw binary files

## Use in your project (CMake)

Pull `torchlite` into your CMake project with FetchContent:

```
include(FetchContent)

FetchContent_Declare(
    torchlite
    GIT_REPOSITORY https://github.com/joshuazhou744/torchlite-cpp
    GIT_TAG main
)
FetchContent_MakeAvailable(torchlite)

target_link_libraries(myapp PRIVATE torchlite)
```

Then `#include <tl/tensor.h>` in your code. The external dependencies (e.g., Eigen3) will propagate automatically.


## Build from source

```bash
mkdir build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

The **Release** build enables optimizations (`-O3 march=native -DNDEBUG`):
- `-O3`: aggressive compiler optimizations
- `-march=native`: targets CPU specific instruction set
- `-DNDEBUG`: disables `assert()` calls in hot paths

Run tests (development):

```
./build/run_tests
```

## Project Structure

```
include/tl/         Public API headers (tensor, ops, nn, activation, factory, autograd)
include/external/   Third-party headers (LibrosaCpp)
src/                Implementation
tests/              Test executables
bench/              Operation benchmarks
examples/           Example usages of the library
```

## Examples

### CNN Binary Classifier
See [here](https://github.com/joshuazhou744/binary-classifier-tl) for a CNN binary classifier built using torchlite-cpp.

## Dependencies
- [Eigen3](https://eigen.tuxfamily.org/): required by LibrosaCpp for audio preprocessing
- [LibrosaCpp](https://github.com/ewan-xu/LibrosaCpp): single-header mel spectrogram computation (included in `include/external/`)
- [OpenMP](https://www.openmp.org/): for the multithreaded GEMM kernel

## Requirements

- C++17 or later
- CMake 3.16+
- Eigen3 (`sudo apt install libeigen3-dev`)
- OpenMP (`sudo apt install libgomp1`)

## License

MIT License

