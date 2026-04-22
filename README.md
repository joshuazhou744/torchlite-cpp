# torchlite-cpp

A lightweight C++ tensor library implementing core PyTorch-like operations for CPU. Designed as a minimal foundation for building neural networks without the complexity of full frameworks.

## Core Features

**Tensor Operations**
- N-dimensional tensor data structure with automatic stride calculation
- Element-wise operations (add, sub, mul, div, neg, exp, log, pow, clamp) with NumPy-style broadcasting
- Matrix multiplication with batch dimension support
- Reductions: sum, mean, variance, argmax, softmax
- Reshape, transpose, cat, stack, slice, pad

**Neural Network Modules**
- Linear, LayerNorm, Dropout
- MultiHeadAttention, TransformerEncoderLayer, TransformerEncoder
- PositionalEncoding
- Activation functions: ReLU, GELU, Sigmoid

**Design**
- CPU-only, float32 operations
- Weight loading from raw binary files

## Quick Start

Build the library:

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```

The **Release** build enables optimizations (`-O3 -match=native -DNDEBUG`):
- `-O3`: aggressive compiler optimizations
- `-match=native`: targets CPU specific instruction set
- `-DNDEBUG`: disables `assert()` calls in hot paths

Run tests:

```
./build/run_tests
```

## Project Structure

```
include/tl/         Public API headers (tensor, ops, nn, activation, factory, autograd)
include/external/   Third-party headers (LibrosaCpp)
src/                Implementation
tests/              Test executables
```

## Dependencies
- [Eigen3](https://eigen.tuxfamily.org/): required by LibrosaCpp for audio preprocessing
- [LibrosaCpp](https://github.com/ewan-xu/LibrosaCpp): single-header mel spectrogram computation (included in `include/external/`)

## Requirements

- C++17 or later
- CMake 3.10+
- Eigen3 (`sudo apt install libeigen3-dev`)

## Usage Notes

## License

MIT License

