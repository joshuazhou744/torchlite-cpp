# torchlite-cpp

A lightweight C++ tensor library implementing core PyTorch-like operations for CPU. Designed as a minimal foundation for building neural networks without the complexity of full frameworks.

## Core Features

**Tensor Operations**
- N-dimensional tensor data structure with automatic stride calculation
- Element-wise operations (add, multiply) with NumPy-style broadcasting
- Matrix multiplication with batch dimension support
- Arbitrary dimension transpose

**Neural Network Primitives**
- Activation functions: ReLU, Sigmoid, Softmax (numerically stable)
- Scalar operations and tensor scaling

**Design Philosophy**
- CPU-only, float32 operations
- No autograd or GPU support
- Minimal dependencies
- Clear, readable implementation

## Quick Start

Build the library:

```bash
mkdir build && cd build
cmake ..
make
```

Run tests:

```bash
./tests/test_tensor
./tests/test_ops
```

## Project Structure

```
include/tl/     Public API headers
src/            Implementation
tests/          Test executables
```
## Requirements

- C++17 or later
- CMake 3.10+

## License

MIT License

