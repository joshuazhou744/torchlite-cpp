## Project Structure

- `CMakeLists.txt`:
	- Defines how the project is built
	- Sets C++ standard
	- Builds the `torchlite` library

- `include/`:
	- Holds public headers
    - Defines the interface of the library
    - What users of the library are allowed to include when using

- `include/tl/`:
	- Public API
	- Declares all the modules

- `include/tl/*.h`:
    - Header files export interfaces
    - Tells the compiler what exists, how it's used, and what the types/function signatures are

- `src/`
    - Holds the implementation of the library
    - Actual code
    - Not directly included by users

## Tensor

A minimal recreation of LibTorch float tensor for CPU only.
Designed to:
- Store n-dimensional data
- Expose its shape and raw memory
- Support basic math operations
- Act as foundational for neural networks
What it won't have:
- autograd
- GPU optimized and aware

### Files

`include/tl/tensor.h`
`src/tensor.cpp`
`CMakeLists.txt`
`include/tl/ops.h`
`src/ops.cpp`
