# High-Performance Distributed Transformer Training Engine

**Stanford CME 213 Final Project**

This repository contains a from-scratch CUDA implementation of a Transformer training engine designed for NanoGPT / GPT-2 Small. It eschews high-level deep learning frameworks (like PyTorch or cuBLAS) in favor of custom-written, highly optimized CUDA kernels to maximize hardware utilization and memory bandwidth.

## 🚀 Current Status: Milestone 3 (Single-GPU Optimization)

The project is currently at **Milestone 3**, focusing on single-GPU correctness, kernel modularity, and baseline performance profiling. 

Distributed parallelisms (MPI Pipeline and Data Parallelism) will be integrated in Milestone 4.

### Key Features Implemented:
* **Custom Matrix Multiplication (GEMM)**: Implemented using thread-tiling and shared memory blocking to minimize global memory roundtrips.
* **Kernel Fusion**: Reduced memory stalls and kernel launch overheads by fusing operators such as MatMul+Bias+Gelu and Residual+LayerNorm.
* **Simplified FlashAttention**: Utilizes shared memory to fuse the attention scores computation, causal masking, softmax, and attention value multiplication, bypassing the instantiation of the $N \times N$ attention matrix in High-Bandwidth Memory (HBM).
* **CPU Verification**: A pure C++ baseline implementation for validating the numerical correctness of the GPU kernels.

## 📁 Project Structure

```text
.
├── Makefile                        # Build configurations for NVCC
├── include/
│   └── transformer_kernels.h       # Shared macros, tile sizes, and kernel signatures
└── src/
    ├── main.cu                     # Host driver, memory allocation, benchmarking, and validation
    ├── cpu_reference.cpp           # CPU implementations for correctness testing
    ├── linear_layers.cu            # Custom GEMM, Bias, Gelu, and Embeddings kernels
    ├── flash_attention.cu          # Fused attention and QKV handling kernels
    └── fused_layernorm.cu          # LayerNorm and Residual+LayerNorm kernels
```

## 🛠️ Build and Execution

This project requires an NVIDIA GPU and the CUDA Toolkit (specifically `nvcc`). It is designed to be compiled and run on the Stanford compute nodes.

### Compilation

Ensure you are on a machine with CUDA installed, then run:

```bash
make
```

This will compile the source files using `-O3` optimization and `-arch=sm_80` (adjust the architecture flag in the `Makefile` if necessary) and generate the `gpt2_benchmark` executable.

### Running the Benchmark

```bash
./gpt2_benchmark
```

The benchmark will:
1. Initialize synthetic weights and input tensors.
2. Execute the CPU reference implementations.
3. Warm-up the GPU.
4. Run a benchmark loop using `cudaEvent_t` to measure elapsed time.
5. Print performance metrics (Average ms per iteration, GFLOPS).
6. Verify the GPU output against the CPU reference for correctness.

## 🧹 Cleanup

To remove compiled object files and the executable:

```bash
make clean
```
