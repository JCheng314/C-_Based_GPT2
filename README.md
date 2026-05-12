# High-Performance Distributed Transformer Training Engine

**Stanford CME 213 Final Project**

This repository contains a from-scratch CUDA implementation of a Transformer training engine designed for NanoGPT / GPT-2 Small. It eschews high-level deep learning frameworks (like PyTorch or cuBLAS) in favor of custom-written, highly optimized CUDA kernels to maximize hardware utilization and memory bandwidth.

## 🚀 Current Status: Milestone 3 (Single-GPU Optimization)

The project has achieved the goals of **Milestone 3**, focusing on single-GPU correctness, deep kernel performance tuning, and automated profiling. 

Distributed parallelisms (MPI Pipeline and Data Parallelism) will be integrated in Milestone 4.

### Key Enhancements & Features:
* **Deep Performance Tuning (Vectorization)**: The LayerNorm kernel utilizes `float4` instructions to fetch 128-bits per memory transaction, drastically increasing HBM (High-Bandwidth Memory) utilization for memory-bound operators.
* **Tiled FlashAttention**: Implements true tiling over the sequence length ($T$) and head dimension ($HS$), maintaining running maximums and normalizers to support arbitrarily long sequences with $O(1)$ SRAM footprint.
* **Custom Matrix Multiplication (GEMM)**: Implemented using thread-tiling and shared memory blocking to minimize global memory roundtrips.
* **Kernel Fusion**: Reduced memory stalls and kernel launch overheads by fusing operators such as MatMul+Bias+Gelu and Residual+LayerNorm.
* **Automated CPU Validation**: A pure C++ baseline implementation validates the numerical correctness of all critical GPU kernels (GEMM, LayerNorm, Attention) automatically during the benchmarking process.

## 📁 Project Structure

```text
.
├── Makefile                        # Build configurations for NVCC
├── include/
│   └── transformer_kernels.h       # Shared macros, tile sizes, and kernel signatures
└── src/
    ├── main.cu                     # Benchmark driver, CPU-GPU validation, and reporting
    ├── cpu_reference.cpp           # CPU baseline math for correctness testing
    ├── linear_layers.cu            # Custom GEMM, Bias, Gelu, and Embeddings kernels
    ├── flash_attention.cu          # Tiled FlashAttention and QKV kernels
    └── fused_layernorm.cu          # Vectorized (float4) LayerNorm kernels
```

## 🛠️ Build and Execution

This project requires an NVIDIA GPU and the CUDA Toolkit (specifically `nvcc`). It is designed to be compiled and run on the Stanford compute nodes (e.g., Pascal architecture).

### Compilation

Ensure you are on a machine with CUDA installed, then run:

```bash
make clean
make
```

This will compile the source files using `-O3` optimization and `-arch=sm_80` (adjust the architecture flag in the `Makefile` if necessary) and generate the `gpt2_benchmark` executable.

### Running the Benchmark

```bash
./gpt2_benchmark
```

The automated benchmark suite will:
1. Initialize synthetic weights and input tensors.
2. Execute the CPU reference implementations for GEMM, LayerNorm, and Attention.
3. Warm-up the GPU.
4. Run a performance benchmarking loop using `cudaEvent_t`.
5. Verify the GPU output against the CPU reference for strict numerical correctness (`PASS`/`FAIL`).
6. Export a detailed report (Time, Bandwidth in GB/s, and GFLOPS) to `output.txt`.

## 📊 Viewing Results

After the run finishes, view the generated report:
```bash
cat output.txt
```
Copy these metrics directly into your CME 213 Progress Report!
