#include "../include/transformer_kernels.h"
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>

// Forward declarations from CPU reference
extern void cpu_gemm(const float* A, const float* W, float* C, int M, int K, int N);

void init_matrix(std::vector<float>& mat) {
    for (size_t i = 0; i < mat.size(); ++i) {
        mat[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

bool check_correctness(const std::vector<float>& a, const std::vector<float>& b, float tol = 1e-4) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::abs(a[i] - b[i]) > tol) {
            std::cout << "Mismatch at " << i << ": " << a[i] << " vs " << b[i] << "\n";
            return false;
        }
    }
    return true;
}

int main() {
    std::cout << "Starting GPT-2 Kernel Benchmark...\n";

    // Matrix dimensions
    int M = 128;
    int K = 256;
    int N = 128;

    std::vector<float> h_A(M * K);
    std::vector<float> h_W(K * N);
    std::vector<float> h_C_cpu(M * N, 0.0f);
    std::vector<float> h_C_gpu(M * N, 0.0f);

    init_matrix(h_A);
    init_matrix(h_W);

    // CPU execution
    cpu_gemm(h_A.data(), h_W.data(), h_C_cpu.data(), M, K, N);

    // GPU execution
    float *d_A, *d_W, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_W, K * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_W, h_W.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(THREADS_X, THREADS_Y);
    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);

    // Warmup
    hierarchical_gemm_2x4_kernel<<<grid, block>>>(d_A, d_W, d_C, M, K, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    int iters = 100;
    for (int i = 0; i < iters; ++i) {
        hierarchical_gemm_2x4_kernel<<<grid, block>>>(d_A, d_W, d_C, M, K, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    float avg_ms = ms / iters;
    
    // 2 operations per MAC, M * N * K elements
    double gflops = (2.0 * M * N * K) / (avg_ms * 1e-3) / 1e9;
    
    std::cout << "GEMM " << M << "x" << K << "x" << N << " -> Time: " << avg_ms << " ms, GFLOPS: " << gflops << "\n";

    CHECK_CUDA(cudaMemcpy(h_C_gpu.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    if (check_correctness(h_C_cpu, h_C_gpu)) {
        std::cout << "Correctness check: PASS\n";
    } else {
        std::cout << "Correctness check: FAIL\n";
    }

    cudaFree(d_A);
    cudaFree(d_W);
    cudaFree(d_C);

    return 0;
}
