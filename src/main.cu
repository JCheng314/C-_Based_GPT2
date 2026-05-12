#include "../include/transformer_kernels.h"
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <iomanip>

// Forward declarations from CPU reference
extern void cpu_gemm(const float* A, const float* W, float* C, int M, int K, int N);
extern void cpu_layernorm(const float* x, const float* gamma, const float* beta, float* out, int B, int T, int C, float eps);
extern void cpu_attention(const float* Q, const float* K, const float* V, float* out, int B, int NH, int T, int HS);

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
    std::ofstream outfile("output.txt");
    if (!outfile.is_open()) {
        std::cerr << "Failed to open output.txt for writing.\n";
        return 1;
    }
    
    outfile << "========================================\n";
    outfile << "  GPT-2 Kernel Benchmark & Validation   \n";
    outfile << "========================================\n\n";
    std::cout << "Starting GPT-2 Kernel Benchmark. Outputting to output.txt...\n";

    int iters = 100;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // ==========================================
    // 1. GEMM Benchmark
    // ==========================================
    {
        int M = 128, K = 256, N = 128;
        std::vector<float> h_A(M * K), h_W(K * N), h_C_cpu(M * N, 0.0f), h_C_gpu(M * N, 0.0f);
        init_matrix(h_A); init_matrix(h_W);

        cpu_gemm(h_A.data(), h_W.data(), h_C_cpu.data(), M, K, N);

        float *d_A, *d_W, *d_C;
        CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_W, K * N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_W, h_W.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));

        dim3 block(THREADS_X, THREADS_Y);
        dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);

        hierarchical_gemm_2x4_kernel<<<grid, block>>>(d_A, d_W, d_C, M, K, N);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(h_C_gpu.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

        bool correct = check_correctness(h_C_cpu, h_C_gpu);

        cudaEventRecord(start);
        for (int i = 0; i < iters; ++i) {
            hierarchical_gemm_2x4_kernel<<<grid, block>>>(d_A, d_W, d_C, M, K, N);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms; cudaEventElapsedTime(&ms, start, stop); float avg_ms = ms / iters;
        double gflops = (2.0 * M * N * K) / (avg_ms * 1e-3) / 1e9;
        
        outfile << "[1] GEMM Kernel (" << M << "x" << K << "x" << N << ")\n";
        outfile << "    Correctness: " << (correct ? "PASS" : "FAIL") << "\n";
        outfile << "    Time:        " << std::fixed << std::setprecision(4) << avg_ms << " ms\n";
        outfile << "    Performance: " << std::fixed << std::setprecision(2) << gflops << " GFLOPS\n\n";

        cudaFree(d_A); cudaFree(d_W); cudaFree(d_C);
    }

    // ==========================================
    // 2. LayerNorm Benchmark (Baseline vs Float4)
    // ==========================================
    {
        int B = 16, T = 1024, C = 768; // GPT-2 Small config
        float eps = 1e-5f;
        std::vector<float> h_x(B * T * C), h_gamma(C), h_beta(C);
        std::vector<float> h_out_cpu(B * T * C, 0.0f), h_out_gpu(B * T * C, 0.0f), h_out_gpu_f4(B * T * C, 0.0f);
        init_matrix(h_x); init_matrix(h_gamma); init_matrix(h_beta);

        cpu_layernorm(h_x.data(), h_gamma.data(), h_beta.data(), h_out_cpu.data(), B, T, C, eps);

        float *d_x, *d_gamma, *d_beta, *d_out;
        CHECK_CUDA(cudaMalloc(&d_x, B * T * C * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_gamma, C * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_beta, C * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_out, B * T * C * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), B * T * C * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_gamma, h_gamma.data(), C * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_beta, h_beta.data(), C * sizeof(float), cudaMemcpyHostToDevice));

        int block_size = 256;
        int grid_size = B * T;
        size_t shared_mem = 2 * block_size * sizeof(float);

        // Baseline LayerNorm
        layernorm_forward_kernel<<<grid_size, block_size, shared_mem>>>(d_x, d_gamma, d_beta, d_out, B, T, C, eps);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(h_out_gpu.data(), d_out, B * T * C * sizeof(float), cudaMemcpyDeviceToHost));
        bool correct_base = check_correctness(h_out_cpu, h_out_gpu);

        cudaEventRecord(start);
        for (int i = 0; i < iters; ++i) {
            layernorm_forward_kernel<<<grid_size, block_size, shared_mem>>>(d_x, d_gamma, d_beta, d_out, B, T, C, eps);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms_base; cudaEventElapsedTime(&ms_base, start, stop); ms_base /= iters;
        double bw_base = (2.0 * B * T * C * sizeof(float)) / (ms_base * 1e-3) / 1e9; // Read + Write

        // Float4 Vectorized LayerNorm
        layernorm_forward_kernel_float4<<<grid_size, block_size, shared_mem>>>(d_x, d_gamma, d_beta, d_out, B, T, C, eps);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(h_out_gpu_f4.data(), d_out, B * T * C * sizeof(float), cudaMemcpyDeviceToHost));
        bool correct_f4 = check_correctness(h_out_cpu, h_out_gpu_f4);

        cudaEventRecord(start);
        for (int i = 0; i < iters; ++i) {
            layernorm_forward_kernel_float4<<<grid_size, block_size, shared_mem>>>(d_x, d_gamma, d_beta, d_out, B, T, C, eps);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms_f4; cudaEventElapsedTime(&ms_f4, start, stop); ms_f4 /= iters;
        double bw_f4 = (2.0 * B * T * C * sizeof(float)) / (ms_f4 * 1e-3) / 1e9;

        outfile << "[2] LayerNorm Benchmark (B=" << B << ", T=" << T << ", C=" << C << ")\n";
        outfile << "    [Baseline 32-bit]\n";
        outfile << "      Correctness: " << (correct_base ? "PASS" : "FAIL") << "\n";
        outfile << "      Time:        " << std::fixed << std::setprecision(4) << ms_base << " ms\n";
        outfile << "      Bandwidth:   " << std::fixed << std::setprecision(2) << bw_base << " GB/s\n";
        outfile << "    [Vectorized float4]\n";
        outfile << "      Correctness: " << (correct_f4 ? "PASS" : "FAIL") << "\n";
        outfile << "      Time:        " << std::fixed << std::setprecision(4) << ms_f4 << " ms\n";
        outfile << "      Bandwidth:   " << std::fixed << std::setprecision(2) << bw_f4 << " GB/s\n";
        outfile << "      Speedup:     " << std::fixed << std::setprecision(2) << (ms_base / ms_f4) << "x\n\n";

        cudaFree(d_x); cudaFree(d_gamma); cudaFree(d_beta); cudaFree(d_out);
    }

    // ==========================================
    // 3. Tiled Flash Attention Benchmark
    // ==========================================
    {
        int B = 2, NH = 12, T = 1024, HS = 64; // GPT-2 Small config (batch 2 for speed)
        std::vector<float> h_Q(B * NH * T * HS), h_K(B * NH * T * HS), h_V(B * NH * T * HS);
        std::vector<float> h_out_cpu(B * NH * T * HS, 0.0f), h_out_gpu(B * NH * T * HS, 0.0f);
        init_matrix(h_Q); init_matrix(h_K); init_matrix(h_V);

        cpu_attention(h_Q.data(), h_K.data(), h_V.data(), h_out_cpu.data(), B, NH, T, HS);

        float *d_Q, *d_K, *d_V, *d_out;
        CHECK_CUDA(cudaMalloc(&d_Q, B * NH * T * HS * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_K, B * NH * T * HS * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_V, B * NH * T * HS * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_out, B * NH * T * HS * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(d_Q, h_Q.data(), B * NH * T * HS * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_K, h_K.data(), B * NH * T * HS * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_V, h_V.data(), B * NH * T * HS * sizeof(float), cudaMemcpyHostToDevice));

        dim3 block(HS); // Thread count = Head Size
        dim3 grid(T, NH, B);
        int Bc = 32; // Matches hardcoded tile size in kernel
        size_t shared_mem = HS * sizeof(float) + 2 * Bc * HS * sizeof(float) + Bc * sizeof(float);

        tiled_flash_attention_kernel<<<grid, block, shared_mem>>>(d_Q, d_K, d_V, d_out, B, NH, T, HS);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(h_out_gpu.data(), d_out, B * NH * T * HS * sizeof(float), cudaMemcpyDeviceToHost));
        
        bool correct = check_correctness(h_out_cpu, h_out_gpu, 1e-3); // Slightly higher tol for exp sum

        cudaEventRecord(start);
        for (int i = 0; i < iters; ++i) {
            tiled_flash_attention_kernel<<<grid, block, shared_mem>>>(d_Q, d_K, d_V, d_out, B, NH, T, HS);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms; cudaEventElapsedTime(&ms, start, stop); float avg_ms = ms / iters;
        
        // Approx MACs: Q*K^T (T^2 * HS / 2) + P*V (T^2 * HS / 2) = T^2 * HS per head
        double gflops = (2.0 * B * NH * T * T * HS) / (avg_ms * 1e-3) / 1e9;

        outfile << "[3] Tiled Flash Attention (B=" << B << ", NH=" << NH << ", T=" << T << ", HS=" << HS << ")\n";
        outfile << "    Correctness: " << (correct ? "PASS" : "FAIL") << "\n";
        outfile << "    Time:        " << std::fixed << std::setprecision(4) << avg_ms << " ms\n";
        outfile << "    Performance: " << std::fixed << std::setprecision(2) << gflops << " GFLOPS (approx)\n";

        cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_out);
    }

    outfile.close();
    std::cout << "Done! Please check output.txt for the full report.\n";
    return 0;
}
