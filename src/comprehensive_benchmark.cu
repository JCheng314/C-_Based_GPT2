#include "../include/transformer_kernels.h"
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <mpi.h>

// Forward declarations from CPU reference
extern void cpu_gemm(const float* A, const float* W, float* C, int M, int K, int N);
extern void cpu_layernorm(const float* x, const float* gamma, const float* beta, float* out, int B, int T, int C, float eps);
extern void cpu_attention(const float* Q, const float* K, const float* V, float* out, int B, int NH, int T, int HS);

// CPU Fused Residual+LN reference
void cpu_residual_layernorm(const float* x, const float* residual, const float* gamma, const float* beta, float* out, int B, int T, int C, float eps) {
    std::vector<float> z(B * T * C);
    for (int i = 0; i < B * T * C; ++i) {
        z[i] = x[i] + residual[i];
    }
    cpu_layernorm(z.data(), gamma, beta, out, B, T, C, eps);
}

// Matrix Initialization
void init_matrix(std::vector<float>& mat) {
    for (size_t i = 0; i < mat.size(); ++i) {
        mat[i] = static_cast<float>(rand()) / RAND_MAX - 0.5f; // Zero-mean initialization
    }
}

// Error statistics helper
void get_error_stats(const std::vector<float>& cpu, const std::vector<float>& gpu, float& max_abs, float& max_rel) {
    max_abs = 0.0f;
    max_rel = 0.0f;
    for (size_t i = 0; i < cpu.size(); ++i) {
        float abs_err = std::abs(cpu[i] - gpu[i]);
        max_abs = std::max(max_abs, abs_err);
        float denom = std::abs(cpu[i]);
        if (denom > 1e-6f) {
            max_rel = std::max(max_rel, abs_err / denom);
        } else {
            max_rel = std::max(max_rel, abs_err);
        }
    }
}

int main(int argc, char** argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int iters = 50;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Only run standard sweeps on rank 0
    if (rank == 0) {
        std::cout << "=== CME 213 Comprehensive CUDA Benchmark (Rank 0) ===\n";

        // Create results directory
        int ret = system("mkdir -p results");
        (void)ret;

        // ==========================================================
        // 1. LayerNorm Sweep (Req 7)
        // ==========================================================
        {
            std::cout << "Running LayerNorm Sweep...\n";
            std::ofstream f_ln("results/req7_layernorm.csv");
            f_ln << "variant,C,time_ms,bandwidth_GBs,correct\n";

            int B = 8;
            int T = 512;
            float eps = 1e-5f;
            std::vector<int> C_sizes = {64, 128, 256, 512, 768, 1024};

            for (int C : C_sizes) {
                int N_elems = B * T * C;
                std::vector<float> h_x(N_elems), h_gamma(C), h_beta(C);
                std::vector<float> h_out_cpu(N_elems, 0.0f), h_out_base(N_elems, 0.0f), h_out_f4(N_elems, 0.0f);
                init_matrix(h_x); init_matrix(h_gamma); init_matrix(h_beta);

                cpu_layernorm(h_x.data(), h_gamma.data(), h_beta.data(), h_out_cpu.data(), B, T, C, eps);

                float *d_x, *d_gamma, *d_beta, *d_out;
                CHECK_CUDA(cudaMalloc(&d_x, N_elems * sizeof(float)));
                CHECK_CUDA(cudaMalloc(&d_gamma, C * sizeof(float)));
                CHECK_CUDA(cudaMalloc(&d_beta, C * sizeof(float)));
                CHECK_CUDA(cudaMalloc(&d_out, N_elems * sizeof(float)));

                CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), N_elems * sizeof(float), cudaMemcpyHostToDevice));
                CHECK_CUDA(cudaMemcpy(d_gamma, h_gamma.data(), C * sizeof(float), cudaMemcpyHostToDevice));
                CHECK_CUDA(cudaMemcpy(d_beta, h_beta.data(), C * sizeof(float), cudaMemcpyHostToDevice));

                int block_size = 256;
                int grid_size = B * T;
                size_t shared_mem = 2 * block_size * sizeof(float);

                // Baseline
                layernorm_forward_kernel<<<grid_size, block_size, shared_mem>>>(d_x, d_gamma, d_beta, d_out, B, T, C, eps);
                CHECK_CUDA(cudaDeviceSynchronize());
                CHECK_CUDA(cudaMemcpy(h_out_base.data(), d_out, N_elems * sizeof(float), cudaMemcpyDeviceToHost));
                float max_abs = 0.0f, max_rel = 0.0f;
                get_error_stats(h_out_cpu, h_out_base, max_abs, max_rel);
                bool correct_base = (max_abs < 1e-4f);

                cudaEventRecord(start);
                for (int i = 0; i < iters; ++i) {
                    layernorm_forward_kernel<<<grid_size, block_size, shared_mem>>>(d_x, d_gamma, d_beta, d_out, B, T, C, eps);
                }
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);
                float ms_base; cudaEventElapsedTime(&ms_base, start, stop); ms_base /= iters;
                double bw_base = (2.0 * N_elems * sizeof(float)) / (ms_base * 1e-3) / 1e9;

                f_ln << "baseline," << C << "," << ms_base << "," << bw_base << "," << (correct_base ? "PASS" : "FAIL") << "\n";

                // float4 (Only if C is multiple of 4)
                if (C % 4 == 0) {
                    layernorm_forward_kernel_float4<<<grid_size, block_size, shared_mem>>>(d_x, d_gamma, d_beta, d_out, B, T, C, eps);
                    CHECK_CUDA(cudaDeviceSynchronize());
                    CHECK_CUDA(cudaMemcpy(h_out_f4.data(), d_out, N_elems * sizeof(float), cudaMemcpyDeviceToHost));
                    get_error_stats(h_out_cpu, h_out_f4, max_abs, max_rel);
                    bool correct_f4 = (max_abs < 1e-4f);

                    cudaEventRecord(start);
                    for (int i = 0; i < iters; ++i) {
                        layernorm_forward_kernel_float4<<<grid_size, block_size, shared_mem>>>(d_x, d_gamma, d_beta, d_out, B, T, C, eps);
                    }
                    cudaEventRecord(stop);
                    cudaEventSynchronize(stop);
                    float ms_f4; cudaEventElapsedTime(&ms_f4, start, stop); ms_f4 /= iters;
                    double bw_f4 = (2.0 * N_elems * sizeof(float)) / (ms_f4 * 1e-3) / 1e9;

                    f_ln << "float4," << C << "," << ms_f4 << "," << bw_f4 << "," << (correct_f4 ? "PASS" : "FAIL") << "\n";
                }

                cudaFree(d_x); cudaFree(d_gamma); cudaFree(d_beta); cudaFree(d_out);
            }
            f_ln.close();
        }

        // ==========================================================
        // 2. Attention Sweep (Req 7)
        // ==========================================================
        {
            std::cout << "Running Attention Sweep...\n";
            std::ofstream f_attn("results/req7_attention.csv");
            f_attn << "variant,T,time_ms,gflops,mem_MB,correct\n";

            int B = 2;
            int NH = 4;
            int HS = 64;
            std::vector<int> T_sizes = {64, 128, 256, 512, 1024};

            for (int T : T_sizes) {
                int size_qkv = B * NH * T * HS;
                int size_scores = B * NH * T * T;
                std::vector<float> h_Q(size_qkv), h_K(size_qkv), h_V(size_qkv);
                std::vector<float> h_out_cpu(size_qkv, 0.0f), h_out_unfused(size_qkv, 0.0f), h_out_flash(size_qkv, 0.0f);
                init_matrix(h_Q); init_matrix(h_K); init_matrix(h_V);

                cpu_attention(h_Q.data(), h_K.data(), h_V.data(), h_out_cpu.data(), B, NH, T, HS);

                float *d_Q, *d_K, *d_V, *d_out, *d_scores, *d_probs;
                CHECK_CUDA(cudaMalloc(&d_Q, size_qkv * sizeof(float)));
                CHECK_CUDA(cudaMalloc(&d_K, size_qkv * sizeof(float)));
                CHECK_CUDA(cudaMalloc(&d_V, size_qkv * sizeof(float)));
                CHECK_CUDA(cudaMalloc(&d_out, size_qkv * sizeof(float)));
                CHECK_CUDA(cudaMalloc(&d_scores, size_scores * sizeof(float)));
                CHECK_CUDA(cudaMalloc(&d_probs, size_scores * sizeof(float)));

                CHECK_CUDA(cudaMemcpy(d_Q, h_Q.data(), size_qkv * sizeof(float), cudaMemcpyHostToDevice));
                CHECK_CUDA(cudaMemcpy(d_K, h_K.data(), size_qkv * sizeof(float), cudaMemcpyHostToDevice));
                CHECK_CUDA(cudaMemcpy(d_V, h_V.data(), size_qkv * sizeof(float), cudaMemcpyHostToDevice));

                // Launch configs for unfused
                dim3 block2d(16, 16);
                dim3 grid_scores((T + block2d.x - 1) / block2d.x, (T + block2d.y - 1) / block2d.y, B * NH);
                dim3 grid_val((HS + block2d.x - 1) / block2d.x, (T + block2d.y - 1) / block2d.y, B * NH);
                int threads = 256;
                size_t softmax_smem = 2 * threads * sizeof(float);

                // Unfused warm-up & verify
                attention_scores_kernel<<<grid_scores, block2d>>>(d_Q, d_K, d_scores, B, NH, T, HS);
                causal_mask_kernel<<<grid_scores, block2d>>>(d_scores, B, NH, T);
                masked_softmax_kernel<<<B * NH * T, threads, softmax_smem>>>(d_scores, d_probs, B, NH, T);
                attention_value_kernel<<<grid_val, block2d>>>(d_probs, d_V, d_out, B, NH, T, HS);
                CHECK_CUDA(cudaDeviceSynchronize());
                CHECK_CUDA(cudaMemcpy(h_out_unfused.data(), d_out, size_qkv * sizeof(float), cudaMemcpyDeviceToHost));
                float max_abs = 0.0f, max_rel = 0.0f;
                get_error_stats(h_out_cpu, h_out_unfused, max_abs, max_rel);
                bool correct_unfused = (max_abs < 1e-3f);

                cudaEventRecord(start);
                for (int i = 0; i < iters; ++i) {
                    attention_scores_kernel<<<grid_scores, block2d>>>(d_Q, d_K, d_scores, B, NH, T, HS);
                    causal_mask_kernel<<<grid_scores, block2d>>>(d_scores, B, NH, T);
                    masked_softmax_kernel<<<B * NH * T, threads, softmax_smem>>>(d_scores, d_probs, B, NH, T);
                    attention_value_kernel<<<grid_val, block2d>>>(d_probs, d_V, d_out, B, NH, T, HS);
                }
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);
                float ms_unfused; cudaEventElapsedTime(&ms_unfused, start, stop); ms_unfused /= iters;
                double gflops_unfused = (2.0 * B * NH * T * T * HS) / (ms_unfused * 1e-3) / 1e9;
                double mem_unfused = (double)size_scores * sizeof(float) / 1e6; // T^2 activation footprint in MB

                f_attn << "unfused," << T << "," << ms_unfused << "," << gflops_unfused << "," << mem_unfused << "," << (correct_unfused ? "PASS" : "FAIL") << "\n";

                // FlashAttention
                dim3 block_flash(HS);
                dim3 grid_flash(T, NH, B);
                int Bc = 32;
                size_t shared_mem_flash = HS * sizeof(float) + 2 * Bc * HS * sizeof(float) + Bc * sizeof(float);

                tiled_flash_attention_kernel<<<grid_flash, block_flash, shared_mem_flash>>>(d_Q, d_K, d_V, d_out, B, NH, T, HS);
                CHECK_CUDA(cudaDeviceSynchronize());
                CHECK_CUDA(cudaMemcpy(h_out_flash.data(), d_out, size_qkv * sizeof(float), cudaMemcpyDeviceToHost));
                get_error_stats(h_out_cpu, h_out_flash, max_abs, max_rel);
                bool correct_flash = (max_abs < 1e-2f); // Slightly higher tolerance for tile accumulator sum

                cudaEventRecord(start);
                for (int i = 0; i < iters; ++i) {
                    tiled_flash_attention_kernel<<<grid_flash, block_flash, shared_mem_flash>>>(d_Q, d_K, d_V, d_out, B, NH, T, HS);
                }
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);
                float ms_flash; cudaEventElapsedTime(&ms_flash, start, stop); ms_flash /= iters;
                double gflops_flash = (2.0 * B * NH * T * T * HS) / (ms_flash * 1e-3) / 1e9;
                double mem_flash = (double)(HS + 2 * Bc * HS + Bc) * sizeof(float) / 1e6; // O(1) SRAM footprint

                f_attn << "flash," << T << "," << ms_flash << "," << gflops_flash << "," << mem_flash << "," << (correct_flash ? "PASS" : "FAIL") << "\n";

                cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_out); cudaFree(d_scores); cudaFree(d_probs);
            }
            f_attn.close();
        }

        // ==========================================================
        // 3. Compute-only Scaling (Req 8 strategy)
        // ==========================================================
        {
            std::cout << "Running Compute-only Scaling Sweep...\n";
            std::ofstream f_scale("results/req8_compute_scaling.csv");
            f_scale << "B_per_rank,time_ms\n";

            int T = 512;
            int C = 768;
            int NH = 12;
            int HS = 64;

            std::vector<int> batches = {1, 2, 4, 8};
            for (int B : batches) {
                int N_elems = B * T * C;
                int size_qkv = B * NH * T * HS;

                float *d_x, *d_gamma, *d_beta, *d_ln_out;
                float *d_qkv_w, *d_qkv_out;
                float *d_Q, *d_K, *d_V, *d_attn_out, *d_attn_merge;
                float *d_attn_proj_w, *d_proj_out, *d_h1;

                CHECK_CUDA(cudaMalloc(&d_x, N_elems * sizeof(float)));
                CHECK_CUDA(cudaMalloc(&d_gamma, C * sizeof(float)));
                CHECK_CUDA(cudaMalloc(&d_beta, C * sizeof(float)));
                CHECK_CUDA(cudaMalloc(&d_ln_out, N_elems * sizeof(float)));
                CHECK_CUDA(cudaMalloc(&d_qkv_w, C * 3 * C * sizeof(float)));
                CHECK_CUDA(cudaMalloc(&d_qkv_out, N_elems * 3 * sizeof(float)));
                CHECK_CUDA(cudaMalloc(&d_Q, size_qkv * sizeof(float)));
                CHECK_CUDA(cudaMalloc(&d_K, size_qkv * sizeof(float)));
                CHECK_CUDA(cudaMalloc(&d_V, size_qkv * sizeof(float)));
                CHECK_CUDA(cudaMalloc(&d_attn_out, size_qkv * sizeof(float)));
                CHECK_CUDA(cudaMalloc(&d_attn_merge, N_elems * sizeof(float)));
                CHECK_CUDA(cudaMalloc(&d_attn_proj_w, C * C * sizeof(float)));
                CHECK_CUDA(cudaMalloc(&d_proj_out, N_elems * sizeof(float)));
                CHECK_CUDA(cudaMalloc(&d_h1, N_elems * sizeof(float)));

                dim3 block_gemm(THREADS_X, THREADS_Y);
                dim3 grid_qkv((3 * C + BLOCK_N - 1) / BLOCK_N, (B * T + BLOCK_M - 1) / BLOCK_M);
                dim3 grid_proj((C + BLOCK_N - 1) / BLOCK_N, (B * T + BLOCK_M - 1) / BLOCK_M);

                dim3 block_flash(HS);
                dim3 grid_flash(T, NH, B);
                int Bc = 32;
                size_t shared_mem_flash = HS * sizeof(float) + 2 * Bc * HS * sizeof(float) + Bc * sizeof(float);

                // Warm-up
                layernorm_forward_kernel<<<B * T, 256, 2 * 256 * sizeof(float)>>>(d_x, d_gamma, d_beta, d_ln_out, B, T, C, 1e-5f);
                hierarchical_gemm_2x4_kernel<<<grid_qkv, block_gemm>>>(d_ln_out, d_qkv_w, d_qkv_out, B * T, C, 3 * C);
                split_qkv_kernel<<<((N_elems * 3 + 255) / 256), 256>>>(d_qkv_out, d_Q, d_K, d_V, B, T, NH, HS);
                tiled_flash_attention_kernel<<<grid_flash, block_flash, shared_mem_flash>>>(d_Q, d_K, d_V, d_attn_out, B, NH, T, HS);
                merge_heads_kernel<<<((N_elems + 255) / 256), 256>>>(d_attn_out, d_attn_merge, B, T, NH, HS);
                hierarchical_gemm_2x4_kernel<<<grid_proj, block_gemm>>>(d_attn_merge, d_attn_proj_w, d_proj_out, B * T, C, C);
                residual_add_kernel<<<((N_elems + 255) / 256), 256>>>(d_proj_out, d_x, d_h1, N_elems);
                CHECK_CUDA(cudaDeviceSynchronize());

                cudaEventRecord(start);
                for (int i = 0; i < iters; ++i) {
                    layernorm_forward_kernel<<<B * T, 256, 2 * 256 * sizeof(float)>>>(d_x, d_gamma, d_beta, d_ln_out, B, T, C, 1e-5f);
                    hierarchical_gemm_2x4_kernel<<<grid_qkv, block_gemm>>>(d_ln_out, d_qkv_w, d_qkv_out, B * T, C, 3 * C);
                    split_qkv_kernel<<<((N_elems * 3 + 255) / 256), 256>>>(d_qkv_out, d_Q, d_K, d_V, B, T, NH, HS);
                    tiled_flash_attention_kernel<<<grid_flash, block_flash, shared_mem_flash>>>(d_Q, d_K, d_V, d_attn_out, B, NH, T, HS);
                    merge_heads_kernel<<<((N_elems + 255) / 256), 256>>>(d_attn_out, d_attn_merge, B, T, NH, HS);
                    hierarchical_gemm_2x4_kernel<<<grid_proj, block_gemm>>>(d_attn_merge, d_attn_proj_w, d_proj_out, B * T, C, C);
                    residual_add_kernel<<<((N_elems + 255) / 256), 256>>>(d_proj_out, d_x, d_h1, N_elems);
                }
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);
                float ms; cudaEventElapsedTime(&ms, start, stop); ms /= iters;

                f_scale << B << "," << ms << "\n";

                cudaFree(d_x); cudaFree(d_gamma); cudaFree(d_beta); cudaFree(d_ln_out);
                cudaFree(d_qkv_w); cudaFree(d_qkv_out); cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
                cudaFree(d_attn_out); cudaFree(d_attn_merge); cudaFree(d_attn_proj_w); cudaFree(d_proj_out);
                cudaFree(d_h1);
            }
            f_scale.close();
        }

        // ==========================================================
        // 4. Correctness Sweep & Verification (Req 9)
        // ==========================================================
        {
            std::cout << "Running Correctness Verification Sweep...\n";
            std::ofstream f_corr("results/req9_correctness.csv");
            f_corr << "kernel,M_or_BT,K_or_C,N_or_T,max_abs_err,max_rel_err,status\n";

            // GEMM
            struct GemmConfig { int M, K, N; };
            std::vector<GemmConfig> gemm_configs = {
                {64,64,64}, {128,256,128}, {512,512,512}, {1024,256,1024}
            };
            for (auto cfg : gemm_configs) {
                std::vector<float> h_A(cfg.M * cfg.K), h_W(cfg.K * cfg.N), h_C_cpu(cfg.M * cfg.N, 0.0f), h_C_gpu(cfg.M * cfg.N, 0.0f);
                init_matrix(h_A); init_matrix(h_W);
                cpu_gemm(h_A.data(), h_W.data(), h_C_cpu.data(), cfg.M, cfg.K, cfg.N);

                float *d_A, *d_W, *d_C;
                CHECK_CUDA(cudaMalloc(&d_A, cfg.M * cfg.K * sizeof(float)));
                CHECK_CUDA(cudaMalloc(&d_W, cfg.K * cfg.N * sizeof(float)));
                CHECK_CUDA(cudaMalloc(&d_C, cfg.M * cfg.N * sizeof(float)));
                CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), cfg.M * cfg.K * sizeof(float), cudaMemcpyHostToDevice));
                CHECK_CUDA(cudaMemcpy(d_W, h_W.data(), cfg.K * cfg.N * sizeof(float), cudaMemcpyHostToDevice));

                dim3 block(THREADS_X, THREADS_Y);
                dim3 grid((cfg.N + BLOCK_N - 1) / BLOCK_N, (cfg.M + BLOCK_M - 1) / BLOCK_M);
                hierarchical_gemm_2x4_kernel<<<grid, block>>>(d_A, d_W, d_C, cfg.M, cfg.K, cfg.N);
                CHECK_CUDA(cudaDeviceSynchronize());
                CHECK_CUDA(cudaMemcpy(h_C_gpu.data(), d_C, cfg.M * cfg.N * sizeof(float), cudaMemcpyDeviceToHost));

                float max_abs = 0.0f, max_rel = 0.0f;
                get_error_stats(h_C_cpu, h_C_gpu, max_abs, max_rel);
                // Tiled GEMM vs CPU naive: FP reorder acceptable up to 1% relative error
                bool pass = (max_rel < 1e-2f);

                f_corr << "GEMM," << cfg.M << "," << cfg.K << "," << cfg.N << "," << max_abs << "," << max_rel << "," << (pass ? "PASS" : "FAIL") << "\n";
                cudaFree(d_A); cudaFree(d_W); cudaFree(d_C);
            }

            // LayerNorm (float32 & float4)
            struct LnConfig { int BT, C; };
            std::vector<LnConfig> ln_configs = {
                {8, 64}, {128, 256}, {1024, 768}
            };
            for (auto cfg : ln_configs) {
                float eps = 1e-5f;
                std::vector<float> h_x(cfg.BT * cfg.C), h_gamma(cfg.C), h_beta(cfg.C), h_out_cpu(cfg.BT * cfg.C, 0.0f), h_out_gpu(cfg.BT * cfg.C, 0.0f);
                init_matrix(h_x); init_matrix(h_gamma); init_matrix(h_beta);
                cpu_layernorm(h_x.data(), h_gamma.data(), h_beta.data(), h_out_cpu.data(), 1, cfg.BT, cfg.C, eps);

                float *d_x, *d_gamma, *d_beta, *d_out;
                CHECK_CUDA(cudaMalloc(&d_x, cfg.BT * cfg.C * sizeof(float)));
                CHECK_CUDA(cudaMalloc(&d_gamma, cfg.C * sizeof(float)));
                CHECK_CUDA(cudaMalloc(&d_beta, cfg.C * sizeof(float)));
                CHECK_CUDA(cudaMalloc(&d_out, cfg.BT * cfg.C * sizeof(float)));
                CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), cfg.BT * cfg.C * sizeof(float), cudaMemcpyHostToDevice));
                CHECK_CUDA(cudaMemcpy(d_gamma, h_gamma.data(), cfg.C * sizeof(float), cudaMemcpyHostToDevice));
                CHECK_CUDA(cudaMemcpy(d_beta, h_beta.data(), cfg.C * sizeof(float), cudaMemcpyHostToDevice));

                int threads = 256;
                size_t smem = 2 * threads * sizeof(float);

                // Baseline
                layernorm_forward_kernel<<<cfg.BT, threads, smem>>>(d_x, d_gamma, d_beta, d_out, 1, cfg.BT, cfg.C, eps);
                CHECK_CUDA(cudaDeviceSynchronize());
                CHECK_CUDA(cudaMemcpy(h_out_gpu.data(), d_out, cfg.BT * cfg.C * sizeof(float), cudaMemcpyDeviceToHost));
                float max_abs = 0.0f, max_rel = 0.0f;
                get_error_stats(h_out_cpu, h_out_gpu, max_abs, max_rel);
                bool pass = (max_abs < 1e-4f);
                f_corr << "LayerNorm_base," << cfg.BT << "," << cfg.C << ",0," << max_abs << "," << max_rel << "," << (pass ? "PASS" : "FAIL") << "\n";

                // float4
                if (cfg.C % 4 == 0) {
                    layernorm_forward_kernel_float4<<<cfg.BT, threads, smem>>>(d_x, d_gamma, d_beta, d_out, 1, cfg.BT, cfg.C, eps);
                    CHECK_CUDA(cudaDeviceSynchronize());
                    CHECK_CUDA(cudaMemcpy(h_out_gpu.data(), d_out, cfg.BT * cfg.C * sizeof(float), cudaMemcpyDeviceToHost));
                    get_error_stats(h_out_cpu, h_out_gpu, max_abs, max_rel);
                    pass = (max_abs < 1e-4f);
                    f_corr << "LayerNorm_float4," << cfg.BT << "," << cfg.C << ",0," << max_abs << "," << max_rel << "," << (pass ? "PASS" : "FAIL") << "\n";
                }

                cudaFree(d_x); cudaFree(d_gamma); cudaFree(d_beta); cudaFree(d_out);
            }

            // Attention
            struct AttnConfig { int T, HS; };
            std::vector<AttnConfig> attn_configs = {
                {64, 64}, {128, 64}, {256, 64}
            };
            int B = 2;
            int NH = 4;
            for (auto cfg : attn_configs) {
                int size_qkv = B * NH * cfg.T * cfg.HS;
                std::vector<float> h_Q(size_qkv), h_K(size_qkv), h_V(size_qkv), h_out_cpu(size_qkv, 0.0f), h_out_gpu(size_qkv, 0.0f);
                init_matrix(h_Q); init_matrix(h_K); init_matrix(h_V);
                cpu_attention(h_Q.data(), h_K.data(), h_V.data(), h_out_cpu.data(), B, NH, cfg.T, cfg.HS);

                float *d_Q, *d_K, *d_V, *d_out;
                CHECK_CUDA(cudaMalloc(&d_Q, size_qkv * sizeof(float)));
                CHECK_CUDA(cudaMalloc(&d_K, size_qkv * sizeof(float)));
                CHECK_CUDA(cudaMalloc(&d_V, size_qkv * sizeof(float)));
                CHECK_CUDA(cudaMalloc(&d_out, size_qkv * sizeof(float)));
                CHECK_CUDA(cudaMemcpy(d_Q, h_Q.data(), size_qkv * sizeof(float), cudaMemcpyHostToDevice));
                CHECK_CUDA(cudaMemcpy(d_K, h_K.data(), size_qkv * sizeof(float), cudaMemcpyHostToDevice));
                CHECK_CUDA(cudaMemcpy(d_V, h_V.data(), size_qkv * sizeof(float), cudaMemcpyHostToDevice));

                dim3 block(cfg.HS);
                dim3 grid(cfg.T, NH, B);
                int Bc = 32;
                size_t shared_mem = cfg.HS * sizeof(float) + 2 * Bc * cfg.HS * sizeof(float) + Bc * sizeof(float);

                tiled_flash_attention_kernel<<<grid, block, shared_mem>>>(d_Q, d_K, d_V, d_out, B, NH, cfg.T, cfg.HS);
                CHECK_CUDA(cudaDeviceSynchronize());
                CHECK_CUDA(cudaMemcpy(h_out_gpu.data(), d_out, size_qkv * sizeof(float), cudaMemcpyDeviceToHost));

                float max_abs = 0.0f, max_rel = 0.0f;
                get_error_stats(h_out_cpu, h_out_gpu, max_abs, max_rel);
                bool pass = (max_abs < 1e-2f);

                f_corr << "FlashAttention," << B * NH * cfg.T << "," << cfg.HS << "," << cfg.T << "," << max_abs << "," << max_rel << "," << (pass ? "PASS" : "FAIL") << "\n";
                cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_out);
            }

            // Fused Residual+LN
            for (auto cfg : ln_configs) {
                float eps = 1e-5f;
                std::vector<float> h_x(cfg.BT * cfg.C), h_res(cfg.BT * cfg.C), h_gamma(cfg.C), h_beta(cfg.C), h_out_cpu(cfg.BT * cfg.C, 0.0f), h_out_gpu(cfg.BT * cfg.C, 0.0f);
                init_matrix(h_x); init_matrix(h_res); init_matrix(h_gamma); init_matrix(h_beta);
                cpu_residual_layernorm(h_x.data(), h_res.data(), h_gamma.data(), h_beta.data(), h_out_cpu.data(), 1, cfg.BT, cfg.C, eps);

                float *d_x, *d_res, *d_gamma, *d_beta, *d_out;
                CHECK_CUDA(cudaMalloc(&d_x, cfg.BT * cfg.C * sizeof(float)));
                CHECK_CUDA(cudaMalloc(&d_res, cfg.BT * cfg.C * sizeof(float)));
                CHECK_CUDA(cudaMalloc(&d_gamma, cfg.C * sizeof(float)));
                CHECK_CUDA(cudaMalloc(&d_beta, cfg.C * sizeof(float)));
                CHECK_CUDA(cudaMalloc(&d_out, cfg.BT * cfg.C * sizeof(float)));

                CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), cfg.BT * cfg.C * sizeof(float), cudaMemcpyHostToDevice));
                CHECK_CUDA(cudaMemcpy(d_res, h_res.data(), cfg.BT * cfg.C * sizeof(float), cudaMemcpyHostToDevice));
                CHECK_CUDA(cudaMemcpy(d_gamma, h_gamma.data(), cfg.C * sizeof(float), cudaMemcpyHostToDevice));
                CHECK_CUDA(cudaMemcpy(d_beta, h_beta.data(), cfg.C * sizeof(float), cudaMemcpyHostToDevice));

                int threads = 256;
                size_t smem = 2 * threads * sizeof(float);

                residual_layernorm_kernel<<<cfg.BT, threads, smem>>>(d_x, d_res, d_gamma, d_beta, d_out, 1, cfg.BT, cfg.C, eps);
                CHECK_CUDA(cudaDeviceSynchronize());
                CHECK_CUDA(cudaMemcpy(h_out_gpu.data(), d_out, cfg.BT * cfg.C * sizeof(float), cudaMemcpyDeviceToHost));

                float max_abs = 0.0f, max_rel = 0.0f;
                get_error_stats(h_out_cpu, h_out_gpu, max_abs, max_rel);
                bool pass = (max_abs < 1e-4f);

                f_corr << "FusedResidualLN," << cfg.BT << "," << cfg.C << ",0," << max_abs << "," << max_rel << "," << (pass ? "PASS" : "FAIL") << "\n";
                cudaFree(d_x); cudaFree(d_res); cudaFree(d_gamma); cudaFree(d_beta); cudaFree(d_out);
            }

            f_corr.close();
        }
    }

    // ==========================================================
    // 5. MPI AllReduce Microbenchmark (Req 8 strategy)
    // ==========================================================
    {
        // Vector of sizes in float count
        std::vector<int> buffer_sizes = {65536, 262144, 1048576, 4194304, 16777216};
        
        std::ofstream f_ar;
        if (rank == 0) {
            // Check if file is empty to write header
            std::ifstream test("results/req8_allreduce.csv");
            bool is_empty = (test.peek() == std::ifstream::traits_type::eof());
            test.close();

            f_ar.open("results/req8_allreduce.csv", std::ios::app);
            if (is_empty) {
                f_ar << "ranks,buffer_floats,latency_ms,bandwidth_GBs\n";
            }
        }

        for (int count : buffer_sizes) {
            float* d_buf = nullptr;
            CHECK_CUDA(cudaMalloc(&d_buf, count * sizeof(float)));
            
            // Initialize buffer on device
            std::vector<float> h_buf(count, 1.0f);
            CHECK_CUDA(cudaMemcpy(d_buf, h_buf.data(), count * sizeof(float), cudaMemcpyHostToDevice));
            
            // Warm-up AllReduce
            for (int i = 0; i < 5; ++i) {
                MPI_Allreduce(MPI_IN_PLACE, d_buf, count, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
            }
            MPI_Barrier(MPI_COMM_WORLD);

            double t_start = MPI_Wtime();
            for (int i = 0; i < iters; ++i) {
                MPI_Allreduce(MPI_IN_PLACE, d_buf, count, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
            }
            MPI_Barrier(MPI_COMM_WORLD);
            double t_end = MPI_Wtime();
            
            double latency_seconds = (t_end - t_start) / iters;
            double latency_ms = latency_seconds * 1000.0;
            
            // Effective bandwidth (GB/s): size read + size written per rank
            // For AllReduce, typical bandwidth formula is 2 * (size_bytes * (world_size - 1)/world_size) / time
            // Let's use 2 * size_bytes / time as standard effective communication bandwidth
            double size_bytes = count * sizeof(float);
            double bandwidth_GBs = (2.0 * size_bytes) / (latency_seconds * 1e9);

            if (rank == 0) {
                f_ar << world_size << "," << count << "," << latency_ms << "," << bandwidth_GBs << "\n";
                std::cout << "AllReduce Sweep: ranks=" << world_size 
                          << ", count=" << count 
                          << ", latency=" << latency_ms << " ms"
                          << ", bandwidth=" << bandwidth_GBs << " GB/s\n";
            }
            cudaFree(d_buf);
        }
        if (rank == 0) {
            f_ar.close();
        }
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    MPI_Finalize();
    return 0;
}
