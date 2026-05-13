#include "../include/transformer_kernels.h"

#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <string>

// ============================================================
// Config
// ============================================================

static const int B = 4;
static const int T = 64;
static const int C = 64;

// GPT-2 tokenizer vocab size.
// If your tokenizer uses another vocab size, change this.
static const int V = 50257;

static const int M = B * T;

static const float LR = 3e-4f;
static const float BETA1 = 0.9f;
static const float BETA2 = 0.999f;
static const float ADAM_EPS = 1e-8f;
static const float WEIGHT_DECAY = 0.01f;

// ============================================================
// Utility
// ============================================================

static void check_cuda(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line
                  << " : " << cudaGetErrorString(err) << std::endl;
        std::exit(1);
    }
}

#define CUDA_CHECK(call) check_cuda((call), __FILE__, __LINE__)

std::vector<uint16_t> load_tokens_u16(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);

    if (!file.is_open()) {
        std::cerr << "Could not open file: " << filename << std::endl;
        std::exit(1);
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    if (size % sizeof(uint16_t) != 0) {
        std::cerr << "File size is not divisible by uint16_t. "
                  << "If your file uses int32 tokens, change the loader."
                  << std::endl;
        std::exit(1);
    }

    std::vector<uint16_t> tokens(size / sizeof(uint16_t));

    if (!file.read(reinterpret_cast<char*>(tokens.data()), size)) {
        std::cerr << "Failed to read file: " << filename << std::endl;
        std::exit(1);
    }

    std::cout << "Loaded " << tokens.size()
              << " tokens from " << filename << std::endl;

    return tokens;
}

void get_batch(
    const std::vector<uint16_t>& data,
    std::vector<int>& x,
    std::vector<int>& y,
    std::mt19937& rng
) {
    std::uniform_int_distribution<int> dist(0, static_cast<int>(data.size()) - T - 2);

    for (int b = 0; b < B; ++b) {
        int start = dist(rng);

        for (int t = 0; t < T; ++t) {
            int idx = b * T + t;

            x[idx] = static_cast<int>(data[start + t]);
            y[idx] = static_cast<int>(data[start + t + 1]);

            if (x[idx] < 0 || x[idx] >= V || y[idx] < 0 || y[idx] >= V) {
                std::cerr << "Token id out of vocab range. "
                          << "x=" << x[idx] << ", y=" << y[idx]
                          << ", V=" << V << std::endl;
                std::exit(1);
            }
        }
    }
}

void init_normal(std::vector<float>& x, float stddev) {
    std::mt19937 rng(1234);
    std::normal_distribution<float> dist(0.0f, stddev);

    for (float& v : x) {
        v = dist(rng);
    }
}

float average_loss_from_device(float* d_losses) {
    std::vector<float> h_losses(M);

    CUDA_CHECK(cudaMemcpy(
        h_losses.data(),
        d_losses,
        M * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    float sum = 0.0f;

    for (float v : h_losses) {
        sum += v;
    }

    return sum / static_cast<float>(M);
}

// ============================================================
// Forward-only validation
// ============================================================

float estimate_loss(
    const std::vector<uint16_t>& tokens,
    int* d_x,
    int* d_y,
    float* d_token_emb,
    float* d_pos_emb,
    float* d_lm_w,
    float* d_hidden,
    float* d_logits,
    float* d_losses,
    int eval_iters,
    std::mt19937& rng
) {
    std::vector<int> h_x(M);
    std::vector<int> h_y(M);

    float total_loss = 0.0f;

    int threads = 256;

    dim3 gemm_block(THREADS_X, THREADS_Y);
    dim3 gemm_grid(
        (V + BLOCK_N - 1) / BLOCK_N,
        (M + BLOCK_M - 1) / BLOCK_M
    );

    for (int iter = 0; iter < eval_iters; ++iter) {
        get_batch(tokens, h_x, h_y, rng);

        CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), M * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_y, h_y.data(), M * sizeof(int), cudaMemcpyHostToDevice));

        embedding_forward_kernel<<<(M * C + threads - 1) / threads, threads>>>(
            d_x,
            d_token_emb,
            d_pos_emb,
            d_hidden,
            B,
            T,
            C
        );

        CUDA_CHECK(cudaGetLastError());

        hierarchical_gemm_2x4_kernel<<<gemm_grid, gemm_block>>>(
            d_hidden,
            d_lm_w,
            d_logits,
            M,
            C,
            V
        );

        CUDA_CHECK(cudaGetLastError());

        cross_entropy_forward_kernel<<<(M + threads - 1) / threads, threads>>>(
            d_logits,
            d_y,
            d_losses,
            M,
            V
        );

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        total_loss += average_loss_from_device(d_losses);
    }

    return total_loss / static_cast<float>(eval_iters);
}

// ============================================================
// Main training
// ============================================================

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: ./train_step1 train.bin valid.bin" << std::endl;
        return 1;
    }

    std::string train_path = argv[1];
    std::string valid_path = argv[2];

    std::vector<uint16_t> train_tokens = load_tokens_u16(train_path);
    std::vector<uint16_t> valid_tokens = load_tokens_u16(valid_path);

    if (train_tokens.size() < T + 2 || valid_tokens.size() < T + 2) {
        std::cerr << "Dataset too small for T=" << T << std::endl;
        return 1;
    }

    std::mt19937 train_rng(42);
    std::mt19937 valid_rng(123);

    // ----------------------------
    // Host batch buffers
    // ----------------------------

    std::vector<int> h_x(M);
    std::vector<int> h_y(M);

    // ----------------------------
    // Parameter sizes
    // ----------------------------

    int token_emb_size = V * C;
    int pos_emb_size = T * C;
    int lm_w_size = C * V;

    int hidden_size = M * C;
    int logits_size = M * V;

    std::cout << "Model config:" << std::endl;
    std::cout << "  B=" << B << ", T=" << T << ", C=" << C << ", V=" << V << std::endl;
    std::cout << "  logits memory = "
              << logits_size * sizeof(float) / 1024.0f / 1024.0f
              << " MB" << std::endl;

    // ----------------------------
    // CPU initialization
    // ----------------------------

    std::vector<float> h_token_emb(token_emb_size);
    std::vector<float> h_pos_emb(pos_emb_size);
    std::vector<float> h_lm_w(lm_w_size);

    init_normal(h_token_emb, 0.02f);
    init_normal(h_pos_emb, 0.01f);
    init_normal(h_lm_w, 0.02f);

    // ----------------------------
    // Device pointers
    // ----------------------------

    int* d_x = nullptr;
    int* d_y = nullptr;

    float* d_token_emb = nullptr;
    float* d_pos_emb = nullptr;
    float* d_lm_w = nullptr;

    float* d_token_emb_grad = nullptr;
    float* d_pos_emb_grad = nullptr;
    float* d_lm_w_grad = nullptr;

    float* d_token_emb_m = nullptr;
    float* d_pos_emb_m = nullptr;
    float* d_lm_w_m = nullptr;

    float* d_token_emb_v = nullptr;
    float* d_pos_emb_v = nullptr;
    float* d_lm_w_v = nullptr;

    float* d_hidden = nullptr;
    float* d_logits = nullptr;
    float* d_losses = nullptr;

    float* d_dhidden = nullptr;
    float* d_dlogits = nullptr;

    // ----------------------------
    // Allocate memory
    // ----------------------------

    CUDA_CHECK(cudaMalloc(&d_x, M * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_y, M * sizeof(int)));

    CUDA_CHECK(cudaMalloc(&d_token_emb, token_emb_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pos_emb, pos_emb_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_lm_w, lm_w_size * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_token_emb_grad, token_emb_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pos_emb_grad, pos_emb_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_lm_w_grad, lm_w_size * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_token_emb_m, token_emb_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pos_emb_m, pos_emb_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_lm_w_m, lm_w_size * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_token_emb_v, token_emb_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pos_emb_v, pos_emb_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_lm_w_v, lm_w_size * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_hidden, hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_logits, logits_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_losses, M * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_dhidden, hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dlogits, logits_size * sizeof(float)));

    // ----------------------------
    // Copy parameters to GPU
    // ----------------------------

    CUDA_CHECK(cudaMemcpy(
        d_token_emb,
        h_token_emb.data(),
        token_emb_size * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        d_pos_emb,
        h_pos_emb.data(),
        pos_emb_size * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        d_lm_w,
        h_lm_w.data(),
        lm_w_size * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    // ----------------------------
    // Zero Adam states
    // ----------------------------

    int threads = 256;

    zero_kernel<<<(token_emb_size + threads - 1) / threads, threads>>>(d_token_emb_m, token_emb_size);
    zero_kernel<<<(pos_emb_size + threads - 1) / threads, threads>>>(d_pos_emb_m, pos_emb_size);
    zero_kernel<<<(lm_w_size + threads - 1) / threads, threads>>>(d_lm_w_m, lm_w_size);

    zero_kernel<<<(token_emb_size + threads - 1) / threads, threads>>>(d_token_emb_v, token_emb_size);
    zero_kernel<<<(pos_emb_size + threads - 1) / threads, threads>>>(d_pos_emb_v, pos_emb_size);
    zero_kernel<<<(lm_w_size + threads - 1) / threads, threads>>>(d_lm_w_v, lm_w_size);

    CUDA_CHECK(cudaDeviceSynchronize());

    // ----------------------------
    // GEMM launch config
    // ----------------------------

    dim3 gemm_block(THREADS_X, THREADS_Y);

    dim3 gemm_grid(
        (V + BLOCK_N - 1) / BLOCK_N,
        (M + BLOCK_M - 1) / BLOCK_M
    );

    dim3 block2d(16, 16);

    dim3 grid_dA(
        (C + block2d.x - 1) / block2d.x,
        (M + block2d.y - 1) / block2d.y
    );

    dim3 grid_dW(
        (V + block2d.x - 1) / block2d.x,
        (C + block2d.y - 1) / block2d.y
    );

    int max_steps = 2000;
    int eval_every = 100;
    int eval_iters = 10;

    std::cout << "Starting training..." << std::endl;

    for (int step = 1; step <= max_steps; ++step) {
        // ====================================================
        // Get batch
        // ====================================================

        get_batch(train_tokens, h_x, h_y, train_rng);

        CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), M * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_y, h_y.data(), M * sizeof(int), cudaMemcpyHostToDevice));

        // ====================================================
        // Zero gradients
        // ====================================================

        zero_kernel<<<(token_emb_size + threads - 1) / threads, threads>>>(
            d_token_emb_grad,
            token_emb_size
        );

        zero_kernel<<<(pos_emb_size + threads - 1) / threads, threads>>>(
            d_pos_emb_grad,
            pos_emb_size
        );

        zero_kernel<<<(lm_w_size + threads - 1) / threads, threads>>>(
            d_lm_w_grad,
            lm_w_size
        );

        zero_kernel<<<(hidden_size + threads - 1) / threads, threads>>>(
            d_dhidden,
            hidden_size
        );

        zero_kernel<<<(logits_size + threads - 1) / threads, threads>>>(
            d_dlogits,
            logits_size
        );

        CUDA_CHECK(cudaGetLastError());

        // ====================================================
        // Forward
        // ====================================================

        embedding_forward_kernel<<<(M * C + threads - 1) / threads, threads>>>(
            d_x,
            d_token_emb,
            d_pos_emb,
            d_hidden,
            B,
            T,
            C
        );

        CUDA_CHECK(cudaGetLastError());

        hierarchical_gemm_2x4_kernel<<<gemm_grid, gemm_block>>>(
            d_hidden,
            d_lm_w,
            d_logits,
            M,
            C,
            V
        );

        CUDA_CHECK(cudaGetLastError());

        cross_entropy_forward_kernel<<<(M + threads - 1) / threads, threads>>>(
            d_logits,
            d_y,
            d_losses,
            M,
            V
        );

        CUDA_CHECK(cudaGetLastError());

        // ====================================================
        // Backward
        // ====================================================

        size_t ce_smem = 2 * threads * sizeof(float);

        cross_entropy_backward_kernel<<<M, threads, ce_smem>>>(
            d_logits,
            d_y,
            d_dlogits,
            M,
            V
        );

        CUDA_CHECK(cudaGetLastError());

        linear_backward_dA_kernel<<<grid_dA, block2d>>>(
            d_dlogits,
            d_lm_w,
            d_dhidden,
            M,
            C,
            V
        );

        CUDA_CHECK(cudaGetLastError());

        linear_backward_dW_kernel<<<grid_dW, block2d>>>(
            d_hidden,
            d_dlogits,
            d_lm_w_grad,
            M,
            C,
            V
        );

        CUDA_CHECK(cudaGetLastError());

        embedding_backward_kernel<<<(M * C + threads - 1) / threads, threads>>>(
            d_x,
            d_dhidden,
            d_token_emb_grad,
            d_pos_emb_grad,
            B,
            T,
            C
        );

        CUDA_CHECK(cudaGetLastError());

        // ====================================================
        // AdamW update
        // ====================================================

        adamw_update_kernel<<<(token_emb_size + threads - 1) / threads, threads>>>(
            d_token_emb,
            d_token_emb_grad,
            d_token_emb_m,
            d_token_emb_v,
            token_emb_size,
            LR,
            BETA1,
            BETA2,
            ADAM_EPS,
            WEIGHT_DECAY,
            step
        );

        adamw_update_kernel<<<(pos_emb_size + threads - 1) / threads, threads>>>(
            d_pos_emb,
            d_pos_emb_grad,
            d_pos_emb_m,
            d_pos_emb_v,
            pos_emb_size,
            LR,
            BETA1,
            BETA2,
            ADAM_EPS,
            WEIGHT_DECAY,
            step
        );

        adamw_update_kernel<<<(lm_w_size + threads - 1) / threads, threads>>>(
            d_lm_w,
            d_lm_w_grad,
            d_lm_w_m,
            d_lm_w_v,
            lm_w_size,
            LR,
            BETA1,
            BETA2,
            ADAM_EPS,
            WEIGHT_DECAY,
            step
        );

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // ====================================================
        // Print training and validation loss
        // ====================================================

        if (step % eval_every == 0 || step == 1) {
            float train_loss = average_loss_from_device(d_losses);

            float valid_loss = estimate_loss(
                valid_tokens,
                d_x,
                d_y,
                d_token_emb,
                d_pos_emb,
                d_lm_w,
                d_hidden,
                d_logits,
                d_losses,
                eval_iters,
                valid_rng
            );

            std::cout << "step " << step
                      << " | train loss " << train_loss
                      << " | valid loss " << valid_loss
                      << std::endl;
        }
    }

    // ----------------------------
    // Free memory
    // ----------------------------

    cudaFree(d_x);
    cudaFree(d_y);

    cudaFree(d_token_emb);
    cudaFree(d_pos_emb);
    cudaFree(d_lm_w);

    cudaFree(d_token_emb_grad);
    cudaFree(d_pos_emb_grad);
    cudaFree(d_lm_w_grad);

    cudaFree(d_token_emb_m);
    cudaFree(d_pos_emb_m);
    cudaFree(d_lm_w_m);

    cudaFree(d_token_emb_v);
    cudaFree(d_pos_emb_v);
    cudaFree(d_lm_w_v);

    cudaFree(d_hidden);
    cudaFree(d_logits);
    cudaFree(d_losses);

    cudaFree(d_dhidden);
    cudaFree(d_dlogits);

    std::cout << "Training finished." << std::endl;

    return 0;
}