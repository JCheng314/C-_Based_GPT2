#include "../include/transformer_kernels.h"

#include <algorithm>
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
static const int H = 4 * C;  // MLP hidden dimension

// GPT-2 tokenizer vocab size.
// If your tokenizer uses another vocab size, change this.
static const int V = 50257;

static const int M = B * T;

static const float LR = 3e-4f;
static const float BETA1 = 0.9f;
static const float BETA2 = 0.999f;
static const float ADAM_EPS = 1e-8f;
static const float WEIGHT_DECAY = 0.01f;

// ----------------------------
    // GEMM launch config
    // ----------------------------

    dim3 gemm_block(THREADS_X, THREADS_Y);

    dim3 gemm_grid(
        (V + BLOCK_N - 1) / BLOCK_N,
        (M + BLOCK_M - 1) / BLOCK_M
    );

    
    dim3 gemm_grid_lm(
        (V + BLOCK_N - 1) / BLOCK_N,
        (M + BLOCK_M - 1) / BLOCK_M
    );

    dim3 gemm_grid_mlp1(
        (H + BLOCK_N - 1) / BLOCK_N,
        (M + BLOCK_M - 1) / BLOCK_M
    );

    dim3 gemm_grid_mlp2(
        (C + BLOCK_N - 1) / BLOCK_N,
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

    dim3 grid_lm_dA(
        (C + block2d.x - 1) / block2d.x,
        (M + block2d.y - 1) / block2d.y
    );

    dim3 grid_lm_dW(
        (V + block2d.x - 1) / block2d.x,
        (C + block2d.y - 1) / block2d.y
    );

    dim3 grid_mlp2_dA(
        (H + block2d.x - 1) / block2d.x,
        (M + block2d.y - 1) / block2d.y
    );

    dim3 grid_mlp2_dW(
        (C + block2d.x - 1) / block2d.x,
        (H + block2d.y - 1) / block2d.y
    );

    dim3 grid_mlp1_dA(
        (C + block2d.x - 1) / block2d.x,
        (M + block2d.y - 1) / block2d.y
    );

    dim3 grid_mlp1_dW(
        (H + block2d.x - 1) / block2d.x,
        (C + block2d.y - 1) / block2d.y
    );

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

std::vector<uint32_t> load_tokens_u32(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);

    if (!file.is_open()) {
        std::cerr << "Could not open file: " << filename << std::endl;
        std::exit(1);
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    if (size % sizeof(uint32_t) != 0) {
        std::cerr << "Token file size is not divisible by uint32_t." << std::endl;
        std::exit(1);
    }

    std::vector<uint32_t> tokens(size / sizeof(uint32_t));

    if (!file.read(reinterpret_cast<char*>(tokens.data()), size)) {
        std::cerr << "Failed to read file: " << filename << std::endl;
        std::exit(1);
    }

    std::cout << "Loaded " << tokens.size()
              << " tokens from " << filename << std::endl;

    return tokens;
}

void get_batch(
    const std::vector<uint32_t>& data,
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
    const std::vector<uint32_t>& tokens,
    int* d_x,
    int* d_y,
    float* d_token_emb,
    float* d_pos_emb,
    float* d_ln_gamma,
    float* d_ln_beta,
    float* d_mlp_w1,
    float* d_mlp_b1,
    float* d_mlp_w2,
    float* d_mlp_b2,
    float* d_lm_w,
    float* d_hidden,
    float* d_ln_out,
    float* d_mlp_pre,
    float* d_mlp_act,
    float* d_resid,
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

        // ====================================================
        // Forward: LayerNorm
        // ln_out = LayerNorm(hidden)
        // ====================================================

        int ln_threads = 256;
        size_t ln_smem = 2 * ln_threads * sizeof(float);

        layernorm_forward_kernel<<<M, ln_threads, ln_smem>>>(
            d_hidden,
            d_ln_gamma,
            d_ln_beta,
            d_ln_out,
            B,
            T,
            C,
            1e-5f
        );

        CUDA_CHECK(cudaGetLastError());

        // ====================================================
        // Forward: MLP first projection
        // mlp_pre = ln_out @ W1
        // ====================================================

        hierarchical_gemm_2x4_kernel<<<gemm_grid_mlp1, gemm_block>>>(
            d_ln_out,
            d_mlp_w1,
            d_mlp_pre,
            M,
            C,
            H
        );

        CUDA_CHECK(cudaGetLastError());

        // ====================================================
        // Forward: bias + GELU
        // mlp_act = GELU(mlp_pre + b1)
        // ====================================================

        bias_gelu_kernel<<<(M * H + threads - 1) / threads, threads>>>(
            d_mlp_pre,
            d_mlp_b1,
            d_mlp_act,
            M,
            H
        );

        CUDA_CHECK(cudaGetLastError());

        // ====================================================
        // Forward: MLP second projection + bias + residual
        // resid = mlp_act @ W2 + b2 + hidden
        // ====================================================

        matmul_bias_residual_kernel<<<
            dim3((C + 15) / 16, (M + 15) / 16),
            dim3(16, 16)
        >>>(
            d_mlp_act,
            d_mlp_w2,
            d_mlp_b2,
            d_hidden,
            d_resid,
            M,
            H,
            C
        );

        CUDA_CHECK(cudaGetLastError());

        // ====================================================
        // Forward: LM head
        // logits = resid @ lm_w
        // ====================================================

        hierarchical_gemm_2x4_kernel<<<gemm_grid_lm, gemm_block>>>(
            d_resid,
            d_lm_w,
            d_logits,
            M,
            C,
            V
        );

        CUDA_CHECK(cudaGetLastError());

        // ====================================================
        // Forward: cross entropy
        // ====================================================

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

    std::vector<uint32_t> train_tokens = load_tokens_u32(train_path);
    std::vector<uint32_t> valid_tokens = load_tokens_u32(valid_path);

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

    int ln_gamma_size = C;
    int ln_beta_size = C;

    int mlp_w1_size = C * H;
    int mlp_b1_size = H;
    int mlp_w2_size = H * C;
    int mlp_b2_size = C;

    int lm_w_size = C * V;

    int hidden_size = M * C;
    int ln_out_size = M * C;
    int mlp_pre_size = M * H;
    int mlp_act_size = M * H;
    int mlp_out_size = M * C;
    int resid_size = M * C;
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

    std::vector<float> h_ln_gamma(ln_gamma_size);
    std::vector<float> h_ln_beta(ln_beta_size);

    std::vector<float> h_mlp_w1(mlp_w1_size);
    std::vector<float> h_mlp_b1(mlp_b1_size);
    std::vector<float> h_mlp_w2(mlp_w2_size);
    std::vector<float> h_mlp_b2(mlp_b2_size);

    std::vector<float> h_lm_w(lm_w_size);

    init_normal(h_token_emb, 0.02f);
    init_normal(h_pos_emb, 0.01f);

    for (int i = 0; i < C; ++i) {
        h_ln_gamma[i] = 1.0f;
        h_ln_beta[i] = 0.0f;
    }

    init_normal(h_mlp_w1, 0.02f);
    std::fill(h_mlp_b1.begin(), h_mlp_b1.end(), 0.0f);

    init_normal(h_mlp_w2, 0.02f);
    std::fill(h_mlp_b2.begin(), h_mlp_b2.end(), 0.0f);

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

    float* d_ln_gamma = nullptr;
    float* d_ln_beta = nullptr;

    float* d_mlp_w1 = nullptr;
    float* d_mlp_b1 = nullptr;
    float* d_mlp_w2 = nullptr;
    float* d_mlp_b2 = nullptr;

    float* d_ln_gamma_grad = nullptr;
    float* d_ln_beta_grad = nullptr;

    float* d_mlp_w1_grad = nullptr;
    float* d_mlp_b1_grad = nullptr;
    float* d_mlp_w2_grad = nullptr;
    float* d_mlp_b2_grad = nullptr;

    float* d_ln_gamma_m = nullptr;
    float* d_ln_beta_m = nullptr;
    float* d_mlp_w1_m = nullptr;
    float* d_mlp_b1_m = nullptr;
    float* d_mlp_w2_m = nullptr;
    float* d_mlp_b2_m = nullptr;

    float* d_ln_gamma_v = nullptr;
    float* d_ln_beta_v = nullptr;
    float* d_mlp_w1_v = nullptr;
    float* d_mlp_b1_v = nullptr;
    float* d_mlp_w2_v = nullptr;
    float* d_mlp_b2_v = nullptr;

    float* d_ln_out = nullptr;
    float* d_mlp_pre = nullptr;
    float* d_mlp_act = nullptr;
    float* d_mlp_out = nullptr;
    float* d_resid = nullptr;

    float* d_dln_out = nullptr;
    float* d_dmlp_pre = nullptr;
    float* d_dmlp_act = nullptr;
    float* d_dmlp_out = nullptr;
    float* d_dresid = nullptr;
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

    CUDA_CHECK(cudaMalloc(&d_ln_gamma, ln_gamma_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ln_beta, ln_beta_size * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_mlp_w1, mlp_w1_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mlp_b1, mlp_b1_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mlp_w2, mlp_w2_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mlp_b2, mlp_b2_size * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_ln_gamma_grad, ln_gamma_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ln_beta_grad, ln_beta_size * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_mlp_w1_grad, mlp_w1_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mlp_b1_grad, mlp_b1_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mlp_w2_grad, mlp_w2_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mlp_b2_grad, mlp_b2_size * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_ln_gamma_m, ln_gamma_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ln_beta_m, ln_beta_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mlp_w1_m, mlp_w1_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mlp_b1_m, mlp_b1_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mlp_w2_m, mlp_w2_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mlp_b2_m, mlp_b2_size * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_ln_gamma_v, ln_gamma_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ln_beta_v, ln_beta_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mlp_w1_v, mlp_w1_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mlp_b1_v, mlp_b1_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mlp_w2_v, mlp_w2_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mlp_b2_v, mlp_b2_size * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_ln_out, ln_out_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mlp_pre, mlp_pre_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mlp_act, mlp_act_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mlp_out, mlp_out_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_resid, resid_size * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_dln_out, ln_out_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dmlp_pre, mlp_pre_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dmlp_act, mlp_act_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dmlp_out, mlp_out_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dresid, resid_size * sizeof(float)));

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

    CUDA_CHECK(cudaMemcpy(
        d_ln_gamma, h_ln_gamma.data(),
        ln_gamma_size * sizeof(float), 
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        d_ln_beta, h_ln_beta.data(),
        ln_beta_size * sizeof(float), 
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        d_mlp_w1, h_mlp_w1.data(),
        mlp_w1_size * sizeof(float), 
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        d_mlp_b1, 
        h_mlp_b1.data(),
        mlp_b1_size * sizeof(float), 
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        d_mlp_w2, 
        h_mlp_w2.data(),
        mlp_w2_size * sizeof(float), 
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        d_mlp_b2, 
        h_mlp_b2.data(),
        mlp_b2_size * sizeof(float), 
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

    zero_kernel<<<(ln_gamma_size + threads - 1) / threads, threads>>>(d_ln_gamma_m, ln_gamma_size);
    zero_kernel<<<(ln_beta_size + threads - 1) / threads, threads>>>(d_ln_beta_m, ln_beta_size);
    zero_kernel<<<(mlp_w1_size + threads - 1) / threads, threads>>>(d_mlp_w1_m, mlp_w1_size);
    zero_kernel<<<(mlp_b1_size + threads - 1) / threads, threads>>>(d_mlp_b1_m, mlp_b1_size);
    zero_kernel<<<(mlp_w2_size + threads - 1) / threads, threads>>>(d_mlp_w2_m, mlp_w2_size);
    zero_kernel<<<(mlp_b2_size + threads - 1) / threads, threads>>>(d_mlp_b2_m, mlp_b2_size);

    zero_kernel<<<(ln_gamma_size + threads - 1) / threads, threads>>>(d_ln_gamma_v, ln_gamma_size);
    zero_kernel<<<(ln_beta_size + threads - 1) / threads, threads>>>(d_ln_beta_v, ln_beta_size);
    zero_kernel<<<(mlp_w1_size + threads - 1) / threads, threads>>>(d_mlp_w1_v, mlp_w1_size);
    zero_kernel<<<(mlp_b1_size + threads - 1) / threads, threads>>>(d_mlp_b1_v, mlp_b1_size);
    zero_kernel<<<(mlp_w2_size + threads - 1) / threads, threads>>>(d_mlp_w2_v, mlp_w2_size);
    zero_kernel<<<(mlp_b2_size + threads - 1) / threads, threads>>>(d_mlp_b2_v, mlp_b2_size);

    zero_kernel<<<(ln_gamma_size + threads - 1) / threads, threads>>>(d_ln_gamma_grad, ln_gamma_size);
    zero_kernel<<<(ln_beta_size + threads - 1) / threads, threads>>>(d_ln_beta_grad, ln_beta_size);

    zero_kernel<<<(mlp_w1_size + threads - 1) / threads, threads>>>(d_mlp_w1_grad, mlp_w1_size);
    zero_kernel<<<(mlp_b1_size + threads - 1) / threads, threads>>>(d_mlp_b1_grad, mlp_b1_size);
    zero_kernel<<<(mlp_w2_size + threads - 1) / threads, threads>>>(d_mlp_w2_grad, mlp_w2_size);
    zero_kernel<<<(mlp_b2_size + threads - 1) / threads, threads>>>(d_mlp_b2_grad, mlp_b2_size);

    zero_kernel<<<(ln_out_size + threads - 1) / threads, threads>>>(d_dln_out, ln_out_size);
    zero_kernel<<<(mlp_pre_size + threads - 1) / threads, threads>>>(d_dmlp_pre, mlp_pre_size);
    zero_kernel<<<(mlp_act_size + threads - 1) / threads, threads>>>(d_dmlp_act, mlp_act_size);
    zero_kernel<<<(mlp_out_size + threads - 1) / threads, threads>>>(d_dmlp_out, mlp_out_size);
    zero_kernel<<<(resid_size + threads - 1) / threads, threads>>>(d_dresid, resid_size);

    

    CUDA_CHECK(cudaDeviceSynchronize());


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

        zero_kernel<<<(ln_gamma_size + threads - 1) / threads, threads>>>(
            d_ln_gamma_grad,
            ln_gamma_size
        );
        zero_kernel<<<(ln_beta_size + threads - 1) / threads, threads>>>(d_ln_beta_grad, ln_beta_size);

        zero_kernel<<<(mlp_w1_size + threads - 1) / threads, threads>>>(d_mlp_w1_grad, mlp_w1_size);
        zero_kernel<<<(mlp_b1_size + threads - 1) / threads, threads>>>(d_mlp_b1_grad, mlp_b1_size);
        zero_kernel<<<(mlp_w2_size + threads - 1) / threads, threads>>>(d_mlp_w2_grad, mlp_w2_size);
        zero_kernel<<<(mlp_b2_size + threads - 1) / threads, threads>>>(d_mlp_b2_grad, mlp_b2_size);

        zero_kernel<<<(ln_out_size + threads - 1) / threads, threads>>>(d_dln_out, ln_out_size);
        zero_kernel<<<(mlp_pre_size + threads - 1) / threads, threads>>>(d_dmlp_pre, mlp_pre_size);
        zero_kernel<<<(mlp_act_size + threads - 1) / threads, threads>>>(d_dmlp_act, mlp_act_size);
        zero_kernel<<<(mlp_out_size + threads - 1) / threads, threads>>>(d_dmlp_out, mlp_out_size);
        zero_kernel<<<(resid_size + threads - 1) / threads, threads>>>(d_dresid, resid_size);

        CUDA_CHECK(cudaGetLastError());

        // ====================================================
        // Forward: embedding
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

        // ====================================================
        // Forward: LayerNorm
        // ln_out = LayerNorm(hidden)
        // ====================================================

        int ln_threads = 256;
        size_t ln_smem = 2 * ln_threads * sizeof(float);

        layernorm_forward_kernel<<<M, ln_threads, ln_smem>>>(
            d_hidden,
            d_ln_gamma,
            d_ln_beta,
            d_ln_out,
            B,
            T,
            C,
            1e-5f
        );

        CUDA_CHECK(cudaGetLastError());

        // ====================================================
        // Forward: MLP first projection
        // mlp_pre = ln_out @ W1
        // ====================================================

        hierarchical_gemm_2x4_kernel<<<gemm_grid_mlp1, gemm_block>>>(
            d_ln_out,
            d_mlp_w1,
            d_mlp_pre,
            M,
            C,
            H
        );

        CUDA_CHECK(cudaGetLastError());

        // ====================================================
        // Forward: bias + GELU
        // mlp_act = GELU(mlp_pre + b1)
        // ====================================================

        bias_gelu_kernel<<<(M * H + threads - 1) / threads, threads>>>(
            d_mlp_pre,
            d_mlp_b1,
            d_mlp_act,
            M,
            H
        );

        CUDA_CHECK(cudaGetLastError());

        // ====================================================
        // Forward: MLP second projection + bias + residual
        // resid = mlp_act @ W2 + b2 + hidden
        // ====================================================

        matmul_bias_residual_kernel<<<
            dim3((C + 15) / 16, (M + 15) / 16),
            dim3(16, 16)
        >>>(
            d_mlp_act,
            d_mlp_w2,
            d_mlp_b2,
            d_hidden,
            d_resid,
            M,
            H,
            C
        );

        CUDA_CHECK(cudaGetLastError());

        // ====================================================
        // Forward: LM head
        // logits = resid @ lm_w
        // ====================================================

        hierarchical_gemm_2x4_kernel<<<gemm_grid_lm, gemm_block>>>(
            d_resid,
            d_lm_w,
            d_logits,
            M,
            C,
            V
        );

        CUDA_CHECK(cudaGetLastError());

        // ====================================================
        // Forward: cross entropy
        // ====================================================

        cross_entropy_forward_kernel<<<(M + threads - 1) / threads, threads>>>(
            d_logits,
            d_y,
            d_losses,
            M,
            V
        );

        CUDA_CHECK(cudaGetLastError());

        // ====================================================
        // Backward: cross entropy
        // d_logits
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

        // ====================================================
        // Backward: LM head
        //
        // logits = resid @ lm_w
        //
        // d_resid += d_logits @ lm_w^T
        // d_lm_w += resid^T @ d_logits
        // ====================================================

        linear_backward_dA_kernel<<<grid_lm_dA, block2d>>>(
            d_dlogits,
            d_lm_w,
            d_dresid,
            M,
            C,
            V
        );

        CUDA_CHECK(cudaGetLastError());

        linear_backward_dW_kernel<<<grid_lm_dW, block2d>>>(
            d_resid,
            d_dlogits,
            d_lm_w_grad,
            M,
            C,
            V
        );

        CUDA_CHECK(cudaGetLastError());

        // ====================================================
        // Backward: residual from MLP output
        //
        // resid = mlp_out + b2 + hidden
        //
        // d_mlp_out += d_resid
        // d_hidden += d_resid
        // d_b2 += sum(d_resid)
        // ====================================================

        residual_add_backward_kernel<<<(M * C + threads - 1) / threads, threads>>>(
            d_dresid,
            d_dmlp_out,
            d_dhidden,
            M * C
        );

        CUDA_CHECK(cudaGetLastError());

        linear_backward_bias_kernel<<<(C + threads - 1) / threads, threads>>>(
            d_dresid,
            d_mlp_b2_grad,
            M,
            C
        );

        CUDA_CHECK(cudaGetLastError());

        // ====================================================
        // Backward: MLP second projection
        //
        // mlp_out = mlp_act @ W2
        //
        // d_mlp_act += d_mlp_out @ W2^T
        // d_W2 += mlp_act^T @ d_mlp_out
        // ====================================================

        linear_backward_dA_kernel<<<grid_mlp2_dA, block2d>>>(
            d_dmlp_out,
            d_mlp_w2,
            d_dmlp_act,
            M,
            H,
            C
        );

        CUDA_CHECK(cudaGetLastError());

        linear_backward_dW_kernel<<<grid_mlp2_dW, block2d>>>(
            d_mlp_act,
            d_dmlp_out,
            d_mlp_w2_grad,
            M,
            H,
            C
        );

        CUDA_CHECK(cudaGetLastError());

        // ====================================================
        // Backward: bias + GELU
        //
        // mlp_act = GELU(mlp_pre + b1)
        // ====================================================

        bias_gelu_backward_kernel<<<(M * H + threads - 1) / threads, threads>>>(
            d_mlp_pre,
            d_mlp_b1,
            d_dmlp_act,
            d_dmlp_pre,
            d_mlp_b1_grad,
            M,
            H
        );

        CUDA_CHECK(cudaGetLastError());

        // ====================================================
        // Backward: MLP first projection
        //
        // mlp_pre = ln_out @ W1
        //
        // d_ln_out += d_mlp_pre @ W1^T
        // d_W1 += ln_out^T @ d_mlp_pre
        // ====================================================

        linear_backward_dA_kernel<<<grid_mlp1_dA, block2d>>>(
            d_dmlp_pre,
            d_mlp_w1,
            d_dln_out,
            M,
            C,
            H
        );

        CUDA_CHECK(cudaGetLastError());

        linear_backward_dW_kernel<<<grid_mlp1_dW, block2d>>>(
            d_ln_out,
            d_dmlp_pre,
            d_mlp_w1_grad,
            M,
            C,
            H
        );

        CUDA_CHECK(cudaGetLastError());

        // ====================================================
        // Backward: LayerNorm
        //
        // ln_out = LayerNorm(hidden)
        //
        // d_hidden += LayerNormBackward(d_ln_out)
        // ====================================================

        size_t ln_bwd_smem = 4 * ln_threads * sizeof(float);

        layernorm_backward_kernel<<<M, ln_threads, ln_bwd_smem>>>(
            d_hidden,
            d_ln_gamma,
            d_dln_out,
            d_dhidden,
            d_ln_gamma_grad,
            d_ln_beta_grad,
            M,
            C,
            1e-5f
        );

        CUDA_CHECK(cudaGetLastError());

        // ====================================================
        // Backward: embedding
        //
        // hidden = token_emb + pos_emb
        // ====================================================

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

        adamw_update_kernel<<<(ln_gamma_size + threads - 1) / threads, threads>>>(
            d_ln_gamma,
            d_ln_gamma_grad,
            d_ln_gamma_m,
            d_ln_gamma_v,
            ln_gamma_size,
            LR,
            BETA1,
            BETA2,
            ADAM_EPS,
            0.0f,
            step
        );

        adamw_update_kernel<<<(ln_beta_size + threads - 1) / threads, threads>>>(
            d_ln_beta,
            d_ln_beta_grad,
            d_ln_beta_m,
            d_ln_beta_v,
            ln_beta_size,
            LR,
            BETA1,
            BETA2,
            ADAM_EPS,
            0.0f,
            step
        );

        adamw_update_kernel<<<(mlp_w1_size + threads - 1) / threads, threads>>>(
            d_mlp_w1,
            d_mlp_w1_grad,
            d_mlp_w1_m,
            d_mlp_w1_v,
            mlp_w1_size,
            LR,
            BETA1,
            BETA2,
            ADAM_EPS,
            WEIGHT_DECAY,
            step
        );

        adamw_update_kernel<<<(mlp_b1_size + threads - 1) / threads, threads>>>(
            d_mlp_b1,
            d_mlp_b1_grad,
            d_mlp_b1_m,
            d_mlp_b1_v,
            mlp_b1_size,
            LR,
            BETA1,
            BETA2,
            ADAM_EPS,
            0.0f,
            step
        );

        adamw_update_kernel<<<(mlp_w2_size + threads - 1) / threads, threads>>>(
            d_mlp_w2,
            d_mlp_w2_grad,
            d_mlp_w2_m,
            d_mlp_w2_v,
            mlp_w2_size,
            LR,
            BETA1,
            BETA2,
            ADAM_EPS,
            WEIGHT_DECAY,
            step
        );

        adamw_update_kernel<<<(mlp_b2_size + threads - 1) / threads, threads>>>(
            d_mlp_b2,
            d_mlp_b2_grad,
            d_mlp_b2_m,
            d_mlp_b2_v,
            mlp_b2_size,
            LR,
            BETA1,
            BETA2,
            ADAM_EPS,
            0.0f,
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
                d_ln_gamma,
                d_ln_beta,
                d_mlp_w1,
                d_mlp_b1,
                d_mlp_w2,
                d_mlp_b2,
                d_lm_w,
                d_hidden,
                d_ln_out,
                d_mlp_pre,
                d_mlp_act,
                d_resid,
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