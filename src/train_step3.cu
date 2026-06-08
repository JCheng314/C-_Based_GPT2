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

static const int B = 2;
static const int T = 64;
static const int C = 64;
static const int NH = 4;
static const int HS = C / NH;
static const int H = 4 * C;

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

// ----------------------------
// Launch configs
// ----------------------------

int threads = 256;

dim3 gemm_block(THREADS_X, THREADS_Y);

dim3 gemm_grid_qkv(
    (3 * C + BLOCK_N - 1) / BLOCK_N,
    (M + BLOCK_M - 1) / BLOCK_M
);

dim3 gemm_grid_mlp1(
    (H + BLOCK_N - 1) / BLOCK_N,
    (M + BLOCK_M - 1) / BLOCK_M
);

dim3 gemm_grid_lm(
    (V + BLOCK_N - 1) / BLOCK_N,
    (M + BLOCK_M - 1) / BLOCK_M
);

dim3 block2d(16, 16);

dim3 grid_scores(
    (T + block2d.x - 1) / block2d.x,
    (T + block2d.y - 1) / block2d.y,
    B * NH
);

dim3 grid_attn_value(
    (HS + block2d.x - 1) / block2d.x,
    (T + block2d.y - 1) / block2d.y,
    B * NH
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

dim3 grid_d_attn_proj_A(
    (C + block2d.x - 1) / block2d.x,
    (M + block2d.y - 1) / block2d.y
);

dim3 grid_d_attn_proj_W(
    (C + block2d.x - 1) / block2d.x,
    (C + block2d.y - 1) / block2d.y
);

dim3 grid_d_qkv_A(
    (C + block2d.x - 1) / block2d.x,
    (M + block2d.y - 1) / block2d.y
);

dim3 grid_d_qkv_W(
    (3 * C + block2d.x - 1) / block2d.x,
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

    float* d_ln1_gamma,
    float* d_ln1_beta,

    float* d_qkv_w,
    float* d_qkv_b,

    float* d_attn_proj_w,
    float* d_attn_proj_b,

    float* d_ln2_gamma,
    float* d_ln2_beta,

    float* d_mlp_w1,
    float* d_mlp_b1,
    float* d_mlp_w2,
    float* d_mlp_b2,

    float* d_lm_w,

    float* d_hidden,
    float* d_ln1_out,

    float* d_qkv_pre,
    float* d_qkv,

    float* d_Q,
    float* d_K,
    float* d_Vv,

    float* d_scores,
    float* d_probs,
    float* d_attn_heads,
    float* d_attn_merge,

    float* d_h1,

    float* d_ln2_out,
    float* d_mlp_pre,
    float* d_mlp_act,
    float* d_mlp_out,

    float* d_h2,

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

        // ====================================================
        // Forward: embedding
        // hidden = token_emb[x] + pos_emb
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
        CUDA_CHECK(cudaDeviceSynchronize());

        // ====================================================
        // Forward: ln1
        // ln1_out = LayerNorm(hidden)
        // ====================================================

        int ln_threads = 256;
        size_t ln_smem = 2 * ln_threads * sizeof(float);

        layernorm_forward_kernel<<<M, ln_threads, ln_smem>>>(
            d_hidden,
            d_ln1_gamma,
            d_ln1_beta,
            d_ln1_out,
            B,
            T,
            C,
            1e-5f
        );

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // ====================================================
        // Forward: qkv projection
        // qkv_pre = ln1_out @ qkv_w
        // ====================================================

        hierarchical_gemm_2x4_kernel<<<gemm_grid_qkv, gemm_block>>>(
            d_ln1_out,
            d_qkv_w,
            d_qkv_pre,
            M,
            C,
            3 * C
        );

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // ====================================================
        // Forward: qkv bias
        // qkv = qkv_pre + qkv_b
        // ====================================================

        add_bias_kernel<<<(M * 3 * C + threads - 1) / threads, threads>>>(
            d_qkv_pre,
            d_qkv_b,
            d_qkv,
            M,
            3 * C
        );

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // ====================================================
        // Forward: split Q/K/V
        // Q, K, V shape: [B, NH, T, HS]
        // ====================================================

        split_qkv_kernel<<<(M * 3 * C + threads - 1) / threads, threads>>>(
            d_qkv,
            d_Q,
            d_K,
            d_Vv,
            B,
            T,
            NH,
            HS
        );

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // ====================================================
        // Forward: attention scores
        // scores = Q @ K^T / sqrt(HS)
        // scores shape: [B, NH, T, T]
        // ====================================================

        attention_scores_kernel<<<grid_scores, block2d>>>(
            d_Q,
            d_K,
            d_scores,
            B,
            NH,
            T,
            HS
        );

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // ====================================================
        // Forward: causal mask
        // scores[j > i] = -1e20f
        // IMPORTANT: causal_mask_kernel expects 3D grid.
        // ====================================================

        causal_mask_kernel<<<grid_scores, block2d>>>(
            d_scores,
            B,
            NH,
            T
        );

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // ====================================================
        // Forward: masked softmax
        // IMPORTANT: masked_softmax_kernel uses dynamic shared memory.
        // ====================================================

        size_t softmax_smem = 2 * threads * sizeof(float);

        masked_softmax_kernel<<<B * NH * T, threads, softmax_smem>>>(
            d_scores,
            d_probs,
            B,
            NH,
            T
        );

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // ====================================================
        // Forward: attention value aggregation
        // attn_heads = probs @ V
        // ====================================================

        attention_value_kernel<<<grid_attn_value, block2d>>>(
            d_probs,
            d_Vv,
            d_attn_heads,
            B,
            NH,
            T,
            HS
        );

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // ====================================================
        // Forward: merge heads
        // attn_merge shape: [M, C]
        // ====================================================

        merge_heads_kernel<<<(M * C + threads - 1) / threads, threads>>>(
            d_attn_heads,
            d_attn_merge,
            B,
            T,
            NH,
            HS
        );

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // ====================================================
        // Forward: attention output projection + residual
        // h1 = attn_merge @ attn_proj_w + attn_proj_b + hidden
        // ====================================================

        matmul_bias_residual_kernel<<<
            dim3((C + 15) / 16, (M + 15) / 16),
            dim3(16, 16)
        >>>(
            d_attn_merge,
            d_attn_proj_w,
            d_attn_proj_b,
            d_hidden,
            d_h1,
            M,
            C,
            C
        );

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // ====================================================
        // Forward: ln2
        // ln2_out = LayerNorm(h1)
        // ====================================================

        layernorm_forward_kernel<<<M, ln_threads, ln_smem>>>(
            d_h1,
            d_ln2_gamma,
            d_ln2_beta,
            d_ln2_out,
            B,
            T,
            C,
            1e-5f
        );

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // ====================================================
        // Forward: MLP first projection
        // mlp_pre = ln2_out @ W1
        // ====================================================

        hierarchical_gemm_2x4_kernel<<<gemm_grid_mlp1, gemm_block>>>(
            d_ln2_out,
            d_mlp_w1,
            d_mlp_pre,
            M,
            C,
            H
        );

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

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
        CUDA_CHECK(cudaDeviceSynchronize());

        // ====================================================
        // Forward: MLP second projection + residual
        // h2 = mlp_act @ W2 + b2 + h1
        // ====================================================

        matmul_bias_residual_kernel<<<
            dim3((C + 15) / 16, (M + 15) / 16),
            dim3(16, 16)
        >>>(
            d_mlp_act,
            d_mlp_w2,
            d_mlp_b2,
            d_h1,
            d_h2,
            M,
            H,
            C
        );

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // ====================================================
        // Forward: LM head
        // logits = h2 @ lm_w
        // ====================================================

        hierarchical_gemm_2x4_kernel<<<gemm_grid_lm, gemm_block>>>(
            d_h2,
            d_lm_w,
            d_logits,
            M,
            C,
            V
        );

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

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

    // attention block ln1
    int ln1_gamma_size = C;
    int ln1_beta_size = C;

    // qkv projection
    int qkv_w_size = C * (3 * C);
    int qkv_b_size = 3 * C;

    // attention output projection
    int attn_proj_w_size = C * C;
    int attn_proj_b_size = C;

    // mlp block ln2
    int ln2_gamma_size = C;
    int ln2_beta_size = C;

    // mlp
    int mlp_w1_size = C * H;
    int mlp_b1_size = H;
    int mlp_w2_size = H * C;
    int mlp_b2_size = C;

    // lm head
    int lm_w_size = C * V;

    // activations

    int hidden_size = M * C;

    int ln1_out_size = M * C;

    int qkv_pre_size = M * (3 * C);
    int qkv_size = M * (3 * C);

    int q_size = B * NH * T * HS;
    int k_size = B * NH * T * HS;
    int v_size = B * NH * T * HS;

    int scores_size = B * NH * T * T;
    int probs_size = B * NH * T * T;

    int attn_heads_size = B * NH * T * HS;
    int attn_merge_size = M * C;

    int h1_size = M * C;

    int ln2_out_size = M * C;

    int mlp_pre_size = M * H;
    int mlp_act_size = M * H;
    int mlp_out_size = M * C;

    int h2_size = M * C;

    int logits_size = M * V;
    int losses_size = M;

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

    std::vector<float> h_ln1_gamma(ln1_gamma_size);
    std::vector<float> h_ln1_beta(ln1_beta_size);

    std::vector<float> h_qkv_w(qkv_w_size);
    std::vector<float> h_qkv_b(qkv_b_size);

    std::vector<float> h_attn_proj_w(attn_proj_w_size);
    std::vector<float> h_attn_proj_b(attn_proj_b_size);

    std::vector<float> h_ln2_gamma(ln2_gamma_size);
    std::vector<float> h_ln2_beta(ln2_beta_size);

    std::vector<float> h_mlp_w1(mlp_w1_size);
    std::vector<float> h_mlp_b1(mlp_b1_size);
    std::vector<float> h_mlp_w2(mlp_w2_size);
    std::vector<float> h_mlp_b2(mlp_b2_size);

    std::vector<float> h_lm_w(lm_w_size);

    init_normal(h_token_emb, 0.02f);
    init_normal(h_pos_emb, 0.01f);

    for (int i = 0; i < C; ++i) {
        h_ln1_gamma[i] = 1.0f;
        h_ln1_beta[i] = 0.0f;

        h_ln2_gamma[i] = 1.0f;
        h_ln2_beta[i] = 0.0f;
    }

    init_normal(h_qkv_w, 0.02f);
    std::fill(h_qkv_b.begin(), h_qkv_b.end(), 0.0f);

    init_normal(h_attn_proj_w, 0.02f);
    std::fill(h_attn_proj_b.begin(), h_attn_proj_b.end(), 0.0f);

    init_normal(h_mlp_w1, 0.02f);
    std::fill(h_mlp_b1.begin(), h_mlp_b1.end(), 0.0f);

    init_normal(h_mlp_w2, 0.02f);
    std::fill(h_mlp_b2.begin(), h_mlp_b2.end(), 0.0f);

    init_normal(h_lm_w, 0.02f);

    // ----------------------------
    // Device pointers
    // ----------------------------

    // input / target batch
    int* d_x = nullptr;
    int* d_y = nullptr;

    // ----------------------------
    // Parameters
    // ----------------------------

    float* d_token_emb = nullptr;
    float* d_pos_emb = nullptr;
    float* d_lm_w = nullptr;

    float* d_ln1_gamma = nullptr;
    float* d_ln1_beta = nullptr;

    float* d_qkv_w = nullptr;
    float* d_qkv_b = nullptr;

    float* d_attn_proj_w = nullptr;
    float* d_attn_proj_b = nullptr;

    float* d_ln2_gamma = nullptr;
    float* d_ln2_beta = nullptr;

    float* d_mlp_w1 = nullptr;
    float* d_mlp_b1 = nullptr;
    float* d_mlp_w2 = nullptr;
    float* d_mlp_b2 = nullptr;

    // ----------------------------
    // Parameter gradients
    // ----------------------------

    float* d_token_emb_grad = nullptr;
    float* d_pos_emb_grad = nullptr;
    float* d_lm_w_grad = nullptr;

    float* d_ln1_gamma_grad = nullptr;
    float* d_ln1_beta_grad = nullptr;

    float* d_qkv_w_grad = nullptr;
    float* d_qkv_b_grad = nullptr;

    float* d_attn_proj_w_grad = nullptr;
    float* d_attn_proj_b_grad = nullptr;

    float* d_ln2_gamma_grad = nullptr;
    float* d_ln2_beta_grad = nullptr;

    float* d_mlp_w1_grad = nullptr;
    float* d_mlp_b1_grad = nullptr;
    float* d_mlp_w2_grad = nullptr;
    float* d_mlp_b2_grad = nullptr;

    // ----------------------------
    // Adam first moment buffers
    // ----------------------------

    float* d_token_emb_m = nullptr;
    float* d_pos_emb_m = nullptr;
    float* d_lm_w_m = nullptr;

    float* d_ln1_gamma_m = nullptr;
    float* d_ln1_beta_m = nullptr;

    float* d_qkv_w_m = nullptr;
    float* d_qkv_b_m = nullptr;

    float* d_attn_proj_w_m = nullptr;
    float* d_attn_proj_b_m = nullptr;

    float* d_ln2_gamma_m = nullptr;
    float* d_ln2_beta_m = nullptr;

    float* d_mlp_w1_m = nullptr;
    float* d_mlp_b1_m = nullptr;
    float* d_mlp_w2_m = nullptr;
    float* d_mlp_b2_m = nullptr;

    // ----------------------------
    // Adam second moment buffers
    // ----------------------------

    float* d_token_emb_v = nullptr;
    float* d_pos_emb_v = nullptr;
    float* d_lm_w_v = nullptr;

    float* d_ln1_gamma_v = nullptr;
    float* d_ln1_beta_v = nullptr;

    float* d_qkv_w_v = nullptr;
    float* d_qkv_b_v = nullptr;

    float* d_attn_proj_w_v = nullptr;
    float* d_attn_proj_b_v = nullptr;

    float* d_ln2_gamma_v = nullptr;
    float* d_ln2_beta_v = nullptr;

    float* d_mlp_w1_v = nullptr;
    float* d_mlp_b1_v = nullptr;
    float* d_mlp_w2_v = nullptr;
    float* d_mlp_b2_v = nullptr;

    // ----------------------------
    // Forward activation buffers
    // ----------------------------

    float* d_hidden = nullptr;        // [M, C]

    float* d_ln1_out = nullptr;       // [M, C]

    float* d_qkv_pre = nullptr;       // [M, 3C], before qkv bias
    float* d_qkv = nullptr;           // [M, 3C], after qkv bias

    float* d_Q = nullptr;             // [B, NH, T, HS]
    float* d_K = nullptr;             // [B, NH, T, HS]
    float* d_Vv = nullptr;            // [B, NH, T, HS]

    float* d_scores = nullptr;        // [B, NH, T, T]
    float* d_probs = nullptr;         // [B, NH, T, T]

    float* d_attn_heads = nullptr;    // [B, NH, T, HS]
    float* d_attn_merge = nullptr;    // [M, C]

    float* d_h1 = nullptr;            // [M, C]

    float* d_ln2_out = nullptr;       // [M, C]

    float* d_mlp_pre = nullptr;       // [M, H]
    float* d_mlp_act = nullptr;       // [M, H]
    float* d_mlp_out = nullptr;       // [M, C]

    float* d_h2 = nullptr;            // [M, C]

    float* d_logits = nullptr;        // [M, V]
    float* d_losses = nullptr;        // [M]

    // ----------------------------
    // Backward activation-gradient buffers
    // ----------------------------

    float* d_dlogits = nullptr;       // [M, V]
    float* d_dh2 = nullptr;           // [M, C]

    float* d_dmlp_out = nullptr;      // [M, C]
    float* d_dmlp_act = nullptr;      // [M, H]
    float* d_dmlp_pre = nullptr;      // [M, H]

    float* d_dln2_out = nullptr;      // [M, C]
    float* d_dh1 = nullptr;           // [M, C]

    float* d_dattn_out = nullptr;     // [M, C]
    float* d_dattn_merge = nullptr;   // [M, C]
    float* d_dattn_heads = nullptr;   // [B, NH, T, HS]

    float* d_dprobs = nullptr;        // [B, NH, T, T]
    float* d_dscores = nullptr;       // [B, NH, T, T]

    float* d_dQ = nullptr;            // [B, NH, T, HS]
    float* d_dK = nullptr;            // [B, NH, T, HS]
    float* d_dVv = nullptr;           // [B, NH, T, HS]

    float* d_dqkv = nullptr;          // [M, 3C]
    float* d_dln1_out = nullptr;      // [M, C]

    float* d_dhidden = nullptr;       // [M, C]
    // ----------------------------
    // Allocate memory
    // ----------------------------

    CUDA_CHECK(cudaMalloc(&d_x, M * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_y, M * sizeof(int)));

    // parameters
    CUDA_CHECK(cudaMalloc(&d_token_emb, token_emb_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pos_emb, pos_emb_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_lm_w, lm_w_size * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_ln1_gamma, ln1_gamma_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ln1_beta, ln1_beta_size * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_qkv_w, qkv_w_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_qkv_b, qkv_b_size * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_attn_proj_w, attn_proj_w_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_attn_proj_b, attn_proj_b_size * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_ln2_gamma, ln2_gamma_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ln2_beta, ln2_beta_size * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_mlp_w1, mlp_w1_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mlp_b1, mlp_b1_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mlp_w2, mlp_w2_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mlp_b2, mlp_b2_size * sizeof(float)));

    // parameter gradients
    CUDA_CHECK(cudaMalloc(&d_token_emb_grad, token_emb_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pos_emb_grad, pos_emb_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_lm_w_grad, lm_w_size * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_ln1_gamma_grad, ln1_gamma_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ln1_beta_grad, ln1_beta_size * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_qkv_w_grad, qkv_w_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_qkv_b_grad, qkv_b_size * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_attn_proj_w_grad, attn_proj_w_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_attn_proj_b_grad, attn_proj_b_size * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_ln2_gamma_grad, ln2_gamma_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ln2_beta_grad, ln2_beta_size * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_mlp_w1_grad, mlp_w1_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mlp_b1_grad, mlp_b1_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mlp_w2_grad, mlp_w2_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mlp_b2_grad, mlp_b2_size * sizeof(float)));

    // Adam m
    CUDA_CHECK(cudaMalloc(&d_token_emb_m, token_emb_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pos_emb_m, pos_emb_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_lm_w_m, lm_w_size * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_ln1_gamma_m, ln1_gamma_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ln1_beta_m, ln1_beta_size * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_qkv_w_m, qkv_w_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_qkv_b_m, qkv_b_size * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_attn_proj_w_m, attn_proj_w_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_attn_proj_b_m, attn_proj_b_size * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_ln2_gamma_m, ln2_gamma_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ln2_beta_m, ln2_beta_size * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_mlp_w1_m, mlp_w1_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mlp_b1_m, mlp_b1_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mlp_w2_m, mlp_w2_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mlp_b2_m, mlp_b2_size * sizeof(float)));

    // Adam v
    CUDA_CHECK(cudaMalloc(&d_token_emb_v, token_emb_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pos_emb_v, pos_emb_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_lm_w_v, lm_w_size * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_ln1_gamma_v, ln1_gamma_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ln1_beta_v, ln1_beta_size * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_qkv_w_v, qkv_w_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_qkv_b_v, qkv_b_size * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_attn_proj_w_v, attn_proj_w_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_attn_proj_b_v, attn_proj_b_size * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_ln2_gamma_v, ln2_gamma_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ln2_beta_v, ln2_beta_size * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_mlp_w1_v, mlp_w1_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mlp_b1_v, mlp_b1_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mlp_w2_v, mlp_w2_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mlp_b2_v, mlp_b2_size * sizeof(float)));

    // forward activations
    CUDA_CHECK(cudaMalloc(&d_hidden, hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ln1_out, ln1_out_size * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_qkv_pre, qkv_pre_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_qkv, qkv_size * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_Q, q_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_K, k_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Vv, v_size * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_scores, scores_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_probs, probs_size * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_attn_heads, attn_heads_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_attn_merge, attn_merge_size * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_h1, h1_size * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_ln2_out, ln2_out_size * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_mlp_pre, mlp_pre_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mlp_act, mlp_act_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mlp_out, mlp_out_size * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_h2, h2_size * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_logits, logits_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_losses, losses_size * sizeof(float)));

    // backward activations
    CUDA_CHECK(cudaMalloc(&d_dlogits, logits_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dh2, h2_size * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_dmlp_out, mlp_out_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dmlp_act, mlp_act_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dmlp_pre, mlp_pre_size * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_dln2_out, ln2_out_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dh1, h1_size * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_dattn_out, attn_merge_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dattn_merge, attn_merge_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dattn_heads, attn_heads_size * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_dprobs, probs_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dscores, scores_size * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_dQ, q_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dK, k_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dVv, v_size * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_dqkv, qkv_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dln1_out, ln1_out_size * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_dhidden, hidden_size * sizeof(float)));

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
        d_ln1_gamma,
        h_ln1_gamma.data(),
        ln1_gamma_size * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        d_ln1_beta,
        h_ln1_beta.data(),
        ln1_beta_size * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        d_qkv_w,
        h_qkv_w.data(),
        qkv_w_size * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        d_qkv_b,
        h_qkv_b.data(),
        qkv_b_size * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        d_attn_proj_w,
        h_attn_proj_w.data(),
        attn_proj_w_size * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        d_attn_proj_b,
        h_attn_proj_b.data(),
        attn_proj_b_size * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        d_ln2_gamma,
        h_ln2_gamma.data(),
        ln2_gamma_size * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        d_ln2_beta,
        h_ln2_beta.data(),
        ln2_beta_size * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        d_mlp_w1,
        h_mlp_w1.data(),
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

    CUDA_CHECK(cudaMemcpy(
        d_lm_w,
        h_lm_w.data(),
        lm_w_size * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    // ----------------------------
    // Zero Adam states
    // ----------------------------

    // m buffers
    zero_kernel<<<(token_emb_size + threads - 1) / threads, threads>>>(d_token_emb_m, token_emb_size);
    zero_kernel<<<(pos_emb_size   + threads - 1) / threads, threads>>>(d_pos_emb_m, pos_emb_size);
    zero_kernel<<<(lm_w_size      + threads - 1) / threads, threads>>>(d_lm_w_m, lm_w_size);

    zero_kernel<<<(ln1_gamma_size + threads - 1) / threads, threads>>>(d_ln1_gamma_m, ln1_gamma_size);
    zero_kernel<<<(ln1_beta_size  + threads - 1) / threads, threads>>>(d_ln1_beta_m, ln1_beta_size);

    zero_kernel<<<(qkv_w_size + threads - 1) / threads, threads>>>(d_qkv_w_m, qkv_w_size);
    zero_kernel<<<(qkv_b_size + threads - 1) / threads, threads>>>(d_qkv_b_m, qkv_b_size);

    zero_kernel<<<(attn_proj_w_size + threads - 1) / threads, threads>>>(d_attn_proj_w_m, attn_proj_w_size);
    zero_kernel<<<(attn_proj_b_size + threads - 1) / threads, threads>>>(d_attn_proj_b_m, attn_proj_b_size);

    zero_kernel<<<(ln2_gamma_size + threads - 1) / threads, threads>>>(d_ln2_gamma_m, ln2_gamma_size);
    zero_kernel<<<(ln2_beta_size  + threads - 1) / threads, threads>>>(d_ln2_beta_m, ln2_beta_size);

    zero_kernel<<<(mlp_w1_size + threads - 1) / threads, threads>>>(d_mlp_w1_m, mlp_w1_size);
    zero_kernel<<<(mlp_b1_size + threads - 1) / threads, threads>>>(d_mlp_b1_m, mlp_b1_size);
    zero_kernel<<<(mlp_w2_size + threads - 1) / threads, threads>>>(d_mlp_w2_m, mlp_w2_size);
    zero_kernel<<<(mlp_b2_size + threads - 1) / threads, threads>>>(d_mlp_b2_m, mlp_b2_size);

    // v buffers
    zero_kernel<<<(token_emb_size + threads - 1) / threads, threads>>>(d_token_emb_v, token_emb_size);
    zero_kernel<<<(pos_emb_size   + threads - 1) / threads, threads>>>(d_pos_emb_v, pos_emb_size);
    zero_kernel<<<(lm_w_size      + threads - 1) / threads, threads>>>(d_lm_w_v, lm_w_size);

    zero_kernel<<<(ln1_gamma_size + threads - 1) / threads, threads>>>(d_ln1_gamma_v, ln1_gamma_size);
    zero_kernel<<<(ln1_beta_size  + threads - 1) / threads, threads>>>(d_ln1_beta_v, ln1_beta_size);

    zero_kernel<<<(qkv_w_size + threads - 1) / threads, threads>>>(d_qkv_w_v, qkv_w_size);
    zero_kernel<<<(qkv_b_size + threads - 1) / threads, threads>>>(d_qkv_b_v, qkv_b_size);

    zero_kernel<<<(attn_proj_w_size + threads - 1) / threads, threads>>>(d_attn_proj_w_v, attn_proj_w_size);
    zero_kernel<<<(attn_proj_b_size + threads - 1) / threads, threads>>>(d_attn_proj_b_v, attn_proj_b_size);

    zero_kernel<<<(ln2_gamma_size + threads - 1) / threads, threads>>>(d_ln2_gamma_v, ln2_gamma_size);
    zero_kernel<<<(ln2_beta_size  + threads - 1) / threads, threads>>>(d_ln2_beta_v, ln2_beta_size);

    zero_kernel<<<(mlp_w1_size + threads - 1) / threads, threads>>>(d_mlp_w1_v, mlp_w1_size);
    zero_kernel<<<(mlp_b1_size + threads - 1) / threads, threads>>>(d_mlp_b1_v, mlp_b1_size);
    zero_kernel<<<(mlp_w2_size + threads - 1) / threads, threads>>>(d_mlp_w2_v, mlp_w2_size);
    zero_kernel<<<(mlp_b2_size + threads - 1) / threads, threads>>>(d_mlp_b2_v, mlp_b2_size);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());


    int max_steps = 2000;
    int eval_every = 100;
    int eval_iters = 10;

    std::cout << "Starting training..." << std::endl;

    // ----------------------------
    // Initial zero gradients
    // ----------------------------

    zero_kernel<<<(token_emb_size + threads - 1) / threads, threads>>>(d_token_emb_grad, token_emb_size);
    zero_kernel<<<(pos_emb_size   + threads - 1) / threads, threads>>>(d_pos_emb_grad, pos_emb_size);
    zero_kernel<<<(lm_w_size      + threads - 1) / threads, threads>>>(d_lm_w_grad, lm_w_size);

    zero_kernel<<<(ln1_gamma_size + threads - 1) / threads, threads>>>(d_ln1_gamma_grad, ln1_gamma_size);
    zero_kernel<<<(ln1_beta_size  + threads - 1) / threads, threads>>>(d_ln1_beta_grad, ln1_beta_size);

    zero_kernel<<<(qkv_w_size + threads - 1) / threads, threads>>>(d_qkv_w_grad, qkv_w_size);
    zero_kernel<<<(qkv_b_size + threads - 1) / threads, threads>>>(d_qkv_b_grad, qkv_b_size);

    zero_kernel<<<(attn_proj_w_size + threads - 1) / threads, threads>>>(d_attn_proj_w_grad, attn_proj_w_size);
    zero_kernel<<<(attn_proj_b_size + threads - 1) / threads, threads>>>(d_attn_proj_b_grad, attn_proj_b_size);

    zero_kernel<<<(ln2_gamma_size + threads - 1) / threads, threads>>>(d_ln2_gamma_grad, ln2_gamma_size);
    zero_kernel<<<(ln2_beta_size  + threads - 1) / threads, threads>>>(d_ln2_beta_grad, ln2_beta_size);

    zero_kernel<<<(mlp_w1_size + threads - 1) / threads, threads>>>(d_mlp_w1_grad, mlp_w1_size);
    zero_kernel<<<(mlp_b1_size + threads - 1) / threads, threads>>>(d_mlp_b1_grad, mlp_b1_size);
    zero_kernel<<<(mlp_w2_size + threads - 1) / threads, threads>>>(d_mlp_w2_grad, mlp_w2_size);
    zero_kernel<<<(mlp_b2_size + threads - 1) / threads, threads>>>(d_mlp_b2_grad, mlp_b2_size);

    zero_kernel<<<(logits_size + threads - 1) / threads, threads>>>(d_dlogits, logits_size);
    zero_kernel<<<(h2_size + threads - 1) / threads, threads>>>(d_dh2, h2_size);

    zero_kernel<<<(mlp_out_size + threads - 1) / threads, threads>>>(d_dmlp_out, mlp_out_size);
    zero_kernel<<<(mlp_act_size + threads - 1) / threads, threads>>>(d_dmlp_act, mlp_act_size);
    zero_kernel<<<(mlp_pre_size + threads - 1) / threads, threads>>>(d_dmlp_pre, mlp_pre_size);

    zero_kernel<<<(ln2_out_size + threads - 1) / threads, threads>>>(d_dln2_out, ln2_out_size);
    zero_kernel<<<(h1_size + threads - 1) / threads, threads>>>(d_dh1, h1_size);

    zero_kernel<<<(attn_merge_size + threads - 1) / threads, threads>>>(d_dattn_out, attn_merge_size);
    zero_kernel<<<(attn_merge_size + threads - 1) / threads, threads>>>(d_dattn_merge, attn_merge_size);
    zero_kernel<<<(attn_heads_size + threads - 1) / threads, threads>>>(d_dattn_heads, attn_heads_size);

    zero_kernel<<<(probs_size  + threads - 1) / threads, threads>>>(d_dprobs, probs_size);
    zero_kernel<<<(scores_size + threads - 1) / threads, threads>>>(d_dscores, scores_size);

    zero_kernel<<<(q_size + threads - 1) / threads, threads>>>(d_dQ, q_size);
    zero_kernel<<<(k_size + threads - 1) / threads, threads>>>(d_dK, k_size);
    zero_kernel<<<(v_size + threads - 1) / threads, threads>>>(d_dVv, v_size);

    zero_kernel<<<(qkv_size + threads - 1) / threads, threads>>>(d_dqkv, qkv_size);
    zero_kernel<<<(ln1_out_size + threads - 1) / threads, threads>>>(d_dln1_out, ln1_out_size);

    zero_kernel<<<(hidden_size + threads - 1) / threads, threads>>>(d_dhidden, hidden_size);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

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

        zero_kernel<<<(token_emb_size + threads - 1) / threads, threads>>>(d_token_emb_grad, token_emb_size);
        zero_kernel<<<(pos_emb_size   + threads - 1) / threads, threads>>>(d_pos_emb_grad, pos_emb_size);
        zero_kernel<<<(lm_w_size      + threads - 1) / threads, threads>>>(d_lm_w_grad, lm_w_size);

        zero_kernel<<<(ln1_gamma_size + threads - 1) / threads, threads>>>(d_ln1_gamma_grad, ln1_gamma_size);
        zero_kernel<<<(ln1_beta_size  + threads - 1) / threads, threads>>>(d_ln1_beta_grad, ln1_beta_size);

        zero_kernel<<<(qkv_w_size + threads - 1) / threads, threads>>>(d_qkv_w_grad, qkv_w_size);
        zero_kernel<<<(qkv_b_size + threads - 1) / threads, threads>>>(d_qkv_b_grad, qkv_b_size);

        zero_kernel<<<(attn_proj_w_size + threads - 1) / threads, threads>>>(d_attn_proj_w_grad, attn_proj_w_size);
        zero_kernel<<<(attn_proj_b_size + threads - 1) / threads, threads>>>(d_attn_proj_b_grad, attn_proj_b_size);

        zero_kernel<<<(ln2_gamma_size + threads - 1) / threads, threads>>>(d_ln2_gamma_grad, ln2_gamma_size);
        zero_kernel<<<(ln2_beta_size  + threads - 1) / threads, threads>>>(d_ln2_beta_grad, ln2_beta_size);

        zero_kernel<<<(mlp_w1_size + threads - 1) / threads, threads>>>(d_mlp_w1_grad, mlp_w1_size);
        zero_kernel<<<(mlp_b1_size + threads - 1) / threads, threads>>>(d_mlp_b1_grad, mlp_b1_size);
        zero_kernel<<<(mlp_w2_size + threads - 1) / threads, threads>>>(d_mlp_w2_grad, mlp_w2_size);
        zero_kernel<<<(mlp_b2_size + threads - 1) / threads, threads>>>(d_mlp_b2_grad, mlp_b2_size);

        // activation gradients
        zero_kernel<<<(logits_size + threads - 1) / threads, threads>>>(d_dlogits, logits_size);
        zero_kernel<<<(h2_size     + threads - 1) / threads, threads>>>(d_dh2, h2_size);

        zero_kernel<<<(mlp_out_size + threads - 1) / threads, threads>>>(d_dmlp_out, mlp_out_size);
        zero_kernel<<<(mlp_act_size + threads - 1) / threads, threads>>>(d_dmlp_act, mlp_act_size);
        zero_kernel<<<(mlp_pre_size + threads - 1) / threads, threads>>>(d_dmlp_pre, mlp_pre_size);

        zero_kernel<<<(ln2_out_size + threads - 1) / threads, threads>>>(d_dln2_out, ln2_out_size);
        zero_kernel<<<(h1_size      + threads - 1) / threads, threads>>>(d_dh1, h1_size);

        zero_kernel<<<(attn_merge_size + threads - 1) / threads, threads>>>(d_dattn_out, attn_merge_size);
        zero_kernel<<<(attn_merge_size + threads - 1) / threads, threads>>>(d_dattn_merge, attn_merge_size);
        zero_kernel<<<(attn_heads_size + threads - 1) / threads, threads>>>(d_dattn_heads, attn_heads_size);

        zero_kernel<<<(probs_size  + threads - 1) / threads, threads>>>(d_dprobs, probs_size);
        zero_kernel<<<(scores_size + threads - 1) / threads, threads>>>(d_dscores, scores_size);

        zero_kernel<<<(q_size + threads - 1) / threads, threads>>>(d_dQ, q_size);
        zero_kernel<<<(k_size + threads - 1) / threads, threads>>>(d_dK, k_size);
        zero_kernel<<<(v_size + threads - 1) / threads, threads>>>(d_dVv, v_size);

        zero_kernel<<<(qkv_size     + threads - 1) / threads, threads>>>(d_dqkv, qkv_size);
        zero_kernel<<<(ln1_out_size + threads - 1) / threads, threads>>>(d_dln1_out, ln1_out_size);

        zero_kernel<<<(hidden_size + threads - 1) / threads, threads>>>(d_dhidden, hidden_size);

        CUDA_CHECK(cudaGetLastError());

        // ====================================================
        // Forward: embedding
        // hidden = token_emb[x] + pos_emb
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
        CUDA_CHECK(cudaDeviceSynchronize());

        // ====================================================
        // Forward: ln1
        // ln1_out = LayerNorm(hidden)
        // ====================================================

        int ln_threads = 256;
        size_t ln_smem = 2 * ln_threads * sizeof(float);

        layernorm_forward_kernel<<<M, ln_threads, ln_smem>>>(
            d_hidden,
            d_ln1_gamma,
            d_ln1_beta,
            d_ln1_out,
            B,
            T,
            C,
            1e-5f
        );

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // ====================================================
        // Forward: qkv projection
        // qkv_pre = ln1_out @ qkv_w
        // ====================================================

        hierarchical_gemm_2x4_kernel<<<gemm_grid_qkv, gemm_block>>>(
            d_ln1_out,
            d_qkv_w,
            d_qkv_pre,
            M,
            C,
            3 * C
        );

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // ====================================================
        // Forward: qkv bias
        // qkv = qkv_pre + qkv_b
        // ====================================================

        add_bias_kernel<<<(M * 3 * C + threads - 1) / threads, threads>>>(
            d_qkv_pre,
            d_qkv_b,
            d_qkv,
            M,
            3 * C
        );

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // ====================================================
        // Forward: split Q/K/V
        // Q, K, V shape: [B, NH, T, HS]
        // ====================================================

        split_qkv_kernel<<<(M * 3 * C + threads - 1) / threads, threads>>>(
            d_qkv,
            d_Q,
            d_K,
            d_Vv,
            B,
            T,
            NH,
            HS
        );

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // ====================================================
        // Forward: attention scores
        // scores = Q @ K^T / sqrt(HS)
        // scores shape: [B, NH, T, T]
        // ====================================================

        attention_scores_kernel<<<grid_scores, block2d>>>(
            d_Q,
            d_K,
            d_scores,
            B,
            NH,
            T,
            HS
        );

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // ====================================================
        // Forward: causal mask
        // scores[j > i] = -1e20f
        // IMPORTANT: causal_mask_kernel expects 3D grid.
        // ====================================================

        causal_mask_kernel<<<grid_scores, block2d>>>(
            d_scores,
            B,
            NH,
            T
        );

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // ====================================================
        // Forward: masked softmax
        // IMPORTANT: masked_softmax_kernel uses dynamic shared memory.
        // ====================================================

        size_t softmax_smem = 2 * threads * sizeof(float);

        masked_softmax_kernel<<<B * NH * T, threads, softmax_smem>>>(
            d_scores,
            d_probs,
            B,
            NH,
            T
        );

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // ====================================================
        // Forward: attention value aggregation
        // attn_heads = probs @ V
        // ====================================================

        attention_value_kernel<<<grid_attn_value, block2d>>>(
            d_probs,
            d_Vv,
            d_attn_heads,
            B,
            NH,
            T,
            HS
        );

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // ====================================================
        // Forward: merge heads
        // attn_merge shape: [M, C]
        // ====================================================

        merge_heads_kernel<<<(M * C + threads - 1) / threads, threads>>>(
            d_attn_heads,
            d_attn_merge,
            B,
            T,
            NH,
            HS
        );

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // ====================================================
        // Forward: attention output projection + residual
        // h1 = attn_merge @ attn_proj_w + attn_proj_b + hidden
        // ====================================================

        matmul_bias_residual_kernel<<<
            dim3((C + 15) / 16, (M + 15) / 16),
            dim3(16, 16)
        >>>(
            d_attn_merge,
            d_attn_proj_w,
            d_attn_proj_b,
            d_hidden,
            d_h1,
            M,
            C,
            C
        );

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // ====================================================
        // Forward: ln2
        // ln2_out = LayerNorm(h1)
        // ====================================================

        layernorm_forward_kernel<<<M, ln_threads, ln_smem>>>(
            d_h1,
            d_ln2_gamma,
            d_ln2_beta,
            d_ln2_out,
            B,
            T,
            C,
            1e-5f
        );

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // ====================================================
        // Forward: MLP first projection
        // mlp_pre = ln2_out @ W1
        // ====================================================

        hierarchical_gemm_2x4_kernel<<<gemm_grid_mlp1, gemm_block>>>(
            d_ln2_out,
            d_mlp_w1,
            d_mlp_pre,
            M,
            C,
            H
        );

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

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
        CUDA_CHECK(cudaDeviceSynchronize());

        // ====================================================
        // Forward: MLP second projection + residual
        // h2 = mlp_act @ W2 + b2 + h1
        // ====================================================

        matmul_bias_residual_kernel<<<
            dim3((C + 15) / 16, (M + 15) / 16),
            dim3(16, 16)
        >>>(
            d_mlp_act,
            d_mlp_w2,
            d_mlp_b2,
            d_h1,
            d_h2,
            M,
            H,
            C
        );

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // ====================================================
        // Forward: LM head
        // logits = h2 @ lm_w
        // ====================================================

        hierarchical_gemm_2x4_kernel<<<gemm_grid_lm, gemm_block>>>(
            d_h2,
            d_lm_w,
            d_logits,
            M,
            C,
            V
        );

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

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

        // ====================================================
        // Backward: cross entropy
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
        CUDA_CHECK(cudaDeviceSynchronize());

        // ====================================================
        // Backward: LM head
        // logits = h2 @ lm_w
        // ====================================================

        linear_backward_dA_kernel<<<grid_lm_dA, block2d>>>(
            d_dlogits,
            d_lm_w,
            d_dh2,
            M,
            C,
            V
        );

        CUDA_CHECK(cudaGetLastError());

        linear_backward_dW_kernel<<<grid_lm_dW, block2d>>>(
            d_h2,
            d_dlogits,
            d_lm_w_grad,
            M,
            C,
            V
        );

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // ====================================================
        // Backward: MLP residual
        // h2 = mlp_out + h1
        // ====================================================

        residual_add_backward_kernel<<<(M * C + threads - 1) / threads, threads>>>(
            d_dh2,
            d_dmlp_out,
            d_dh1,
            M * C
        );

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // b2 gradient
        linear_backward_bias_kernel<<<(C + threads - 1) / threads, threads>>>(
            d_dh2,
            d_mlp_b2_grad,
            M,
            C
        );

        CUDA_CHECK(cudaGetLastError());

        // ====================================================
        // Backward: MLP second projection
        // mlp_out = mlp_act @ W2
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
        CUDA_CHECK(cudaDeviceSynchronize());

        // ====================================================
        // Backward: GELU + b1
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
        CUDA_CHECK(cudaDeviceSynchronize());

        // ====================================================
        // Backward: MLP first projection
        // mlp_pre = ln2_out @ W1
        // ====================================================

        linear_backward_dA_kernel<<<grid_mlp1_dA, block2d>>>(
            d_dmlp_pre,
            d_mlp_w1,
            d_dln2_out,
            M,
            C,
            H
        );

        CUDA_CHECK(cudaGetLastError());

        linear_backward_dW_kernel<<<grid_mlp1_dW, block2d>>>(
            d_ln2_out,
            d_dmlp_pre,
            d_mlp_w1_grad,
            M,
            C,
            H
        );

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // ====================================================
        // Backward: ln2
        // This accumulates into d_dh1.
        // ====================================================

        size_t ln_bwd_smem = 4 * ln_threads * sizeof(float);

        layernorm_backward_kernel<<<M, ln_threads, ln_bwd_smem>>>(
            d_h1,
            d_ln2_gamma,
            d_dln2_out,
            d_dh1,
            d_ln2_gamma_grad,
            d_ln2_beta_grad,
            M,
            C,
            1e-5f
        );

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // ====================================================
        // Backward: attention residual
        // h1 = attn_out + hidden
        // ====================================================

        residual_add_backward_kernel<<<(M * C + threads - 1) / threads, threads>>>(
            d_dh1,
            d_dattn_out,
            d_dhidden,
            M * C
        );

        CUDA_CHECK(cudaGetLastError());

        // attention output bias gradient
        linear_backward_bias_kernel<<<(C + threads - 1) / threads, threads>>>(
            d_dh1,
            d_attn_proj_b_grad,
            M,
            C
        );

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // ====================================================
        // Backward: attention output projection
        // attn_out = attn_merge @ attn_proj_w
        // ====================================================

        linear_backward_dA_kernel<<<grid_d_attn_proj_A, block2d>>>(
            d_dattn_out,
            d_attn_proj_w,
            d_dattn_merge,
            M,
            C,
            C
        );

        CUDA_CHECK(cudaGetLastError());

        linear_backward_dW_kernel<<<grid_d_attn_proj_W, block2d>>>(
            d_attn_merge,
            d_dattn_out,
            d_attn_proj_w_grad,
            M,
            C,
            C
        );

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // ====================================================
        // Backward: merge heads
        // ====================================================

        merge_heads_backward_kernel<<<(M * C + threads - 1) / threads, threads>>>(
            d_dattn_merge,
            d_dattn_heads,
            B,
            T,
            NH,
            HS
        );

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // ====================================================
        // Backward: attention value
        // attn_heads = probs @ V
        // Produces dprobs and dV.
        // ====================================================

        attention_value_backward_kernel<<<grid_attn_value, block2d>>>(
            d_probs,
            d_Vv,
            d_dattn_heads,
            d_dprobs,
            d_dVv,
            B,
            NH,
            T,
            HS
        );

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // ====================================================
        // Backward: masked softmax
        // ====================================================

        size_t softmax_bwd_smem = 2 * threads * sizeof(float);

        masked_softmax_backward_kernel<<<B * NH * T, threads, softmax_bwd_smem>>>(
            d_probs,
            d_dprobs,
            d_dscores,
            B,
            NH,
            T
        );

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // ====================================================
        // Backward: attention scores wrt Q
        // scores = Q @ K^T / sqrt(HS)
        // ====================================================

        attention_scores_backward_dQ_kernel<<<grid_attn_value, block2d>>>(
            d_dscores,
            d_K,
            d_dQ,
            B,
            NH,
            T,
            HS
        );


        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // ====================================================
        // Backward: attention scores wrt K
        // scores = Q @ K^T / sqrt(HS)
        // ====================================================

        attention_scores_backward_dK_kernel<<<grid_attn_value, block2d>>>(
            d_dscores,
            d_Q,
            d_dK,
            B,
            NH,
            T,
            HS
        );

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // ====================================================
        // Backward: combine dQ, dK, dV into d_qkv
        // ====================================================

        split_qkv_backward_kernel<<<(M * 3 * C + threads - 1) / threads, threads>>>(
            d_dQ,
            d_dK,
            d_dVv,
            d_dqkv,
            B,
            T,
            NH,
            HS
        );

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // ====================================================
        // Backward: qkv bias
        // qkv = qkv_pre + qkv_b
        // Therefore dqkv_pre = dqkv, and qkv_b_grad = sum(dqkv)
        // ====================================================

        linear_backward_bias_kernel<<<(3 * C + threads - 1) / threads, threads>>>(
            d_dqkv,
            d_qkv_b_grad,
            M,
            3 * C
        );

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // ====================================================
        // Backward: qkv projection
        // qkv_pre = ln1_out @ qkv_w
        // ====================================================

        linear_backward_dA_kernel<<<grid_d_qkv_A, block2d>>>(
            d_dqkv,
            d_qkv_w,
            d_dln1_out,
            M,
            C,
            3 * C
        );

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        linear_backward_dW_kernel<<<grid_d_qkv_W, block2d>>>(
            d_ln1_out,
            d_dqkv,
            d_qkv_w_grad,
            M,
            C,
            3 * C
        );

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // ====================================================
        // Backward: ln1
        // This accumulates into d_dhidden.
        // ====================================================

        layernorm_backward_kernel<<<M, ln_threads, ln_bwd_smem>>>(
            d_hidden,
            d_ln1_gamma,
            d_dln1_out,
            d_dhidden,
            d_ln1_gamma_grad,
            d_ln1_beta_grad,
            M,
            C,
            1e-5f
        );

        CUDA_CHECK(cudaGetLastError());

        // ====================================================
        // Backward: embedding
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

        // ----------------------------
        // Embeddings
        // ----------------------------

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

        // ----------------------------
        // LayerNorm 1
        // Usually no weight decay for LayerNorm parameters
        // ----------------------------

        adamw_update_kernel<<<(ln1_gamma_size + threads - 1) / threads, threads>>>(
            d_ln1_gamma,
            d_ln1_gamma_grad,
            d_ln1_gamma_m,
            d_ln1_gamma_v,
            ln1_gamma_size,
            LR,
            BETA1,
            BETA2,
            ADAM_EPS,
            0.0f,
            step
        );

        adamw_update_kernel<<<(ln1_beta_size + threads - 1) / threads, threads>>>(
            d_ln1_beta,
            d_ln1_beta_grad,
            d_ln1_beta_m,
            d_ln1_beta_v,
            ln1_beta_size,
            LR,
            BETA1,
            BETA2,
            ADAM_EPS,
            0.0f,
            step
        );

        // ----------------------------
        // QKV projection
        // Weight uses decay, bias does not
        // ----------------------------

        adamw_update_kernel<<<(qkv_w_size + threads - 1) / threads, threads>>>(
            d_qkv_w,
            d_qkv_w_grad,
            d_qkv_w_m,
            d_qkv_w_v,
            qkv_w_size,
            LR,
            BETA1,
            BETA2,
            ADAM_EPS,
            WEIGHT_DECAY,
            step
        );

        adamw_update_kernel<<<(qkv_b_size + threads - 1) / threads, threads>>>(
            d_qkv_b,
            d_qkv_b_grad,
            d_qkv_b_m,
            d_qkv_b_v,
            qkv_b_size,
            LR,
            BETA1,
            BETA2,
            ADAM_EPS,
            0.0f,
            step
        );

        // ----------------------------
        // Attention output projection
        // Weight uses decay, bias does not
        // ----------------------------

        adamw_update_kernel<<<(attn_proj_w_size + threads - 1) / threads, threads>>>(
            d_attn_proj_w,
            d_attn_proj_w_grad,
            d_attn_proj_w_m,
            d_attn_proj_w_v,
            attn_proj_w_size,
            LR,
            BETA1,
            BETA2,
            ADAM_EPS,
            WEIGHT_DECAY,
            step
        );

        adamw_update_kernel<<<(attn_proj_b_size + threads - 1) / threads, threads>>>(
            d_attn_proj_b,
            d_attn_proj_b_grad,
            d_attn_proj_b_m,
            d_attn_proj_b_v,
            attn_proj_b_size,
            LR,
            BETA1,
            BETA2,
            ADAM_EPS,
            0.0f,
            step
        );

        // ----------------------------
        // LayerNorm 2
        // Usually no weight decay for LayerNorm parameters
        // ----------------------------

        adamw_update_kernel<<<(ln2_gamma_size + threads - 1) / threads, threads>>>(
            d_ln2_gamma,
            d_ln2_gamma_grad,
            d_ln2_gamma_m,
            d_ln2_gamma_v,
            ln2_gamma_size,
            LR,
            BETA1,
            BETA2,
            ADAM_EPS,
            0.0f,
            step
        );

        adamw_update_kernel<<<(ln2_beta_size + threads - 1) / threads, threads>>>(
            d_ln2_beta,
            d_ln2_beta_grad,
            d_ln2_beta_m,
            d_ln2_beta_v,
            ln2_beta_size,
            LR,
            BETA1,
            BETA2,
            ADAM_EPS,
            0.0f,
            step
        );

        // ----------------------------
        // MLP first projection
        // Weight uses decay, bias does not
        // ----------------------------

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

        // ----------------------------
        // MLP second projection
        // Weight uses decay, bias does not
        // ----------------------------

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

        // ----------------------------
        // LM head
        // ----------------------------

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

                d_ln1_gamma,
                d_ln1_beta,

                d_qkv_w,
                d_qkv_b,

                d_attn_proj_w,
                d_attn_proj_b,

                d_ln2_gamma,
                d_ln2_beta,

                d_mlp_w1,
                d_mlp_b1,
                d_mlp_w2,
                d_mlp_b2,

                d_lm_w,

                d_hidden,
                d_ln1_out,

                d_qkv_pre,
                d_qkv,

                d_Q,
                d_K,
                d_Vv,

                d_scores,
                d_probs,
                d_attn_heads,
                d_attn_merge,

                d_h1,

                d_ln2_out,
                d_mlp_pre,
                d_mlp_act,
                d_mlp_out,

                d_h2,

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

    cudaFree(d_qkv_pre);
    cudaFree(d_qkv);

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_Vv);

    cudaFree(d_scores);
    cudaFree(d_probs);

    cudaFree(d_attn_heads);
    cudaFree(d_attn_merge);

    cudaFree(d_dattn_out);
    cudaFree(d_dattn_merge);
    cudaFree(d_dattn_heads);

    cudaFree(d_dprobs);
    cudaFree(d_dscores);

    cudaFree(d_dQ);
    cudaFree(d_dK);
    cudaFree(d_dVv);

    cudaFree(d_dqkv);

    std::cout << "Training finished." << std::endl;

    return 0;
}