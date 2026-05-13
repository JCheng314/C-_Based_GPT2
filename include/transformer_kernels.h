#ifndef TRANSFORMER_KERNELS_H
#define TRANSFORMER_KERNELS_H

#include <cuda_runtime.h>
#include <stdio.h>

#define CHECK_CUDA(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
                   cudaGetErrorString(err));                               \
        }                                                                  \
    } while (0)

// GEMM Tile Sizes
#define BLOCK_M 64
#define BLOCK_N 64
#define BLOCK_K 8
#define THREAD_TILE_M 4
#define THREAD_TILE_N 4
#define THREADS_X (BLOCK_N / THREAD_TILE_N)
#define THREADS_Y (BLOCK_M / THREAD_TILE_M)

// ============================================================
// Utility: GELU
// ============================================================
__device__ __forceinline__ float gelu(float x) {
    const float sqrt_2_over_pi = 0.7978845608028654f;
    return 0.5f * x * (1.0f + tanhf(sqrt_2_over_pi * (x + 0.044715f * x * x * x)));
}

// ============================================================
// Kernel Signatures
// ============================================================

// linear_layers.cu
__global__ void embedding_forward_kernel(
    const int* tokens, const float* token_emb, const float* pos_emb,
    float* out, int B, int T, int C);

__global__ void hierarchical_gemm_2x4_kernel(
    const float* __restrict__ A, const float* __restrict__ W,
    float* __restrict__ C, int M, int K, int N);

__global__ void gelu_forward_kernel(const float* x, float* out, int N);

__global__ void bias_gelu_kernel(const float* x, const float* bias, float* out, int M, int N);

__global__ void matmul_bias_gelu_kernel(
    const float* A, const float* W, const float* bias,
    float* out, int M, int K, int N);

__global__ void residual_add_kernel(const float* x, const float* residual, float* out, int N);

__global__ void matmul_bias_residual_kernel(
    const float* A, const float* W, const float* bias, const float* residual,
    float* out, int M, int K, int N);

// fused_layernorm.cu
__global__ void layernorm_forward_kernel(
    const float* x, const float* gamma, const float* beta,
    float* out, int B, int T, int C, float eps);

__global__ void layernorm_forward_kernel_float4(
    const float* x, const float* gamma, const float* beta,
    float* out, int B, int T, int C, float eps);

__global__ void residual_layernorm_kernel(
    const float* x, const float* residual, const float* gamma, const float* beta,
    float* out, int B, int T, int C, float eps);

// flash_attention.cu
__global__ void split_qkv_kernel(
    const float* qkv, float* Q, float* K, float* V,
    int B, int T, int NH, int HS);

__global__ void merge_heads_kernel(
    const float* attn_out, float* out,
    int B, int T, int NH, int HS);

__global__ void attention_scores_kernel(
    const float* Q, const float* K, float* scores,
    int B, int NH, int T, int HS);

__global__ void causal_mask_kernel(float* scores, int B, int NH, int T);

__global__ void masked_softmax_kernel(const float* scores, float* probs, int B, int NH, int T);

__global__ void attention_value_kernel(
    const float* probs, const float* V, float* out,
    int B, int NH, int T, int HS);

__global__ void simple_fused_attention_kernel(
    const float* Q, const float* K, const float* V, float* out,
    int B, int NH, int T, int HS);

__global__ void tiled_flash_attention_kernel(
    const float* Q, const float* K, const float* V, float* out,
    int B, int NH, int T, int HS);

// ============================================================
// Backward kernels: backward_kernels.cu
// ============================================================

__global__ void zero_kernel(float* x, int N);

__global__ void cross_entropy_forward_kernel(
    const float* logits,
    const int* targets,
    float* losses,
    int M,
    int V
);

__global__ void cross_entropy_backward_kernel(
    const float* logits,
    const int* targets,
    float* dlogits,
    int M,
    int V
);

__global__ void linear_backward_dA_kernel(
    const float* dC,
    const float* W,
    float* dA,
    int M,
    int K,
    int N
);

__global__ void linear_backward_dW_kernel(
    const float* A,
    const float* dC,
    float* dW,
    int M,
    int K,
    int N
);

__global__ void linear_backward_bias_kernel(
    const float* dC,
    float* dbias,
    int M,
    int N
);

__global__ void embedding_backward_kernel(
    const int* tokens,
    const float* dout,
    float* dtoken_emb,
    float* dpos_emb,
    int B,
    int T,
    int C
);

__global__ void adamw_update_kernel(
    float* param,
    const float* grad,
    float* m,
    float* v,
    int N,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    int step
);

#endif // TRANSFORMER_KERNELS_H
