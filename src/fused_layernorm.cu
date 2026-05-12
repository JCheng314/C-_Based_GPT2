#include "../include/transformer_kernels.h"

// ============================================================
// 10. LayerNorm
// x:     [B, T, C]
// gamma: [C]
// beta:  [C]
// out:   [B, T, C]
//
// One CUDA block handles one token vector x[b, t, :].
// blockDim.x should be power of two, e.g. 256 or 1024.
// ============================================================

__global__ void layernorm_forward_kernel(
    const float* x,
    const float* gamma,
    const float* beta,
    float* out,
    int B, int T, int C,
    float eps
) {
    extern __shared__ float shared[];

    float* s_sum = shared;
    float* s_sqsum = shared + blockDim.x;

    int token_idx = blockIdx.x;  // from 0 to B*T-1
    int tid = threadIdx.x;

    if (token_idx >= B * T) return;

    const float* x_token = x + token_idx * C;
    float* out_token = out + token_idx * C;

    float sum = 0.0f;
    float sqsum = 0.0f;

    for (int c = tid; c < C; c += blockDim.x) {
        float v = x_token[c];
        sum += v;
        sqsum += v * v;
    }

    s_sum[tid] = sum;
    s_sqsum[tid] = sqsum;

    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
            s_sqsum[tid] += s_sqsum[tid + stride];
        }
        __syncthreads();
    }

    float mean = s_sum[0] / C;
    float var = s_sqsum[0] / C - mean * mean;
    float inv_std = rsqrtf(var + eps);

    for (int c = tid; c < C; c += blockDim.x) {
        float normalized = (x_token[c] - mean) * inv_std;
        out_token[c] = normalized * gamma[c] + beta[c];
    }
}


// ============================================================
// 11. Residual + LayerNorm
// z = x + residual
// out = LayerNorm(z)
// ============================================================

__global__ void residual_layernorm_kernel(
    const float* x,
    const float* residual,
    const float* gamma,
    const float* beta,
    float* out,
    int B, int T, int C,
    float eps
) {
    extern __shared__ float shared[];

    float* s_sum = shared;
    float* s_sqsum = shared + blockDim.x;

    int token_idx = blockIdx.x;
    int tid = threadIdx.x;

    if (token_idx >= B * T) return;

    int base = token_idx * C;

    float sum = 0.0f;
    float sqsum = 0.0f;

    for (int c = tid; c < C; c += blockDim.x) {
        float v = x[base + c] + residual[base + c];
        sum += v;
        sqsum += v * v;
    }

    s_sum[tid] = sum;
    s_sqsum[tid] = sqsum;

    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
            s_sqsum[tid] += s_sqsum[tid + stride];
        }
        __syncthreads();
    }

    float mean = s_sum[0] / C;
    float var = s_sqsum[0] / C - mean * mean;
    float inv_std = rsqrtf(var + eps);

    for (int c = tid; c < C; c += blockDim.x) {
        float v = x[base + c] + residual[base + c];
        float normalized = (v - mean) * inv_std;
        out[base + c] = normalized * gamma[c] + beta[c];
    }
}
