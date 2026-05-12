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
// 10b. LayerNorm (Vectorized float4)
// ============================================================

__global__ void layernorm_forward_kernel_float4(
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

    int token_idx = blockIdx.x;
    int tid = threadIdx.x;

    if (token_idx >= B * T) return;

    // Use float4 for vectorized memory access
    const float4* x_token_f4 = reinterpret_cast<const float4*>(x + token_idx * C);
    float4* out_token_f4 = reinterpret_cast<float4*>(out + token_idx * C);
    const float4* gamma_f4 = reinterpret_cast<const float4*>(gamma);
    const float4* beta_f4 = reinterpret_cast<const float4*>(beta);

    int C4 = C / 4; // Assuming C is a multiple of 4

    float sum = 0.0f;
    float sqsum = 0.0f;

    for (int c = tid; c < C4; c += blockDim.x) {
        float4 v = x_token_f4[c];
        sum += v.x + v.y + v.z + v.w;
        sqsum += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
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

    for (int c = tid; c < C4; c += blockDim.x) {
        float4 v = x_token_f4[c];
        float4 g = gamma_f4[c];
        float4 b = beta_f4[c];
        
        float4 norm_v;
        norm_v.x = (v.x - mean) * inv_std * g.x + b.x;
        norm_v.y = (v.y - mean) * inv_std * g.y + b.y;
        norm_v.z = (v.z - mean) * inv_std * g.z + b.z;
        norm_v.w = (v.w - mean) * inv_std * g.w + b.w;
        
        out_token_f4[c] = norm_v;
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
