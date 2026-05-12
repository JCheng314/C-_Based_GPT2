#include "../include/transformer_kernels.h"
#include <float.h>

// ============================================================
// 12. Split QKV
// qkv: [B, T, 3C]
// Q/K/V: [B, NH, T, HS]
// C = NH * HS
// ============================================================

__global__ void split_qkv_kernel(
    const float* qkv,
    float* Q,
    float* K,
    float* V,
    int B, int T, int NH, int HS
) {
    int C = NH * HS;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * T * C;

    if (idx >= total) return;

    int hs = idx % HS;
    int h = (idx / HS) % NH;
    int t = (idx / (HS * NH)) % T;
    int b = idx / (T * NH * HS);

    int c = h * HS + hs;

    int qkv_base = (b * T + t) * (3 * C);

    int out_idx = ((b * NH + h) * T + t) * HS + hs;

    Q[out_idx] = qkv[qkv_base + c];
    K[out_idx] = qkv[qkv_base + C + c];
    V[out_idx] = qkv[qkv_base + 2 * C + c];
}


// ============================================================
// 13. Merge heads
// attn_out: [B, NH, T, HS]
// out:      [B, T, C]
// ============================================================

__global__ void merge_heads_kernel(
    const float* attn_out,
    float* out,
    int B, int T, int NH, int HS
) {
    int C = NH * HS;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * T * C;

    if (idx >= total) return;

    int c = idx % C;
    int t = (idx / C) % T;
    int b = idx / (T * C);

    int h = c / HS;
    int hs = c % HS;

    int in_idx = ((b * NH + h) * T + t) * HS + hs;

    out[idx] = attn_out[in_idx];
}


// ============================================================
// 14. Attention scores
// scores = Q @ K^T / sqrt(HS)
// Q:      [B, NH, T, HS]
// K:      [B, NH, T, HS]
// scores: [B, NH, T, T]
// ============================================================

__global__ void attention_scores_kernel(
    const float* Q,
    const float* K,
    float* scores,
    int B, int NH, int T, int HS
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;  // key position
    int i = blockIdx.y * blockDim.y + threadIdx.y;  // query position
    int bh = blockIdx.z;                            // combined batch/head

    if (i >= T || j >= T) return;

    int b = bh / NH;
    int h = bh % NH;

    float acc = 0.0f;

    for (int d = 0; d < HS; ++d) {
        int q_idx = ((b * NH + h) * T + i) * HS + d;
        int k_idx = ((b * NH + h) * T + j) * HS + d;
        acc += Q[q_idx] * K[k_idx];
    }

    float scale = rsqrtf((float)HS);

    int score_idx = ((b * NH + h) * T + i) * T + j;
    scores[score_idx] = acc * scale;
}


// ============================================================
// 15. Causal mask
// scores[b,h,i,j] = -inf if j > i
// ============================================================

__global__ void causal_mask_kernel(
    float* scores,
    int B, int NH, int T
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int bh = blockIdx.z;

    if (i >= T || j >= T) return;

    if (j > i) {
        int idx = (bh * T + i) * T + j;
        scores[idx] = -1e20f;
    }
}


// ============================================================
// 16. Masked softmax
// scores: [B, NH, T, T]
// probs:  [B, NH, T, T]
//
// One block handles one attention row.
// blockIdx.x indexes B * NH * T rows.
// ============================================================

__global__ void masked_softmax_kernel(
    const float* scores,
    float* probs,
    int B, int NH, int T
) {
    extern __shared__ float shared[];

    float* s_max = shared;
    float* s_sum = shared + blockDim.x;

    int row_idx = blockIdx.x;
    int tid = threadIdx.x;

    int total_rows = B * NH * T;
    if (row_idx >= total_rows) return;

    int i = row_idx % T;  // query position

    const float* row_scores = scores + row_idx * T;
    float* row_probs = probs + row_idx * T;

    float local_max = -FLT_MAX;

    for (int j = tid; j < T; j += blockDim.x) {
        float v = (j <= i) ? row_scores[j] : -1e20f;
        local_max = fmaxf(local_max, v);
    }

    s_max[tid] = local_max;

    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_max[tid] = fmaxf(s_max[tid], s_max[tid + stride]);
        }
        __syncthreads();
    }

    float max_val = s_max[0];

    float local_sum = 0.0f;

    for (int j = tid; j < T; j += blockDim.x) {
        float v = (j <= i) ? expf(row_scores[j] - max_val) : 0.0f;
        row_probs[j] = v;
        local_sum += v;
    }

    s_sum[tid] = local_sum;

    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
        }
        __syncthreads();
    }

    float denom = s_sum[0] + 1e-9f;

    for (int j = tid; j < T; j += blockDim.x) {
        row_probs[j] /= denom;
    }
}


// ============================================================
// 17. Attention value
// out = probs @ V
// probs: [B, NH, T, T]
// V:     [B, NH, T, HS]
// out:   [B, NH, T, HS]
// ============================================================

__global__ void attention_value_kernel(
    const float* probs,
    const float* V,
    float* out,
    int B, int NH, int T, int HS
) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int bh = blockIdx.z;

    if (i >= T || d >= HS) return;

    int b = bh / NH;
    int h = bh % NH;

    float acc = 0.0f;

    for (int j = 0; j < T; ++j) {
        int p_idx = ((b * NH + h) * T + i) * T + j;
        int v_idx = ((b * NH + h) * T + j) * HS + d;
        acc += probs[p_idx] * V[v_idx];
    }

    int out_idx = ((b * NH + h) * T + i) * HS + d;
    out[out_idx] = acc;
}


// ============================================================
// 18. Simple fused attention kernel
//
// This is a simple correctness-oriented fused attention kernel.
// It does not materialize scores/probs.
// One block handles one (b, h, query position i).
//
// Q/K/V: [B, NH, T, HS]
// out:   [B, NH, T, HS]
//
// Limitation:
// This version assumes T <= blockDim.x for simple reduction.
// Good for small T experiments.
// For larger T, use tiled FlashAttention-style implementation.
// ============================================================

__global__ void simple_fused_attention_kernel(
    const float* Q,
    const float* K,
    const float* V,
    float* out,
    int B, int NH, int T, int HS
) {
    extern __shared__ float shared[];

    float* score_buf = shared;
    float* exp_buf = shared + T;

    int row = blockIdx.x;  // B * NH * T rows
    int tid = threadIdx.x;

    int i = row % T;
    int h = (row / T) % NH;
    int b = row / (T * NH);

    if (b >= B) return;

    if (tid < T) {
        int j = tid;

        if (j <= i) {
            float acc = 0.0f;
            for (int d = 0; d < HS; ++d) {
                int q_idx = ((b * NH + h) * T + i) * HS + d;
                int k_idx = ((b * NH + h) * T + j) * HS + d;
                acc += Q[q_idx] * K[k_idx];
            }
            score_buf[j] = acc * rsqrtf((float)HS);
        } else {
            score_buf[j] = -1e20f;
        }
    }

    __syncthreads();

    float max_val = -FLT_MAX;

    if (tid == 0) {
        for (int j = 0; j < T; ++j) {
            max_val = fmaxf(max_val, score_buf[j]);
        }

        float denom = 0.0f;
        for (int j = 0; j < T; ++j) {
            exp_buf[j] = expf(score_buf[j] - max_val);
            denom += exp_buf[j];
        }

        for (int j = 0; j < T; ++j) {
            exp_buf[j] /= denom;
        }
    }

    __syncthreads();

    for (int d = tid; d < HS; d += blockDim.x) {
        float acc = 0.0f;

        for (int j = 0; j < T; ++j) {
            int v_idx = ((b * NH + h) * T + j) * HS + d;
            acc += exp_buf[j] * V[v_idx];
        }

        int out_idx = ((b * NH + h) * T + i) * HS + d;
        out[out_idx] = acc;
    }
}
