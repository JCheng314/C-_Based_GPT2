// gpt2_kernels.cu
#include <cuda_runtime.h>
#include <math.h>
#include <float.h>

#define CHECK_CUDA(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
                   cudaGetErrorString(err));                               \
        }                                                                  \
    } while (0)


// ============================================================
// Utility: GELU
// ============================================================

__device__ __forceinline__ float gelu(float x) {
    const float sqrt_2_over_pi = 0.7978845608028654f;
    return 0.5f * x * (1.0f + tanhf(sqrt_2_over_pi * (x + 0.044715f * x * x * x)));
}


// ============================================================
// 1. Embedding: token embedding + positional embedding
// tokens:    [B, T]
// token_emb: [V, C]
// pos_emb:   [T, C]
// out:       [B, T, C]
// ============================================================

__global__ void embedding_forward_kernel(
    const int* tokens,
    const float* token_emb,
    const float* pos_emb,
    float* out,
    int B, int T, int C
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * T * C;

    if (idx >= total) return;

    int c = idx % C;
    int t = (idx / C) % T;
    int b = idx / (T * C);

    int token_id = tokens[b * T + t];

    out[idx] = token_emb[token_id * C + c] + pos_emb[t * C + c];
}
// ============================================================
// 2. GEMM: matrix multiplication  
// A: [M, K]
// W: [K, N]
// C: [M, N]
// ============================================================s

__global__ void hierarchical_gemm_2x4_kernel(
    const float* __restrict__ A,
    const float* __restrict__ W,
    float* __restrict__ C,
    int M, int K, int N
) {
    __shared__ float As[BLOCK_M][BLOCK_K];
    __shared__ float Ws[BLOCK_K][BLOCK_N];

    int tx = threadIdx.x;  // 0..15
    int ty = threadIdx.y;  // 0..15

    int block_row = blockIdx.y * BLOCK_M;
    int block_col = blockIdx.x * BLOCK_N;

    int thread_row_base = ty * THREAD_TILE_M;
    int thread_col_base = tx * THREAD_TILE_N;

    float acc[THREAD_TILE_M][THREAD_TILE_N];

    #pragma unroll
    for (int i = 0; i < THREAD_TILE_M; ++i) {
        #pragma unroll
        for (int j = 0; j < THREAD_TILE_N; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    for (int k0 = 0; k0 < K; k0 += BLOCK_K) {
        int linear_tid = ty * THREADS_X + tx;

        int num_a_elements = BLOCK_M * BLOCK_K;
        int num_b_elements = BLOCK_K * BLOCK_N;

        for (int idx = linear_tid; idx < num_a_elements; idx += THREADS_X * THREADS_Y) {
            int smem_row = idx / BLOCK_K;
            int smem_col = idx % BLOCK_K;

            int global_row = block_row + smem_row;
            int global_col = k0 + smem_col;

            if (global_row < M && global_col < K) {
                As[smem_row][smem_col] = A[global_row * K + global_col];
            } else {
                As[smem_row][smem_col] = 0.0f;
            }
        }

        for (int idx = linear_tid; idx < num_b_elements; idx += THREADS_X * THREADS_Y) {
            int smem_row = idx / BLOCK_N;
            int smem_col = idx % BLOCK_N;

            int global_row = k0 + smem_row;
            int global_col = block_col + smem_col;

            if (global_row < K && global_col < N) {
                Ws[smem_row][smem_col] = W[global_row * N + global_col];
            } else {
                Ws[smem_row][smem_col] = 0.0f;
            }
        }

        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BLOCK_K; ++kk) {
            float a_frag[THREAD_TILE_M];
            float b_frag[THREAD_TILE_N];

            #pragma unroll
            for (int i = 0; i < THREAD_TILE_M; ++i) {
                a_frag[i] = As[thread_row_base + i][kk];
            }

            #pragma unroll
            for (int j = 0; j < THREAD_TILE_N; ++j) {
                b_frag[j] = Ws[kk][thread_col_base + j];
            }

            #pragma unroll
            for (int i = 0; i < THREAD_TILE_M; ++i) {
                #pragma unroll
                for (int j = 0; j < THREAD_TILE_N; ++j) {
                    acc[i][j] += a_frag[i] * b_frag[j];
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < THREAD_TILE_M; ++i) {
        #pragma unroll
        for (int j = 0; j < THREAD_TILE_N; ++j) {
            int global_row = block_row + thread_row_base + i;
            int global_col = block_col + thread_col_base + j;

            if (global_row < M && global_col < N) {
                C[global_row * N + global_col] = acc[i][j];
            }
        }
    }
}

// ============================================================
// 5. GELU
// x:   [N]
// out: [N]
// ============================================================

__global__ void gelu_forward_kernel(
    const float* x,
    float* out,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        out[idx] = gelu(x[idx]);
    }
}


// ============================================================
// 6. Bias + GELU
// x:    [M, N]
// bias: [N]
// out:  [M, N]
// ============================================================

__global__ void bias_gelu_kernel(
    const float* x,
    const float* bias,
    float* out,
    int M, int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * N;

    if (idx >= total) return;

    int col = idx % N;
    float v = x[idx] + bias[col];

    out[idx] = gelu(v);
}


// ============================================================
// 7. GEMM + bias + GELU
// Useful for MLP first linear layer:
// out = GELU(A @ W + bias)
// ============================================================

__global__ void matmul_bias_gelu_kernel(
    const float* A,
    const float* W,
    const float* bias,
    float* out,
    int M, int K, int N
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    float acc = 0.0f;

    for (int k = 0; k < K; ++k) {
        acc += A[row * K + k] * W[k * N + col];
    }

    float v = acc + bias[col];
    out[row * N + col] = gelu(v);
}


// ============================================================
// 8. Residual add
// out = x + residual
// ============================================================

__global__ void residual_add_kernel(
    const float* x,
    const float* residual,
    float* out,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        out[idx] = x[idx] + residual[idx];
    }
}


// ============================================================
// 9. GEMM + bias + residual
// Useful for attention output projection and MLP second projection.
// out = A @ W + bias + residual
// ============================================================

__global__ void matmul_bias_residual_kernel(
    const float* A,
    const float* W,
    const float* bias,
    const float* residual,
    float* out,
    int M, int K, int N
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    float acc = 0.0f;

    for (int k = 0; k < K; ++k) {
        acc += A[row * K + k] * W[k * N + col];
    }

    int idx = row * N + col;
    out[idx] = acc + bias[col] + residual[idx];
}


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