#include "../include/transformer_kernels.h"
#include <math.h>

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
// ============================================================

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
