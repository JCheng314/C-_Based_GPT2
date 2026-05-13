#include "../include/transformer_kernels.h"
#include <math.h>
#include <float.h>

// ============================================================
// 0. helper: gelu backward
// ============================================================

__device__ __forceinline__ float gelu_backward(float x) {
    const float a = 0.7978845608028654f;  // sqrt(2 / pi)
    const float b = 0.044715f;

    float x2 = x * x;
    float x3 = x2 * x;

    float u = a * (x + b * x3);
    float tanh_u = tanhf(u);
    float sech2 = 1.0f - tanh_u * tanh_u;

    float du_dx = a * (1.0f + 3.0f * b * x2);

    return 0.5f * (1.0f + tanh_u) + 0.5f * x * sech2 * du_dx;
}

// ============================================================
// 1. Zero kernel
// ============================================================

__global__ void zero_kernel(float* x, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        x[idx] = 0.0f;
    }
}

// ============================================================
// 2. Cross entropy forward
//
// logits:  [M, V]
// targets: [M]
// losses:  [M]
// ============================================================

__global__ void cross_entropy_forward_kernel(
    const float* logits,
    const int* targets,
    float* losses,
    int M,
    int V
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;

    const float* row_logits = logits + row * V;
    int target = targets[row];

    float max_logit = -FLT_MAX;
    for (int v = 0; v < V; ++v) {
        max_logit = fmaxf(max_logit, row_logits[v]);
    }

    float sum_exp = 0.0f;
    for (int v = 0; v < V; ++v) {
        sum_exp += expf(row_logits[v] - max_logit);
    }

    float log_sum_exp = logf(sum_exp) + max_logit;
    losses[row] = -row_logits[target] + log_sum_exp;
}

// ============================================================
// 3. Cross entropy backward
//
// dlogits = (softmax(logits) - one_hot(target)) / M
// ============================================================

__global__ void cross_entropy_backward_kernel(
    const float* logits,
    const int* targets,
    float* dlogits,
    int M,
    int V
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= M) return;

    extern __shared__ float shared[];
    float* s_max = shared;
    float* s_sum = shared + blockDim.x;

    const float* row_logits = logits + row * V;
    float* row_dlogits = dlogits + row * V;

    float local_max = -FLT_MAX;

    for (int v = tid; v < V; v += blockDim.x) {
        local_max = fmaxf(local_max, row_logits[v]);
    }

    s_max[tid] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_max[tid] = fmaxf(s_max[tid], s_max[tid + stride]);
        }
        __syncthreads();
    }

    float max_logit = s_max[0];

    float local_sum = 0.0f;

    for (int v = tid; v < V; v += blockDim.x) {
        local_sum += expf(row_logits[v] - max_logit);
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
    int target = targets[row];

    for (int v = tid; v < V; v += blockDim.x) {
        float prob = expf(row_logits[v] - max_logit) / denom;
        float grad = prob;

        if (v == target) {
            grad -= 1.0f;
        }

        row_dlogits[v] = grad / static_cast<float>(M);
    }
}

// ============================================================
// 4. Linear backward
//
// Forward:
// C = A @ W
//
// A:  [M, K]
// W:  [K, N]
// C:  [M, N]
//
// dA = dC @ W^T
// dW = A^T @ dC
// ============================================================

__global__ void linear_backward_dA_kernel(
    const float* dC,
    const float* W,
    float* dA,
    int M,
    int K,
    int N
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y * blockDim.y + threadIdx.y;

    if (m >= M || k >= K) return;

    float acc = 0.0f;

    for (int n = 0; n < N; ++n) {
        acc += dC[m * N + n] * W[k * N + n];
    }

    dA[m * K + k] += acc;
}

__global__ void linear_backward_dW_kernel(
    const float* A,
    const float* dC,
    float* dW,
    int M,
    int K,
    int N
) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;

    if (k >= K || n >= N) return;

    float acc = 0.0f;

    for (int m = 0; m < M; ++m) {
        acc += A[m * K + k] * dC[m * N + n];
    }

    dW[k * N + n] += acc;
}

__global__ void linear_backward_bias_kernel(
    const float* dC,
    float* dbias,
    int M,
    int N
) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n >= N) return;

    float acc = 0.0f;

    for (int m = 0; m < M; ++m) {
        acc += dC[m * N + n];
    }

    dbias[n] += acc;
}

// ============================================================
// 5. Embedding backward
//
// out[b,t,c] = token_emb[token,c] + pos_emb[t,c]
// ============================================================

__global__ void embedding_backward_kernel(
    const int* tokens,
    const float* dout,
    float* dtoken_emb,
    float* dpos_emb,
    int B,
    int T,
    int C
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * T * C;

    if (idx >= total) return;

    int c = idx % C;
    int t = (idx / C) % T;
    int b = idx / (T * C);

    int token_id = tokens[b * T + t];

    atomicAdd(&dtoken_emb[token_id * C + c], dout[idx]);
    atomicAdd(&dpos_emb[t * C + c], dout[idx]);
}

// ============================================================
// 6. AdamW update
// ============================================================

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
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N) return;

    float g = grad[idx];

    // Decoupled weight decay
    if (weight_decay != 0.0f) {
        param[idx] -= lr * weight_decay * param[idx];
    }

    float m_new = beta1 * m[idx] + (1.0f - beta1) * g;
    float v_new = beta2 * v[idx] + (1.0f - beta2) * g * g;

    m[idx] = m_new;
    v[idx] = v_new;

    float m_hat = m_new / (1.0f - powf(beta1, static_cast<float>(step)));
    float v_hat = v_new / (1.0f - powf(beta2, static_cast<float>(step)));

    param[idx] -= lr * m_hat / (sqrtf(v_hat) + eps);
}

// ============================================================
// Bias + GELU backward
//
// Forward:
// out = GELU(x + bias)
//
// x:     [M, N]
// bias:  [N]
// dout:  [M, N]
// dx:    [M, N]
// dbias: [N]
// ============================================================

__global__ void bias_gelu_backward_kernel(
    const float* x,
    const float* bias,
    const float* dout,
    float* dx,
    float* dbias,
    int M,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * N;

    if (idx >= total) return;

    int n = idx % N;

    float preact = x[idx] + bias[n];
    float grad = dout[idx] * gelu_backward(preact);

    dx[idx] += grad;
    atomicAdd(&dbias[n], grad);
}

// ============================================================
// Residual add backward
//
// Forward:
// out = x + residual
//
// Backward:
// dx += dout
// dresidual += dout
// ============================================================

__global__ void residual_add_backward_kernel(
    const float* dout,
    float* dx,
    float* dresidual,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        dx[idx] += dout[idx];
        dresidual[idx] += dout[idx];
    }
}

// ============================================================
// LayerNorm backward
//
// x:      [M, C]
// gamma:  [C]
// dout:   [M, C]
// dx:     [M, C]
// dgamma: [C]
// dbeta:  [C]
// ============================================================

__global__ void layernorm_backward_kernel(
    const float* x,
    const float* gamma,
    const float* dout,
    float* dx,
    float* dgamma,
    float* dbeta,
    int M,
    int C,
    float eps
) {
    extern __shared__ float shared[];

    float* s_sum = shared;
    float* s_sqsum = shared + blockDim.x;
    float* s_dy_gamma_sum = shared + 2 * blockDim.x;
    float* s_dy_gamma_xhat_sum = shared + 3 * blockDim.x;

    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= M) return;

    const float* x_row = x + row * C;
    const float* dout_row = dout + row * C;
    float* dx_row = dx + row * C;

    float local_sum = 0.0f;
    float local_sqsum = 0.0f;

    for (int c = tid; c < C; c += blockDim.x) {
        float v = x_row[c];
        local_sum += v;
        local_sqsum += v * v;
    }

    s_sum[tid] = local_sum;
    s_sqsum[tid] = local_sqsum;

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

    float local_dy_gamma_sum = 0.0f;
    float local_dy_gamma_xhat_sum = 0.0f;

    for (int c = tid; c < C; c += blockDim.x) {
        float xhat = (x_row[c] - mean) * inv_std;
        float dy_gamma = dout_row[c] * gamma[c];

        local_dy_gamma_sum += dy_gamma;
        local_dy_gamma_xhat_sum += dy_gamma * xhat;

        atomicAdd(&dgamma[c], dout_row[c] * xhat);
        atomicAdd(&dbeta[c], dout_row[c]);
    }

    s_dy_gamma_sum[tid] = local_dy_gamma_sum;
    s_dy_gamma_xhat_sum[tid] = local_dy_gamma_xhat_sum;

    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_dy_gamma_sum[tid] += s_dy_gamma_sum[tid + stride];
            s_dy_gamma_xhat_sum[tid] += s_dy_gamma_xhat_sum[tid + stride];
        }
        __syncthreads();
    }

    float dy_gamma_sum = s_dy_gamma_sum[0];
    float dy_gamma_xhat_sum = s_dy_gamma_xhat_sum[0];

    for (int c = tid; c < C; c += blockDim.x) {
        float xhat = (x_row[c] - mean) * inv_std;
        float dy_gamma = dout_row[c] * gamma[c];

        float grad = inv_std / C *
            ((float)C * dy_gamma - dy_gamma_sum - xhat * dy_gamma_xhat_sum);

        dx_row[c] += grad;
    }
}