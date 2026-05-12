#include <cmath>
#include <iostream>
#include <vector>

void cpu_gemm(const float* A, const float* W, float* C, int M, int K, int N) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * W[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

float cpu_gelu(float x) {
    const float sqrt_2_over_pi = 0.7978845608028654f;
    return 0.5f * x * (1.0f + std::tanh(sqrt_2_over_pi * (x + 0.044715f * x * x * x)));
}

void cpu_bias_gelu(const float* x, const float* bias, float* out, int M, int N) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float v = x[i * N + j] + bias[j];
            out[i * N + j] = cpu_gelu(v);
        }
    }
}

void cpu_layernorm(const float* x, const float* gamma, const float* beta, float* out, int B, int T, int C, float eps) {
    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            int base = (b * T + t) * C;
            float sum = 0.0f;
            for (int c = 0; c < C; ++c) {
                sum += x[base + c];
            }
            float mean = sum / C;
            
            float sqsum = 0.0f;
            for (int c = 0; c < C; ++c) {
                float diff = x[base + c] - mean;
                sqsum += diff * diff;
            }
            float var = sqsum / C;
            float inv_std = 1.0f / std::sqrt(var + eps);
            
            for (int c = 0; c < C; ++c) {
                out[base + c] = (x[base + c] - mean) * inv_std * gamma[c] + beta[c];
            }
        }
    }
}

void cpu_attention(const float* Q, const float* K, const float* V, float* out, int B, int NH, int T, int HS) {
    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < NH; ++h) {
            for (int i = 0; i < T; ++i) { // Query position
                float max_score = -1e20f;
                std::vector<float> scores(T, -1e20f);
                
                // Q * K^T
                for (int j = 0; j <= i; ++j) { // Key position (causal mask)
                    float acc = 0.0f;
                    for (int d = 0; d < HS; ++d) {
                        int q_idx = ((b * NH + h) * T + i) * HS + d;
                        int k_idx = ((b * NH + h) * T + j) * HS + d;
                        acc += Q[q_idx] * K[k_idx];
                    }
                    float score = acc / std::sqrt(static_cast<float>(HS));
                    scores[j] = score;
                    if (score > max_score) max_score = score;
                }
                
                // Softmax
                float denom = 0.0f;
                for (int j = 0; j <= i; ++j) {
                    scores[j] = std::exp(scores[j] - max_score);
                    denom += scores[j];
                }
                
                // probs * V
                for (int d = 0; d < HS; ++d) {
                    float out_val = 0.0f;
                    for (int j = 0; j <= i; ++j) {
                        int v_idx = ((b * NH + h) * T + j) * HS + d;
                        out_val += (scores[j] / denom) * V[v_idx];
                    }
                    int out_idx = ((b * NH + h) * T + i) * HS + d;
                    out[out_idx] = out_val;
                }
            }
        }
    }
}
