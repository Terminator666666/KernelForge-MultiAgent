#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace softmax_ns {
#include "../../../../kernels/operators/softmax/softmax_final.cu"
#include "../../../../kernels/operators/softmax/softmax_true_naive.cu"
}  // namespace softmax_ns

namespace matmul_ns {
#include "../../../../kernels/operators/matmul/matmul_final.cu"
#include "../../../../kernels/operators/matmul/matmul_true_naive.cu"
}  // namespace matmul_ns

namespace rmsnorm_ns {
#include "../../../../kernels/operators/rmsnorm/rmsnorm_final.cu"
#include "../../../../kernels/operators/rmsnorm/rmsnorm_true_naive.cu"
}  // namespace rmsnorm_ns

namespace layernorm_ns {
#include "../../../../kernels/operators/layernorm/layernorm_final.cu"
#include "../../../../kernels/operators/layernorm/layernorm_true_naive.cu"
}  // namespace layernorm_ns

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err__ = (call);                                        \
        if (err__ != cudaSuccess) {                                        \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__,    \
                         __LINE__, cudaGetErrorString(err__));             \
            std::exit(1);                                                  \
        }                                                                  \
    } while (0)

struct DiffStats {
    float max_abs = 0.0f;
    float mean_abs = 0.0f;
    float max_rel = 0.0f;
};

static std::mt19937 make_rng() {
    return std::mt19937(1234);
}

static float random_float(std::mt19937& rng, float lo, float hi) {
    std::uniform_real_distribution<float> dist(lo, hi);
    return dist(rng);
}

static void fill_random(std::vector<float>& values, std::mt19937& rng, float lo, float hi) {
    for (float& v : values) {
        v = random_float(rng, lo, hi);
    }
}

static void fill_random(std::vector<half>& values, std::mt19937& rng, float lo, float hi) {
    for (half& v : values) {
        v = __float2half(random_float(rng, lo, hi));
    }
}

static std::vector<float> to_float(const std::vector<half>& values) {
    std::vector<float> out(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        out[i] = __half2float(values[i]);
    }
    return out;
}

static DiffStats compare_vectors(const std::vector<float>& a, const std::vector<float>& b) {
    DiffStats stats;
    if (a.size() != b.size()) {
        stats.max_abs = std::numeric_limits<float>::infinity();
        stats.max_rel = std::numeric_limits<float>::infinity();
        return stats;
    }

    float sum_abs = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        float abs_err = std::fabs(a[i] - b[i]);
        float denom = std::max(1.0f, std::fabs(b[i]));
        float rel_err = abs_err / denom;
        if (abs_err > stats.max_abs) stats.max_abs = abs_err;
        if (rel_err > stats.max_rel) stats.max_rel = rel_err;
        sum_abs += abs_err;
    }

    stats.mean_abs = sum_abs / static_cast<float>(a.size());
    return stats;
}

static void print_stats(const char* name, const DiffStats& stats) {
    std::printf("%s: max_abs=%.6f mean_abs=%.6f max_rel=%.6f\n",
                name, stats.max_abs, stats.mean_abs, stats.max_rel);
}

static bool softmax_cpu_ref(const std::vector<float>& input, std::vector<float>& output,
                            int batch_size, int seq_len) {
    output.resize(input.size());
    for (int row = 0; row < batch_size; ++row) {
        const float* row_input = input.data() + row * seq_len;
        float* row_output = output.data() + row * seq_len;

        float row_max = row_input[0];
        for (int i = 1; i < seq_len; ++i) {
            if (row_input[i] > row_max) row_max = row_input[i];
        }

        float sum = 0.0f;
        for (int i = 0; i < seq_len; ++i) {
            sum += std::exp(row_input[i] - row_max);
        }

        for (int i = 0; i < seq_len; ++i) {
            row_output[i] = std::exp(row_input[i] - row_max) / sum;
        }
    }
    return true;
}

static bool matmul_cpu_ref(const std::vector<half>& A, const std::vector<half>& B,
                           std::vector<float>& output, int M, int N, int K) {
    output.assign(M * N, 0.0f);
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += __half2float(A[row * K + k]) * __half2float(B[k * N + col]);
            }
            output[row * N + col] = sum;
        }
    }
    return true;
}

static bool rmsnorm_cpu_ref(const std::vector<half>& input, const std::vector<half>& weight,
                            std::vector<float>& output, int batch_size, int hidden_size,
                            float eps) {
    output.assign(input.size(), 0.0f);
    for (int row = 0; row < batch_size; ++row) {
        const half* row_input = input.data() + row * hidden_size;
        float sum_sq = 0.0f;
        for (int i = 0; i < hidden_size; ++i) {
            float v = __half2float(row_input[i]);
            sum_sq += v * v;
        }
        float inv_rms = 1.0f / std::sqrt(sum_sq / hidden_size + eps);
        for (int i = 0; i < hidden_size; ++i) {
            float v = __half2float(row_input[i]);
            float w = __half2float(weight[i]);
            output[row * hidden_size + i] = v * inv_rms * w;
        }
    }
    return true;
}

static bool layernorm_cpu_ref(const std::vector<float>& input, const std::vector<float>& gamma,
                              const std::vector<float>& beta, std::vector<float>& output,
                              int batch_size, int hidden_size, float eps) {
    output.assign(input.size(), 0.0f);
    for (int row = 0; row < batch_size; ++row) {
        const float* row_input = input.data() + row * hidden_size;
        float* row_output = output.data() + row * hidden_size;

        float sum = 0.0f;
        for (int i = 0; i < hidden_size; ++i) {
            sum += row_input[i];
        }
        float mean = sum / hidden_size;

        float var_sum = 0.0f;
        for (int i = 0; i < hidden_size; ++i) {
            float diff = row_input[i] - mean;
            var_sum += diff * diff;
        }
        float inv_std = 1.0f / std::sqrt(var_sum / hidden_size + eps);

        for (int i = 0; i < hidden_size; ++i) {
            row_output[i] = (row_input[i] - mean) * inv_std * gamma[i] + beta[i];
        }
    }
    return true;
}

static bool run_softmax_case(int batch_size, int seq_len, std::mt19937& rng) {
    std::printf("\n[Softmax] batch=%d seq_len=%d\n", batch_size, seq_len);

    std::vector<float> input(batch_size * seq_len);
    fill_random(input, rng, -4.0f, 4.0f);

    std::vector<float> cpu_ref;
    softmax_cpu_ref(input, cpu_ref, batch_size, seq_len);

    float *d_input = nullptr, *d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, input.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, input.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice));

    std::vector<float> optimized(input.size());
    softmax_ns::launch_softmax_optimized(d_input, d_output, batch_size, seq_len, 0);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(optimized.data(), d_output, optimized.size() * sizeof(float), cudaMemcpyDeviceToHost));

    std::vector<float> naive(input.size());
    softmax_ns::launch_softmax_true_naive(d_input, d_output, batch_size, seq_len, 0);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(naive.data(), d_output, naive.size() * sizeof(float), cudaMemcpyDeviceToHost));

    auto opt_stats = compare_vectors(optimized, cpu_ref);
    auto naive_stats = compare_vectors(naive, cpu_ref);
    auto cross_stats = compare_vectors(optimized, naive);

    print_stats("  optimized vs cpu", opt_stats);
    print_stats("  naive vs cpu", naive_stats);
    print_stats("  optimized vs naive", cross_stats);

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return opt_stats.max_abs < 1e-4f && naive_stats.max_abs < 1e-4f;
}

static bool run_matmul_case(int M, int N, int K, std::mt19937& rng) {
    std::printf("\n[MatMul] M=%d N=%d K=%d\n", M, N, K);

    std::vector<half> A(M * K);
    std::vector<half> B(K * N);
    fill_random(A, rng, -1.0f, 1.0f);
    fill_random(B, rng, -1.0f, 1.0f);

    std::vector<float> cpu_ref;
    matmul_cpu_ref(A, B, cpu_ref, M, N, K);

    half *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, A.size() * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_B, B.size() * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(half)));
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), A.size() * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B.data(), B.size() * sizeof(half), cudaMemcpyHostToDevice));

    std::vector<half> optimized_half(M * N);
    matmul_ns::launch_matmul_optimized(d_A, d_B, d_C, M, N, K, 0);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(optimized_half.data(), d_C, optimized_half.size() * sizeof(half), cudaMemcpyDeviceToHost));

    std::vector<half> naive_half(M * N);
    matmul_ns::launch_matmul_true_naive(d_A, d_B, d_C, M, N, K, 0);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(naive_half.data(), d_C, naive_half.size() * sizeof(half), cudaMemcpyDeviceToHost));

    auto optimized = to_float(optimized_half);
    auto naive = to_float(naive_half);
    auto ref_optimized = compare_vectors(optimized, cpu_ref);
    auto ref_naive = compare_vectors(naive, cpu_ref);
    auto cross_stats = compare_vectors(optimized, naive);

    print_stats("  optimized vs cpu", ref_optimized);
    print_stats("  naive vs cpu", ref_naive);
    print_stats("  optimized vs naive", cross_stats);

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return ref_optimized.max_abs < 2e-2f && ref_naive.max_abs < 2e-2f;
}

static bool run_rmsnorm_case(int batch_size, int hidden_size, float eps, std::mt19937& rng) {
    std::printf("\n[RMSNorm] batch=%d hidden=%d\n", batch_size, hidden_size);

    std::vector<half> input(batch_size * hidden_size);
    std::vector<half> weight(hidden_size);
    fill_random(input, rng, -1.0f, 1.0f);
    fill_random(weight, rng, 0.5f, 1.5f);

    std::vector<float> cpu_ref;
    rmsnorm_cpu_ref(input, weight, cpu_ref, batch_size, hidden_size, eps);

    half *d_input = nullptr, *d_output = nullptr, *d_weight = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, input.size() * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_output, input.size() * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_weight, weight.size() * sizeof(half)));
    CUDA_CHECK(cudaMemcpy(d_input, input.data(), input.size() * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weight, weight.data(), weight.size() * sizeof(half), cudaMemcpyHostToDevice));

    std::vector<half> optimized_half(input.size());
    rmsnorm_ns::launch_rmsnorm_optimized_ex(d_input, d_output, d_weight, batch_size, hidden_size, eps, 0);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(optimized_half.data(), d_output, optimized_half.size() * sizeof(half), cudaMemcpyDeviceToHost));

    std::vector<half> naive_half(input.size());
    rmsnorm_ns::launch_rmsnorm_true_naive_ex(d_input, d_output, d_weight, batch_size, hidden_size, eps, 0);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(naive_half.data(), d_output, naive_half.size() * sizeof(half), cudaMemcpyDeviceToHost));

    auto optimized = to_float(optimized_half);
    auto naive = to_float(naive_half);
    auto ref_optimized = compare_vectors(optimized, cpu_ref);
    auto ref_naive = compare_vectors(naive, cpu_ref);
    auto cross_stats = compare_vectors(optimized, naive);

    print_stats("  optimized vs cpu", ref_optimized);
    print_stats("  naive vs cpu", ref_naive);
    print_stats("  optimized vs naive", cross_stats);

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_weight));

    return ref_optimized.max_abs < 2e-2f && ref_naive.max_abs < 2e-2f;
}

static bool run_layernorm_case(int batch_size, int hidden_size, float eps, std::mt19937& rng) {
    std::printf("\n[LayerNorm] batch=%d hidden=%d\n", batch_size, hidden_size);

    std::vector<float> input(batch_size * hidden_size);
    std::vector<float> gamma(hidden_size);
    std::vector<float> beta(hidden_size);
    fill_random(input, rng, -1.0f, 1.0f);
    fill_random(gamma, rng, 0.5f, 1.5f);
    fill_random(beta, rng, -0.2f, 0.2f);

    std::vector<float> cpu_ref;
    layernorm_cpu_ref(input, gamma, beta, cpu_ref, batch_size, hidden_size, eps);

    float *d_input = nullptr, *d_output = nullptr, *d_gamma = nullptr, *d_beta = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, input.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, input.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gamma, gamma.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_beta, beta.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gamma, gamma.data(), gamma.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_beta, beta.data(), beta.size() * sizeof(float), cudaMemcpyHostToDevice));

    std::vector<float> optimized(input.size());
    layernorm_ns::launch_layernorm_optimized(d_input, d_output, d_gamma, d_beta, batch_size, hidden_size, eps, 0);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(optimized.data(), d_output, optimized.size() * sizeof(float), cudaMemcpyDeviceToHost));

    std::vector<float> naive(input.size());
    layernorm_ns::launch_layernorm_true_naive(d_input, d_output, d_gamma, d_beta, batch_size, hidden_size, eps, 0);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(naive.data(), d_output, naive.size() * sizeof(float), cudaMemcpyDeviceToHost));

    auto ref_optimized = compare_vectors(optimized, cpu_ref);
    auto ref_naive = compare_vectors(naive, cpu_ref);
    auto cross_stats = compare_vectors(optimized, naive);

    print_stats("  optimized vs cpu", ref_optimized);
    print_stats("  naive vs cpu", ref_naive);
    print_stats("  optimized vs naive", cross_stats);

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_gamma));
    CUDA_CHECK(cudaFree(d_beta));

    return ref_optimized.max_abs < 1e-4f && ref_naive.max_abs < 1e-4f;
}

int main() {
    std::printf("============================================\n");
    std::printf("KernelForge CUDA smoke test\n");
    std::printf("============================================\n");

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        std::fprintf(stderr, "No CUDA device found.\n");
        return 1;
    }

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("Device: %s\n", prop.name);
    std::printf("Capability: %d.%d\n", prop.major, prop.minor);

    std::mt19937 rng = make_rng();

    bool softmax_ok = run_softmax_case(4, 128, rng);
    bool matmul_ok = run_matmul_case(64, 64, 64, rng);
    bool rmsnorm_ok = run_rmsnorm_case(4, 1024, 1.0e-6f, rng);
    bool layernorm_ok = run_layernorm_case(4, 1024, 1.0e-5f, rng);

    std::printf("\n============================================\n");
    std::printf("Summary\n");
    std::printf("============================================\n");
    std::printf("Softmax:   %s\n", softmax_ok ? "PASS" : "FAIL");
    std::printf("MatMul:    %s\n", matmul_ok ? "PASS" : "FAIL");
    std::printf("RMSNorm:   %s\n", rmsnorm_ok ? "PASS" : "FAIL");
    std::printf("LayerNorm: %s\n", layernorm_ok ? "PASS" : "FAIL");
    std::printf("============================================\n");

    return (softmax_ok && matmul_ok && rmsnorm_ok && layernorm_ok) ? 0 : 1;
}
