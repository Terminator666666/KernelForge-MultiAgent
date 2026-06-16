// ============================================================================
// 完整的性能测试程序 - 真正的 Naive Baseline vs 最终优化版本
// 目的: 重新计算基于真实 naive 实现的加速比
// ============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <iomanip>
#include <cmath>

// ============================================================================
// 1. Softmax
// ============================================================================

// True Naive 版本
__global__ void softmax_true_naive_kernel(const float* input, float* output, int batch_size, int seq_len) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= batch_size) return;

    const float* row_input = input + row * seq_len;
    float* row_output = output + row * seq_len;

    float max_val = -INFINITY;
    for (int i = 0; i < seq_len; i++) {
        max_val = fmaxf(max_val, row_input[i]);
    }

    float sum = 0.0f;
    for (int i = 0; i < seq_len; i++) {
        sum += expf(row_input[i] - max_val);
    }

    for (int i = 0; i < seq_len; i++) {
        row_output[i] = expf(row_input[i] - max_val) / sum;
    }
}

// 最终优化版本（从 softmax_final.cu 复制）
__global__ void softmax_optimized_kernel(const float* input, float* output, int batch_size, int seq_len) {
    int row = blockIdx.x;
    if (row >= batch_size) return;

    const float* row_input = input + row * seq_len;
    float* row_output = output + row * seq_len;

    __shared__ float smem[256];
    int tid = threadIdx.x;

    // Warp-level reduction for max
    float thread_max = -INFINITY;
    for (int i = tid; i < seq_len; i += 256) {
        thread_max = fmaxf(thread_max, row_input[i]);
    }

    smem[tid] = thread_max;
    __syncthreads();

    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] = fmaxf(smem[tid], smem[tid + s]);
        }
        __syncthreads();
    }
    float row_max = smem[0];
    __syncthreads();

    // Warp-level reduction for sum
    float thread_sum = 0.0f;
    for (int i = tid; i < seq_len; i += 256) {
        thread_sum += expf(row_input[i] - row_max);
    }

    smem[tid] = thread_sum;
    __syncthreads();

    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }
    float row_sum = smem[0];
    __syncthreads();

    // Normalize
    for (int i = tid; i < seq_len; i += 256) {
        row_output[i] = expf(row_input[i] - row_max) / row_sum;
    }
}

// ============================================================================
// 2. MatMul
// ============================================================================

// True Naive 版本
__global__ void matmul_true_naive_kernel(const half* A, const half* B, half* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            float a = __half2float(A[row * K + k]);
            float b = __half2float(B[k * N + col]);
            sum += a * b;
        }
        C[row * N + col] = __float2half(sum);
    }
}

// 最终优化版本（简化的 Tensor Core 版本）
__global__ void matmul_optimized_kernel(const half* A, const half* B, half* C, int M, int N, int K) {
    // 使用 shared memory tiling
    __shared__ half As[64][64];
    __shared__ half Bs[64][64];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * 64 + ty;
    int col = bx * 64 + tx;

    float sum = 0.0f;

    for (int t = 0; t < (K + 63) / 64; t++) {
        if (row < M && (t * 64 + tx) < K)
            As[ty][tx] = A[row * K + t * 64 + tx];
        else
            As[ty][tx] = __float2half(0.0f);

        if ((t * 64 + ty) < K && col < N)
            Bs[ty][tx] = B[(t * 64 + ty) * N + col];
        else
            Bs[ty][tx] = __float2half(0.0f);

        __syncthreads();

        for (int k = 0; k < 64; k++) {
            sum += __half2float(As[ty][k]) * __half2float(Bs[k][tx]);
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = __float2half(sum);
    }
}

// ============================================================================
// 3. LayerNorm
// ============================================================================

// True Naive 版本
__global__ void layernorm_true_naive_kernel(const float* input, float* output,
                                             const float* gamma, const float* beta,
                                             int batch_size, int hidden_size, float eps) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= batch_size) return;

    const float* row_input = input + row * hidden_size;
    float* row_output = output + row * hidden_size;

    float sum = 0.0f;
    for (int i = 0; i < hidden_size; i++) {
        sum += row_input[i];
    }
    float mean = sum / hidden_size;

    float var_sum = 0.0f;
    for (int i = 0; i < hidden_size; i++) {
        float diff = row_input[i] - mean;
        var_sum += diff * diff;
    }
    float inv_std = rsqrtf(var_sum / hidden_size + eps);

    for (int i = 0; i < hidden_size; i++) {
        float normalized = (row_input[i] - mean) * inv_std;
        row_output[i] = normalized * gamma[i] + beta[i];
    }
}

// 最终优化版本
__global__ void layernorm_optimized_kernel(const float* input, float* output,
                                            const float* gamma, const float* beta,
                                            int batch_size, int hidden_size, float eps) {
    int row = blockIdx.x;
    if (row >= batch_size) return;

    const float* row_input = input + row * hidden_size;
    float* row_output = output + row * hidden_size;

    __shared__ float smem[256];
    int tid = threadIdx.x;

    // Parallel mean calculation
    float sum = 0.0f;
    for (int i = tid; i < hidden_size; i += 256) {
        sum += row_input[i];
    }

    smem[tid] = sum;
    __syncthreads();

    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    float mean = smem[0] / hidden_size;
    __syncthreads();

    // Parallel variance calculation
    float var_sum = 0.0f;
    for (int i = tid; i < hidden_size; i += 256) {
        float diff = row_input[i] - mean;
        var_sum += diff * diff;
    }

    smem[tid] = var_sum;
    __syncthreads();

    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    float inv_std = rsqrtf(smem[0] / hidden_size + eps);
    __syncthreads();

    // Parallel normalization
    for (int i = tid; i < hidden_size; i += 256) {
        float normalized = (row_input[i] - mean) * inv_std;
        row_output[i] = normalized * gamma[i] + beta[i];
    }
}

// ============================================================================
// 4. RMSNorm
// ============================================================================

// True Naive 版本
__global__ void rmsnorm_true_naive_kernel(const half* input, half* output, const half* weight,
                                           int batch_size, int hidden_size, float eps) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= batch_size) return;

    const half* row_input = input + row * hidden_size;
    half* row_output = output + row * hidden_size;

    float sum_sq = 0.0f;
    for (int i = 0; i < hidden_size; i++) {
        float val = __half2float(row_input[i]);
        sum_sq += val * val;
    }
    float rms_inv = rsqrtf(sum_sq / hidden_size + eps);

    for (int i = 0; i < hidden_size; i++) {
        float val = __half2float(row_input[i]);
        float w = __half2float(weight[i]);
        row_output[i] = __float2half(val * rms_inv * w);
    }
}

// 最终优化版本
__global__ void rmsnorm_optimized_kernel(const half* input, half* output, const half* weight,
                                          int batch_size, int hidden_size, float eps) {
    int row = blockIdx.x;
    if (row >= batch_size) return;

    const half* row_input = input + row * hidden_size;
    half* row_output = output + row * hidden_size;

    __shared__ float smem[256];
    int tid = threadIdx.x;

    // Parallel sum of squares
    float sum_sq = 0.0f;
    for (int i = tid; i < hidden_size; i += 256) {
        float val = __half2float(row_input[i]);
        sum_sq += val * val;
    }

    smem[tid] = sum_sq;
    __syncthreads();

    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    float rms_inv = rsqrtf(smem[0] / hidden_size + eps);
    __syncthreads();

    // Parallel normalization
    for (int i = tid; i < hidden_size; i += 256) {
        float val = __half2float(row_input[i]);
        float w = __half2float(weight[i]);
        row_output[i] = __float2half(val * rms_inv * w);
    }
}

// ============================================================================
// Benchmark 工具函数
// ============================================================================

float benchmark_kernel(void (*kernel_func)(), int warmup, int iterations) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup
    for (int i = 0; i < warmup; i++) {
        kernel_func();
    }
    cudaDeviceSynchronize();

    // Benchmark
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        kernel_func();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return milliseconds / iterations;
}

// ============================================================================
// Main 测试程序
// ============================================================================

int main() {
    std::cout << "================================================================================\n";
    std::cout << "  重新计算加速比 - 基于真正的 Naive Baseline\n";
    std::cout << "================================================================================\n\n";

    const int WARMUP = 10;
    const int ITERATIONS = 100;

    // 1. Softmax 测试
    {
        const int batch_size = 1;
        const int seq_len = 1000000;  // 1M 元素

        float *d_input, *d_output;
        cudaMalloc(&d_input, batch_size * seq_len * sizeof(float));
        cudaMalloc(&d_output, batch_size * seq_len * sizeof(float));

        // True Naive
        auto naive_func = [&]() {
            int threads = 256;
            int blocks = (batch_size + threads - 1) / threads;
            softmax_true_naive_kernel<<<blocks, threads>>>(d_input, d_output, batch_size, seq_len);
        };

        // Optimized
        auto opt_func = [&]() {
            softmax_optimized_kernel<<<batch_size, 256>>>(d_input, d_output, batch_size, seq_len);
        };

        float naive_time = benchmark_kernel(naive_func, WARMUP, ITERATIONS);
        float opt_time = benchmark_kernel(opt_func, WARMUP, ITERATIONS);
        float speedup = naive_time / opt_time;

        std::cout << "算子1: Softmax (1M 元素)\n";
        std::cout << "----------------------------------------\n";
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "True Naive Baseline: " << naive_time << " ms\n";
        std::cout << "Final Optimized: " << opt_time << " ms\n";
        std::cout << "真实加速比: " << speedup << "×\n\n";

        cudaFree(d_input);
        cudaFree(d_output);
    }

    // 2. MatMul 测试
    {
        const int M = 2048, N = 2048, K = 2048;

        half *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, M * K * sizeof(half));
        cudaMalloc(&d_B, K * N * sizeof(half));
        cudaMalloc(&d_C, M * N * sizeof(half));

        // True Naive
        auto naive_func = [&]() {
            dim3 block(16, 16);
            dim3 grid((N + 15) / 16, (M + 15) / 16);
            matmul_true_naive_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        };

        // Optimized
        auto opt_func = [&]() {
            dim3 block(64, 64);
            dim3 grid((N + 63) / 64, (M + 63) / 64);
            matmul_optimized_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        };

        float naive_time = benchmark_kernel(naive_func, WARMUP, ITERATIONS);
        float opt_time = benchmark_kernel(opt_func, WARMUP, ITERATIONS);
        float speedup = naive_time / opt_time;

        std::cout << "算子2: MatMul (2048×2048×2048 FP16)\n";
        std::cout << "----------------------------------------\n";
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "True Naive Baseline: " << naive_time << " ms\n";
        std::cout << "Final Optimized: " << opt_time << " ms\n";
        std::cout << "真实加速比: " << speedup << "×\n\n";

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }

    // 3. LayerNorm 测试
    {
        const int batch_size = 1024;
        const int hidden_size = 4096;
        const float eps = 1e-5f;

        float *d_input, *d_output, *d_gamma, *d_beta;
        cudaMalloc(&d_input, batch_size * hidden_size * sizeof(float));
        cudaMalloc(&d_output, batch_size * hidden_size * sizeof(float));
        cudaMalloc(&d_gamma, hidden_size * sizeof(float));
        cudaMalloc(&d_beta, hidden_size * sizeof(float));

        // True Naive
        auto naive_func = [&]() {
            int threads = 256;
            int blocks = (batch_size + threads - 1) / threads;
            layernorm_true_naive_kernel<<<blocks, threads>>>(d_input, d_output, d_gamma, d_beta, batch_size, hidden_size, eps);
        };

        // Optimized
        auto opt_func = [&]() {
            layernorm_optimized_kernel<<<batch_size, 256>>>(d_input, d_output, d_gamma, d_beta, batch_size, hidden_size, eps);
        };

        float naive_time = benchmark_kernel(naive_func, WARMUP, ITERATIONS);
        float opt_time = benchmark_kernel(opt_func, WARMUP, ITERATIONS);
        float speedup = naive_time / opt_time;

        std::cout << "算子3: LayerNorm (batch=1024, hidden=4096)\n";
        std::cout << "----------------------------------------\n";
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "True Naive Baseline: " << naive_time << " ms\n";
        std::cout << "Final Optimized: " << opt_time << " ms\n";
        std::cout << "真实加速比: " << speedup << "×\n\n";

        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_gamma);
        cudaFree(d_beta);
    }

    // 4. RMSNorm 测试
    {
        const int batch_size = 1024;
        const int hidden_size = 4096;
        const float eps = 1e-6f;

        half *d_input, *d_output, *d_weight;
        cudaMalloc(&d_input, batch_size * hidden_size * sizeof(half));
        cudaMalloc(&d_output, batch_size * hidden_size * sizeof(half));
        cudaMalloc(&d_weight, hidden_size * sizeof(half));

        // True Naive
        auto naive_func = [&]() {
            int threads = 256;
            int blocks = (batch_size + threads - 1) / threads;
            rmsnorm_true_naive_kernel<<<blocks, threads>>>(d_input, d_output, d_weight, batch_size, hidden_size, eps);
        };

        // Optimized
        auto opt_func = [&]() {
            rmsnorm_optimized_kernel<<<batch_size, 256>>>(d_input, d_output, d_weight, batch_size, hidden_size, eps);
        };

        float naive_time = benchmark_kernel(naive_func, WARMUP, ITERATIONS);
        float opt_time = benchmark_kernel(opt_func, WARMUP, ITERATIONS);
        float speedup = naive_time / opt_time;

        std::cout << "算子4: RMSNorm (batch=1024, hidden=4096)\n";
        std::cout << "----------------------------------------\n";
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "True Naive Baseline: " << naive_time << " ms\n";
        std::cout << "Final Optimized: " << opt_time << " ms\n";
        std::cout << "真实加速比: " << speedup << "×\n\n";

        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_weight);
    }

    std::cout << "================================================================================\n";
    std::cout << "测试完成！\n";
    std::cout << "================================================================================\n";

    return 0;
}
