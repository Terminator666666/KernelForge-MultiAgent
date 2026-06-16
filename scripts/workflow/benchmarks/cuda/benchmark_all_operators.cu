// ============================================================================
// 完整的 4 个算子性能测试程序
// ============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <ctime>

// ============================================================================
// Softmax Kernels
// ============================================================================

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

__global__ void softmax_optimized_kernel(const float* input, float* output, int batch_size, int seq_len) {
    int row = blockIdx.x;
    if (row >= batch_size) return;

    const float* row_input = input + row * seq_len;
    float* row_output = output + row * seq_len;

    __shared__ float smem[256];
    int tid = threadIdx.x;

    float thread_max = -INFINITY;
    for (int i = tid; i < seq_len; i += 256) {
        thread_max = fmaxf(thread_max, row_input[i]);
    }
    smem[tid] = thread_max;
    __syncthreads();

    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) smem[tid] = fmaxf(smem[tid], smem[tid + s]);
        __syncthreads();
    }
    float row_max = smem[0];
    __syncthreads();

    float thread_sum = 0.0f;
    for (int i = tid; i < seq_len; i += 256) {
        thread_sum += expf(row_input[i] - row_max);
    }
    smem[tid] = thread_sum;
    __syncthreads();

    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    float row_sum = smem[0];
    __syncthreads();

    for (int i = tid; i < seq_len; i += 256) {
        row_output[i] = expf(row_input[i] - row_max) / row_sum;
    }
}

// ============================================================================
// MatMul Kernels
// ============================================================================

__global__ void matmul_true_naive_kernel(const half* A, const half* B, half* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += __half2float(A[row * K + k]) * __half2float(B[k * N + col]);
        }
        C[row * N + col] = __float2half(sum);
    }
}

__global__ void matmul_optimized_kernel(const half* A, const half* B, half* C, int M, int N, int K) {
    const int TILE = 32;
    __shared__ half As[TILE][TILE];
    __shared__ half Bs[TILE][TILE];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = by * TILE + ty;
    int col = bx * TILE + tx;

    float sum = 0.0f;
    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        if (row < M && (t * TILE + tx) < K)
            As[ty][tx] = A[row * K + t * TILE + tx];
        else
            As[ty][tx] = __float2half(0.0f);

        if ((t * TILE + ty) < K && col < N)
            Bs[ty][tx] = B[(t * TILE + ty) * N + col];
        else
            Bs[ty][tx] = __float2half(0.0f);

        __syncthreads();
        for (int k = 0; k < TILE; k++) {
            sum += __half2float(As[ty][k]) * __half2float(Bs[k][tx]);
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = __float2half(sum);
    }
}

// ============================================================================
// LayerNorm Kernels
// ============================================================================

__global__ void layernorm_true_naive_kernel(const float* input, float* output,
                                             const float* gamma, const float* beta,
                                             int batch_size, int hidden_size, float eps) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= batch_size) return;

    const float* row_input = input + row * hidden_size;
    float* row_output = output + row * hidden_size;

    float sum = 0.0f;
    for (int i = 0; i < hidden_size; i++) sum += row_input[i];
    float mean = sum / hidden_size;

    float var_sum = 0.0f;
    for (int i = 0; i < hidden_size; i++) {
        float diff = row_input[i] - mean;
        var_sum += diff * diff;
    }
    float inv_std = rsqrtf(var_sum / hidden_size + eps);

    for (int i = 0; i < hidden_size; i++) {
        row_output[i] = (row_input[i] - mean) * inv_std * gamma[i] + beta[i];
    }
}

__global__ void layernorm_optimized_kernel(const float* input, float* output,
                                            const float* gamma, const float* beta,
                                            int batch_size, int hidden_size, float eps) {
    int row = blockIdx.x;
    if (row >= batch_size) return;

    const float* row_input = input + row * hidden_size;
    float* row_output = output + row * hidden_size;
    __shared__ float smem[256];
    int tid = threadIdx.x;

    float sum = 0.0f;
    for (int i = tid; i < hidden_size; i += 256) sum += row_input[i];
    smem[tid] = sum;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    float mean = smem[0] / hidden_size;
    __syncthreads();

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

    for (int i = tid; i < hidden_size; i += 256) {
        row_output[i] = (row_input[i] - mean) * inv_std * gamma[i] + beta[i];
    }
}

// ============================================================================
// RMSNorm Kernels
// ============================================================================

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

__global__ void rmsnorm_optimized_kernel(const half* input, half* output, const half* weight,
                                          int batch_size, int hidden_size, float eps) {
    int row = blockIdx.x;
    if (row >= batch_size) return;

    const half* row_input = input + row * hidden_size;
    half* row_output = output + row * hidden_size;
    __shared__ float smem[256];
    int tid = threadIdx.x;

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

    for (int i = tid; i < hidden_size; i += 256) {
        float val = __half2float(row_input[i]);
        float w = __half2float(weight[i]);
        row_output[i] = __float2half(val * rms_inv * w);
    }
}

// ============================================================================
// Benchmark Functions
// ============================================================================

template<typename Func, typename... Args>
float benchmark(Func kernel, dim3 grid, dim3 block, int warmup, int iterations, Args... args) {
    for (int i = 0; i < warmup; i++) kernel<<<grid, block>>>(args...);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) kernel<<<grid, block>>>(args...);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / iterations;
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "================================================================================\n";
    std::cout << "  真实 Naive Baseline 加速比测试\n";
    std::cout << "================================================================================\n\n";

    const int WARMUP = 10, ITER = 100;

    // 1. Softmax
    {
        const int batch = 1, seq_len = 1000000;
        float *d_in, *d_out;
        cudaMalloc(&d_in, batch * seq_len * sizeof(float));
        cudaMalloc(&d_out, batch * seq_len * sizeof(float));

        float t_naive = benchmark(softmax_true_naive_kernel, dim3((batch+255)/256), dim3(256), WARMUP, ITER, d_in, d_out, batch, seq_len);
        float t_opt = benchmark(softmax_optimized_kernel, dim3(batch), dim3(256), WARMUP, ITER, d_in, d_out, batch, seq_len);

        std::cout << "算子1: Softmax (1M 元素)\n";
        std::cout << "  True Naive: " << std::fixed << std::setprecision(3) << t_naive << " ms\n";
        std::cout << "  Optimized:  " << t_opt << " ms\n";
        std::cout << "  加速比: " << (t_naive/t_opt) << "×\n\n";

        cudaFree(d_in); cudaFree(d_out);
    }

    // 2. MatMul
    {
        const int M = 2048, N = 2048, K = 2048;
        half *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, M*K*sizeof(half));
        cudaMalloc(&d_B, K*N*sizeof(half));
        cudaMalloc(&d_C, M*N*sizeof(half));

        float t_naive = benchmark(matmul_true_naive_kernel, dim3((N+15)/16, (M+15)/16), dim3(16, 16), WARMUP, ITER, d_A, d_B, d_C, M, N, K);
        float t_opt = benchmark(matmul_optimized_kernel, dim3((N+31)/32, (M+31)/32), dim3(32, 32), WARMUP, ITER, d_A, d_B, d_C, M, N, K);

        std::cout << "算子2: MatMul (2048×2048×2048 FP16)\n";
        std::cout << "  True Naive: " << std::fixed << std::setprecision(3) << t_naive << " ms\n";
        std::cout << "  Optimized:  " << t_opt << " ms\n";
        std::cout << "  加速比: " << (t_naive/t_opt) << "×\n\n";

        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    }

    // 3. LayerNorm
    {
        const int batch = 1024, hidden = 4096;
        float *d_in, *d_out, *d_gamma, *d_beta;
        cudaMalloc(&d_in, batch*hidden*sizeof(float));
        cudaMalloc(&d_out, batch*hidden*sizeof(float));
        cudaMalloc(&d_gamma, hidden*sizeof(float));
        cudaMalloc(&d_beta, hidden*sizeof(float));

        float t_naive = benchmark(layernorm_true_naive_kernel, dim3((batch+255)/256), dim3(256), WARMUP, ITER, d_in, d_out, d_gamma, d_beta, batch, hidden, 1e-5f);
        float t_opt = benchmark(layernorm_optimized_kernel, dim3(batch), dim3(256), WARMUP, ITER, d_in, d_out, d_gamma, d_beta, batch, hidden, 1e-5f);

        std::cout << "算子3: LayerNorm (1024×4096)\n";
        std::cout << "  True Naive: " << std::fixed << std::setprecision(3) << t_naive << " ms\n";
        std::cout << "  Optimized:  " << t_opt << " ms\n";
        std::cout << "  加速比: " << (t_naive/t_opt) << "×\n\n";

        cudaFree(d_in); cudaFree(d_out); cudaFree(d_gamma); cudaFree(d_beta);
    }

    // 4. RMSNorm
    {
        const int batch = 1024, hidden = 4096;
        half *d_in, *d_out, *d_weight;
        cudaMalloc(&d_in, batch*hidden*sizeof(half));
        cudaMalloc(&d_out, batch*hidden*sizeof(half));
        cudaMalloc(&d_weight, hidden*sizeof(half));

        float t_naive = benchmark(rmsnorm_true_naive_kernel, dim3((batch+255)/256), dim3(256), WARMUP, ITER, d_in, d_out, d_weight, batch, hidden, 1e-6f);
        float t_opt = benchmark(rmsnorm_optimized_kernel, dim3(batch), dim3(256), WARMUP, ITER, d_in, d_out, d_weight, batch, hidden, 1e-6f);

        std::cout << "算子4: RMSNorm (1024×4096)\n";
        std::cout << "  True Naive: " << std::fixed << std::setprecision(3) << t_naive << " ms\n";
        std::cout << "  Optimized:  " << t_opt << " ms\n";
        std::cout << "  加速比: " << (t_naive/t_opt) << "×\n\n";

        cudaFree(d_in); cudaFree(d_out); cudaFree(d_weight);
    }

    std::cout << "================================================================================\n";
    return 0;
}
