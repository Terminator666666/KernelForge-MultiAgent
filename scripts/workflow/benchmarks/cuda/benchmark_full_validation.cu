// ============================================================================
// 完整验证方案 - 对比 CPU 参考实现
// ============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <cmath>

// ============================================================================
// Softmax Kernels
// ============================================================================

#define WARP_SIZE 32
#define BLOCK_SIZE 256
#define WARPS_PER_BLOCK (BLOCK_SIZE / WARP_SIZE)

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

// True Naive Softmax
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

// Optimized Softmax - 使用标准 expf 而非 __expf
__global__ void softmax_optimized_kernel(const float* __restrict__ input, float* __restrict__ output,
                                          int batch_size, int seq_len) {
    int row = blockIdx.x;
    if (row >= batch_size) return;

    const float* row_input = input + row * seq_len;
    float* row_output = output + row * seq_len;

    __shared__ float smem[WARPS_PER_BLOCK];

    int tid = threadIdx.x;
    int lane_id = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;

    // Phase 1: 找最大值
    float thread_max = -INFINITY;
    for (int i = tid; i < seq_len; i += BLOCK_SIZE) {
        thread_max = fmaxf(thread_max, row_input[i]);
    }

    thread_max = warp_reduce_max(thread_max);
    if (lane_id == 0) smem[warp_id] = thread_max;
    __syncthreads();

    float row_max = -INFINITY;
    if (tid < WARPS_PER_BLOCK) row_max = smem[tid];
    row_max = warp_reduce_max(row_max);
    if (tid == 0) smem[0] = row_max;
    __syncthreads();
    row_max = smem[0];

    // Phase 2: 计算 exp 和 (使用标准 expf)
    float thread_sum = 0.0f;
    for (int i = tid; i < seq_len; i += BLOCK_SIZE) {
        thread_sum += expf(row_input[i] - row_max);
    }

    thread_sum = warp_reduce_sum(thread_sum);
    if (lane_id == 0) smem[warp_id] = thread_sum;
    __syncthreads();

    float row_sum = 0.0f;
    if (tid < WARPS_PER_BLOCK) row_sum = smem[tid];
    row_sum = warp_reduce_sum(row_sum);
    if (tid == 0) smem[0] = row_sum;
    __syncthreads();
    row_sum = smem[0];

    // Phase 3: 归一化
    for (int i = tid; i < seq_len; i += BLOCK_SIZE) {
        row_output[i] = expf(row_input[i] - row_max) / row_sum;
    }
}

// ============================================================================
// CPU 参考实现
// ============================================================================

void softmax_cpu_reference(const float* input, float* output, int batch_size, int seq_len) {
    for (int b = 0; b < batch_size; b++) {
        const float* row_in = input + b * seq_len;
        float* row_out = output + b * seq_len;

        // 找最大值
        float max_val = -INFINITY;
        for (int i = 0; i < seq_len; i++) {
            max_val = std::max(max_val, row_in[i]);
        }

        // 计算 exp 和
        float sum = 0.0f;
        for (int i = 0; i < seq_len; i++) {
            sum += std::exp(row_in[i] - max_val);
        }

        // 归一化
        for (int i = 0; i < seq_len; i++) {
            row_out[i] = std::exp(row_in[i] - max_val) / sum;
        }
    }
}

// ============================================================================
// 辅助函数
// ============================================================================

void init_random_data(float* data, int size, float min_val = -5.0f, float max_val = 5.0f) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(min_val, max_val);
    for (int i = 0; i < size; i++) {
        data[i] = dis(gen);
    }
}

bool compare_results(const float* result1, const float* result2, int size,
                     float rel_tol = 1e-4f, float abs_tol = 1e-5f) {
    int mismatches = 0;
    float max_rel_error = 0.0f;

    for (int i = 0; i < size; i++) {
        float diff = std::abs(result1[i] - result2[i]);
        float rel_error = diff / (std::abs(result2[i]) + 1e-8f);

        if (rel_error > rel_tol && diff > abs_tol) {
            if (mismatches < 5) {  // 只打印前5个错误
                std::cout << "    [" << i << "] GPU=" << result1[i]
                          << " CPU=" << result2[i]
                          << " diff=" << diff
                          << " rel_err=" << rel_error << "\n";
            }
            mismatches++;
        }
        max_rel_error = std::max(max_rel_error, rel_error);
    }

    if (mismatches > 0) {
        std::cout << "  总计 " << mismatches << " 个不匹配 (相对误差 > " << rel_tol << ")\n";
        std::cout << "  最大相对误差: " << max_rel_error << "\n";
    }

    return mismatches == 0;
}

bool verify_softmax_sum(const float* output, int batch_size, int seq_len, float tolerance = 2e-3f) {
    bool passed = true;
    for (int b = 0; b < batch_size; b++) {
        float sum = 0.0f;
        for (int i = 0; i < seq_len; i++) {
            sum += output[b * seq_len + i];
        }
        if (std::abs(sum - 1.0f) > tolerance) {
            std::cout << "  行 " << b << " sum = " << sum << " (误差: " << std::abs(sum - 1.0f) << ")\n";
            passed = false;
        }
    }
    return passed;
}

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
    std::cout << "  Softmax 加速比验证测试（包含 CPU 参考实现对比）\n";
    std::cout << "================================================================================\n\n";

    const int batch = 1, seq_len = 1000000;
    const int WARMUP = 10, ITER = 100;

    // 分配内存
    std::vector<float> h_input(batch * seq_len);
    std::vector<float> h_output_cpu(batch * seq_len);
    std::vector<float> h_output_naive(batch * seq_len);
    std::vector<float> h_output_opt(batch * seq_len);

    float *d_in, *d_out_naive, *d_out_opt;
    cudaMalloc(&d_in, batch * seq_len * sizeof(float));
    cudaMalloc(&d_out_naive, batch * seq_len * sizeof(float));
    cudaMalloc(&d_out_opt, batch * seq_len * sizeof(float));

    // 初始化数据
    init_random_data(h_input.data(), batch * seq_len, -5.0f, 5.0f);
    cudaMemcpy(d_in, h_input.data(), batch * seq_len * sizeof(float), cudaMemcpyHostToDevice);

    std::cout << "1️⃣  运行 CPU 参考实现...\n";
    softmax_cpu_reference(h_input.data(), h_output_cpu.data(), batch, seq_len);
    bool cpu_sum_check = verify_softmax_sum(h_output_cpu.data(), batch, seq_len);
    std::cout << "  CPU 和验证: " << (cpu_sum_check ? "✅ 通过" : "❌ 失败") << "\n\n";

    std::cout << "2️⃣  运行 GPU Naive 版本...\n";
    float t_naive = benchmark(softmax_true_naive_kernel,
                               dim3((batch+255)/256), dim3(256),
                               WARMUP, ITER,
                               d_in, d_out_naive, batch, seq_len);
    cudaMemcpy(h_output_naive.data(), d_out_naive, batch * seq_len * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "  性能: " << std::fixed << std::setprecision(3) << t_naive << " ms\n";
    std::cout << "  和验证: ";
    bool naive_sum_check = verify_softmax_sum(h_output_naive.data(), batch, seq_len);
    std::cout << (naive_sum_check ? "✅ 通过" : "❌ 失败") << "\n";

    std::cout << "  vs CPU: ";
    bool naive_vs_cpu = compare_results(h_output_naive.data(), h_output_cpu.data(), batch * seq_len);
    std::cout << (naive_vs_cpu ? "✅ 匹配" : "⚠️  有差异") << "\n\n";

    std::cout << "3️⃣  运行 GPU Optimized 版本...\n";
    float t_opt = benchmark(softmax_optimized_kernel,
                            dim3(batch), dim3(256),
                            WARMUP, ITER,
                            d_in, d_out_opt, batch, seq_len);
    cudaMemcpy(h_output_opt.data(), d_out_opt, batch * seq_len * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "  性能: " << t_opt << " ms\n";
    std::cout << "  和验证: ";
    bool opt_sum_check = verify_softmax_sum(h_output_opt.data(), batch, seq_len);
    std::cout << (opt_sum_check ? "✅ 通过" : "❌ 失败") << "\n";

    std::cout << "  vs CPU: ";
    bool opt_vs_cpu = compare_results(h_output_opt.data(), h_output_cpu.data(), batch * seq_len);
    std::cout << (opt_vs_cpu ? "✅ 匹配" : "⚠️  有差异") << "\n\n";

    std::cout << "================================================================================\n";
    std::cout << "📊 性能汇总\n";
    std::cout << "================================================================================\n";
    std::cout << "  True Naive: " << t_naive << " ms\n";
    std::cout << "  Optimized:  " << t_opt << " ms\n";
    std::cout << "  加速比:     " << (t_naive / t_opt) << "×\n";
    std::cout << "\n";
    std::cout << "  精度验证: " << (naive_vs_cpu && opt_vs_cpu ? "✅ 所有版本与 CPU 一致" : "⚠️  存在精度差异") << "\n";
    std::cout << "================================================================================\n";

    cudaFree(d_in);
    cudaFree(d_out_naive);
    cudaFree(d_out_opt);

    return 0;
}
