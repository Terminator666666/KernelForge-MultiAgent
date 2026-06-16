// ============================================================================
// 完整的性能测试程序 - 包含数据初始化和精度验证
// ============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <random>

// ============================================================================
// 1. Softmax - 使用项目 final 版本
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

// Optimized Softmax (from softmax_final.cu)
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

    float thread_sum = 0.0f;
    for (int i = tid; i < seq_len; i += BLOCK_SIZE) {
        thread_sum += __expf(row_input[i] - row_max);
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

    for (int i = tid; i < seq_len; i += BLOCK_SIZE) {
        row_output[i] = __expf(row_input[i] - row_max) / row_sum;
    }
}

// ============================================================================
// 辅助函数
// ============================================================================

void init_random_data(float* data, int size, float min_val = -5.0f, float max_val = 5.0f) {
    std::random_device rd;
    std::mt19937 gen(42); // 固定种子保证可重复
    std::uniform_real_distribution<float> dis(min_val, max_val);
    for (int i = 0; i < size; i++) {
        data[i] = dis(gen);
    }
}

void init_random_data_half(half* data, int size, float min_val = -1.0f, float max_val = 1.0f) {
    std::random_device rd;
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(min_val, max_val);
    for (int i = 0; i < size; i++) {
        data[i] = __float2half(dis(gen));
    }
}

bool verify_softmax(const float* output, int batch_size, int seq_len, float tolerance = 1e-3f) {
    std::vector<float> h_output(batch_size * seq_len);
    cudaMemcpy(h_output.data(), output, batch_size * seq_len * sizeof(float), cudaMemcpyDeviceToHost);

    bool passed = true;
    for (int b = 0; b < batch_size; b++) {
        float sum = 0.0f;
        for (int i = 0; i < seq_len; i++) {
            float val = h_output[b * seq_len + i];
            if (val < 0.0f || val > 1.0f) {
                std::cout << "❌ 值超出范围 [0,1]: " << val << " at (" << b << "," << i << ")\n";
                passed = false;
            }
            sum += val;
        }
        if (fabs(sum - 1.0f) > tolerance) {
            std::cout << "❌ 行 " << b << " 的和不等于1: " << sum << "\n";
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
    std::cout << "  真实 Naive Baseline 加速比测试 (包含数据初始化和精度验证)\n";
    std::cout << "================================================================================\n\n";

    const int WARMUP = 10, ITER = 100;

    // 1. Softmax
    {
        std::cout << "算子1: Softmax (1M 元素)\n";
        std::cout << "----------------------------------------\n";

        const int batch = 1, seq_len = 1000000;

        // 分配内存
        float *d_in, *d_out_naive, *d_out_opt;
        cudaMalloc(&d_in, batch * seq_len * sizeof(float));
        cudaMalloc(&d_out_naive, batch * seq_len * sizeof(float));
        cudaMalloc(&d_out_opt, batch * seq_len * sizeof(float));

        // 初始化数据
        std::vector<float> h_input(batch * seq_len);
        init_random_data(h_input.data(), batch * seq_len, -5.0f, 5.0f);
        cudaMemcpy(d_in, h_input.data(), batch * seq_len * sizeof(float), cudaMemcpyHostToDevice);

        // Naive 版本测试
        float t_naive = benchmark(softmax_true_naive_kernel,
                                   dim3((batch+255)/256), dim3(256),
                                   WARMUP, ITER,
                                   d_in, d_out_naive, batch, seq_len);

        // Optimized 版本测试
        float t_opt = benchmark(softmax_optimized_kernel,
                                dim3(batch), dim3(256),
                                WARMUP, ITER,
                                d_in, d_out_opt, batch, seq_len);

        // 精度验证
        std::cout << "精度验证:\n";
        std::cout << "  Naive 版本: ";
        bool naive_pass = verify_softmax(d_out_naive, batch, seq_len);
        std::cout << (naive_pass ? "✅ 通过\n" : "❌ 失败\n");

        std::cout << "  Optimized 版本: ";
        bool opt_pass = verify_softmax(d_out_opt, batch, seq_len);
        std::cout << (opt_pass ? "✅ 通过\n" : "❌ 失败\n");

        std::cout << "\n性能结果:\n";
        std::cout << "  True Naive: " << std::fixed << std::setprecision(3) << t_naive << " ms\n";
        std::cout << "  Optimized:  " << t_opt << " ms\n";
        std::cout << "  加速比: " << (t_naive/t_opt) << "×\n\n";

        cudaFree(d_in);
        cudaFree(d_out_naive);
        cudaFree(d_out_opt);
    }

    std::cout << "================================================================================\n";
    std::cout << "注意: 本测试仅包含 Softmax，其他算子需要链接项目中的 final 版本\n";
    std::cout << "MatMul 需要使用 matmul/matmul_final.cu (带 Tensor Core)\n";
    std::cout << "LayerNorm 需要使用 layernorm/layernorm_final.cu\n";
    std::cout << "RMSNorm 需要使用 rmsnorm/rmsnorm_final.cu\n";
    std::cout << "================================================================================\n";

    return 0;
}
