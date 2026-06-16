// ============================================================================
// Softmax 验证程序
// 功能: 验证Softmax final版本的正确性
// ============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>

namespace cg = cooperative_groups;

// ============================================================================
// 复制Softmax实现
// ============================================================================

// 配置参数
constexpr int WARP_SIZE = 32;
constexpr int BLOCK_SIZE = 256;  // 8 warps per block
constexpr int WARPS_PER_BLOCK = BLOCK_SIZE / WARP_SIZE;
constexpr int VEC_SIZE = 4;      // float4 向量化

// Warp-level reduction：计算 warp 内最大值
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Warp-level reduction：计算 warp 内求和
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block-level reduction：跨 warp 归约（使用共享内存）
template<typename ReduceOp>
__device__ __forceinline__ float block_reduce(
    float val,
    float* shared,
    ReduceOp op,
    float init_val
) {
    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;

    // Warp 内归约
    float warp_val = val;
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        warp_val = op(warp_val, __shfl_down_sync(0xffffffff, warp_val, offset));
    }

    // Warp leader 写入共享内存
    if (lane_id == 0) {
        shared[warp_id] = warp_val;
    }
    __syncthreads();

    // 第一个 warp 完成最终归约
    if (warp_id == 0) {
        warp_val = (lane_id < WARPS_PER_BLOCK) ? shared[lane_id] : init_val;
        #pragma unroll
        for (int offset = WARPS_PER_BLOCK / 2; offset > 0; offset /= 2) {
            warp_val = op(warp_val, __shfl_down_sync(0xffffffff, warp_val, offset));
        }
    }
    __syncthreads();

    return __shfl_sync(0xffffffff, warp_val, 0);  // 广播给所有线程
}

// Online Softmax Kernel
__global__ void __launch_bounds__(BLOCK_SIZE)
softmax_online_kernel(
    const float* __restrict__ input,   // [B, N]
    float* __restrict__ output,         // [B, N]
    int batch_size,
    int seq_len
) {
    // 每个 block 处理一行
    int row = blockIdx.x;
    if (row >= batch_size) return;

    const float* input_row = input + row * seq_len;
    float* output_row = output + row * seq_len;

    // 共享内存：用于 block-level 归约
    __shared__ float shared_max[WARPS_PER_BLOCK];
    __shared__ float shared_sum[WARPS_PER_BLOCK];

    int tid = threadIdx.x;
    int lane_id = tid % WARP_SIZE;

    // Phase 1: 单遍扫描计算 max 和 sum（Online Softmax 算法）
    float thread_max = -INFINITY;
    float thread_sum = 0.0f;

    // 向量加载：每个线程处理 VEC_SIZE 个元素
    int vec_seq_len = seq_len / VEC_SIZE;
    int remaining = seq_len % VEC_SIZE;

    // 处理对齐的部分（float4 向量化）
    for (int i = tid; i < vec_seq_len; i += BLOCK_SIZE) {
        float4 data = reinterpret_cast<const float4*>(input_row)[i];

        // 展开处理 4 个元素（Online Softmax 更新）
        #pragma unroll
        for (int j = 0; j < VEC_SIZE; j++) {
            float val = reinterpret_cast<float*>(&data)[j];
            float old_max = thread_max;
            thread_max = fmaxf(thread_max, val);

            // Online 更新 sum（关键：避免数值溢出）
            // sum = sum * exp(old_max - new_max) + exp(val - new_max)
            thread_sum = thread_sum * expf(old_max - thread_max) + expf(val - thread_max);
        }
    }

    // 处理剩余元素
    for (int i = vec_seq_len * VEC_SIZE + tid; i < seq_len; i += BLOCK_SIZE) {
        float val = input_row[i];
        float old_max = thread_max;
        thread_max = fmaxf(thread_max, val);
        thread_sum = thread_sum * expf(old_max - thread_max) + expf(val - thread_max);
    }

    // Phase 2: Block-level 归约 max
    auto max_op = [](float a, float b) { return fmaxf(a, b); };
    float block_max = block_reduce(thread_max, shared_max, max_op, -INFINITY);

    // Phase 3: 更新每个线程的 sum（基于全局 max）
    thread_sum = thread_sum * expf(thread_max - block_max);

    // Block-level 归约 sum
    auto sum_op = [](float a, float b) { return a + b; };
    float block_sum = block_reduce(thread_sum, shared_sum, sum_op, 0.0f);

    // Phase 4: 归一化并写回（单遍扫描）
    float inv_sum = 1.0f / block_sum;

    // 向量化写回
    for (int i = tid; i < vec_seq_len; i += BLOCK_SIZE) {
        float4 data = reinterpret_cast<const float4*>(input_row)[i];
        float4 result;

        #pragma unroll
        for (int j = 0; j < VEC_SIZE; j++) {
            float val = reinterpret_cast<float*>(&data)[j];
            reinterpret_cast<float*>(&result)[j] = expf(val - block_max) * inv_sum;
        }

        reinterpret_cast<float4*>(output_row)[i] = result;
    }

    // 处理剩余元素
    for (int i = vec_seq_len * VEC_SIZE + tid; i < seq_len; i += BLOCK_SIZE) {
        output_row[i] = expf(input_row[i] - block_max) * inv_sum;
    }
}

// Host 接口函数
void softmax_online(
    const float* d_input,
    float* d_output,
    int batch_size,
    int seq_len,
    cudaStream_t stream = 0
) {
    // 配置 kernel
    dim3 grid(batch_size);
    dim3 block(BLOCK_SIZE);

    // 启动 kernel
    softmax_online_kernel<<<grid, block, 0, stream>>>(
        d_input, d_output, batch_size, seq_len
    );

    // 错误检查
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Softmax kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}

// ============================================================================
// 验证函数
// ============================================================================

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

// 生成随机数据
void generate_random_data(float* data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;  // [-1, 1]
    }
}

bool verify_softmax() {
    printf("\n=== 验证 Softmax ===\n");

    const int test_cases[][2] = {
        {1, 10},      // seq_len=10
        {1, 100},     // seq_len=100
        {1, 1000},    // seq_len=1000
        {32, 128},    // batch=32, seq_len=128
        {8, 512}      // batch=8, seq_len=512
    };

    bool all_passed = true;

    for (int t = 0; t < 5; t++) {
        int batch_size = test_cases[t][0];
        int seq_len = test_cases[t][1];
        int size = batch_size * seq_len;

        printf("\n测试 [batch=%d, seq_len=%d]: ", batch_size, seq_len);

        // 分配内存
        float *h_input = new float[size];
        float *h_output = new float[size];
        float *d_input, *d_output;

        generate_random_data(h_input, size);

        CUDA_CHECK(cudaMalloc(&d_input, size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_output, size * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice));

        // 调用kernel
        softmax_online(d_input, d_output, batch_size, seq_len);
        CUDA_CHECK(cudaDeviceSynchronize());

        // 拷贝结果
        CUDA_CHECK(cudaMemcpy(h_output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost));

        // 验证: 每行和应为1.0
        bool passed = true;
        float max_error = 0.0f;

        for (int b = 0; b < batch_size; b++) {
            float sum = 0.0f;
            for (int i = 0; i < seq_len; i++) {
                float val = h_output[b * seq_len + i];
                sum += val;

                // 检查是否有NaN或负数
                if (isnan(val) || val < 0.0f) {
                    passed = false;
                    printf("错误: 输出包含无效值 (NaN或负数)\n");
                    break;
                }
            }

            float error = fabs(sum - 1.0f);
            max_error = fmax(max_error, error);

            if (error > 1e-3f) {  // 允许0.1%的误差
                passed = false;
                printf("错误: 行%d的和=%f (期望1.0, 误差=%f)\n", b, sum, error);
            }
        }

        if (passed) {
            printf("✅ 通过 (最大误差: %.6f)\n", max_error);
        } else {
            printf("❌ 失败\n");
            all_passed = false;
        }

        // 清理
        delete[] h_input;
        delete[] h_output;
        cudaFree(d_input);
        cudaFree(d_output);
    }

    return all_passed;
}

// ============================================================================
// 主函数
// ============================================================================
int main() {
    printf("============================================\n");
    printf("        Softmax 验证程序\n");
    printf("============================================\n");

    // 检查CUDA设备
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        printf("错误: 没有找到CUDA设备\n");
        return 1;
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("\n使用设备: %s\n", prop.name);
    printf("计算能力: %d.%d\n", prop.major, prop.minor);

    // 运行测试
    bool softmax_ok = verify_softmax();

    // 总结
    printf("\n============================================\n");
    printf("              验证结果\n");
    printf("============================================\n");
    printf("Softmax:   %s\n", softmax_ok ? "✅ 通过" : "❌ 失败");
    printf("============================================\n");

    return softmax_ok ? 0 : 1;
}
