#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// 声明外部kernel启动函数
extern "C" {
    void launch_matmul_optimized(
        const half* d_A, const half* d_B, half* d_C,
        int M, int N, int K, cudaStream_t stream
    );
}

// CPU参考实现（FP32用于验证）
void matmul_cpu(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// 验证结果正确性
bool verify_result(const half* C, const float* C_ref, int M, int N, float threshold = 0.01f) {
    int errors = 0;
    float max_error = 0.0f;

    for (int i = 0; i < M * N; i++) {
        float gpu_val = __half2float(C[i]);
        float cpu_val = C_ref[i];
        float error = fabs(gpu_val - cpu_val);

        if (error > threshold) {
            errors++;
            if (errors < 10) {  // 只打印前10个错误
                printf("Error at %d: GPU=%.6f, CPU=%.6f, diff=%.6f\n",
                       i, gpu_val, cpu_val, error);
            }
        }

        max_error = fmax(max_error, error);
    }

    printf("验证结果: 错误数=%d/%d, 最大误差=%.6f\n", errors, M * N, max_error);
    return (errors == 0 || (float)errors / (M * N) < 0.001f);  // 允许0.1%的误差
}

// Benchmark函数
double benchmark_kernel(
    const half* d_A, const half* d_B, half* d_C,
    int M, int N, int K,
    int warmup_runs = 10,
    int benchmark_runs = 100
) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup
    for (int i = 0; i < warmup_runs; i++) {
        launch_matmul_optimized(d_A, d_B, d_C, M, N, K, 0);
    }
    cudaDeviceSynchronize();

    // Benchmark
    cudaEventRecord(start);
    for (int i = 0; i < benchmark_runs; i++) {
        launch_matmul_optimized(d_A, d_B, d_C, M, N, K, 0);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return milliseconds / benchmark_runs;  // 返回平均时间（ms）
}

int main() {
    const int M = 2048;
    const int N = 2048;
    const int K = 2048;

    printf("=== MatMul 优化测试 ===\n");
    printf("矩阵尺寸: M=%d, N=%d, K=%d\n\n", M, N, K);

    // 打印设备信息
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Shared Memory per Block: %zu KB\n", prop.sharedMemPerBlock / 1024);
    printf("Max Threads per Block: %d\n\n", prop.maxThreadsPerBlock);

    // 分配Host内存
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    float* h_A = (float*)malloc(size_A);
    float* h_B = (float*)malloc(size_B);
    float* h_C_ref = (float*)malloc(size_C);
    half* h_C_gpu = (half*)malloc(M * N * sizeof(half));

    // 初始化输入数据（随机小数）
    srand(42);
    for (int i = 0; i < M * K; i++) {
        h_A[i] = (float)(rand() % 100) / 100.0f;
    }
    for (int i = 0; i < K * N; i++) {
        h_B[i] = (float)(rand() % 100) / 100.0f;
    }

    // 转换为FP16
    half* h_A_fp16 = (half*)malloc(M * K * sizeof(half));
    half* h_B_fp16 = (half*)malloc(K * N * sizeof(half));
    for (int i = 0; i < M * K; i++) {
        h_A_fp16[i] = __float2half(h_A[i]);
    }
    for (int i = 0; i < K * N; i++) {
        h_B_fp16[i] = __float2half(h_B[i]);
    }

    // 分配Device内存
    half *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(half));
    cudaMalloc(&d_B, K * N * sizeof(half));
    cudaMalloc(&d_C, M * N * sizeof(half));

    // 拷贝数据到Device
    cudaMemcpy(d_A, h_A_fp16, M * K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B_fp16, K * N * sizeof(half), cudaMemcpyHostToDevice);

    // === 1. CPU参考结果（仅用于小规模验证） ===
    if (M <= 512) {  // 仅对小矩阵计算CPU结果
        printf("计算CPU参考结果...\n");
        clock_t cpu_start = clock();
        matmul_cpu(h_A, h_B, h_C_ref, M, N, K);
        clock_t cpu_end = clock();
        double cpu_time = (double)(cpu_end - cpu_start) / CLOCKS_PER_SEC * 1000.0;
        printf("CPU时间: %.2f ms\n\n", cpu_time);
    } else {
        printf("矩阵过大，跳过CPU验证\n\n");
    }

    // === 2. GPU Kernel测试 ===
    printf("运行优化kernel...\n");
    launch_matmul_optimized(d_A, d_B, d_C, M, N, K, 0);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel启动失败: %s\n", cudaGetErrorString(err));
        return -1;
    }

    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel执行失败: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // 拷贝结果回Host
    cudaMemcpy(h_C_gpu, d_C, M * N * sizeof(half), cudaMemcpyDeviceToHost);

    // === 3. 验证正确性 ===
    if (M <= 512) {
        printf("\n正确性验证:\n");
        verify_result(h_C_gpu, h_C_ref, M, N, 0.1f);  // FP16误差容忍度更大
    }

    // === 4. 性能测试 ===
    printf("\n性能测试:\n");
    double avg_time = benchmark_kernel(d_A, d_B, d_C, M, N, K, 10, 100);

    // 计算性能指标
    double gflops = (2.0 * M * N * K) / (avg_time * 1e-3) / 1e9;
    double bandwidth = (M * K + K * N + M * N) * sizeof(half) / (avg_time * 1e-3) / 1e9;

    printf("平均时间: %.4f ms\n", avg_time);
    printf("性能: %.2f GFLOPS\n", gflops);
    printf("带宽: %.2f GB/s\n", bandwidth);

    // RTX 5070 理论峰值（估算）
    // FP16 Tensor Core: ~200 TFLOPS
    // 内存带宽: ~448 GB/s
    double peak_tflops = 200.0;  // TFLOPS
    double peak_bandwidth = 448.0;  // GB/s

    printf("\n效率分析:\n");
    printf("计算效率: %.2f%% (峰值 %.0f TFLOPS)\n",
           (gflops / 1000.0) / peak_tflops * 100.0, peak_tflops);
    printf("带宽效率: %.2f%% (峰值 %.0f GB/s)\n",
           bandwidth / peak_bandwidth * 100.0, peak_bandwidth);

    // 清理资源
    free(h_A);
    free(h_B);
    free(h_C_ref);
    free(h_C_gpu);
    free(h_A_fp16);
    free(h_B_fp16);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    printf("\n测试完成!\n");
    return 0;
}
