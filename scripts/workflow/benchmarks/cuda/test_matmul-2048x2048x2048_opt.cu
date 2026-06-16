
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// 导入优化的wrapper函数
void launch_matmul_optimized(
    const half* d_A, const half* d_B, half* d_C,
    int M, int N, int K, cudaStream_t stream
);

int main() {
    const int M = 2048, N = 2048, K = 2048;

    // 分配内存
    half *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(half));
    cudaMalloc(&d_B, K * N * sizeof(half));
    cudaMalloc(&d_C, M * N * sizeof(half));

    // 初始化数据
    half *h_A = (half*)malloc(M * K * sizeof(half));
    half *h_B = (half*)malloc(K * N * sizeof(half));
    for (int i = 0; i < M * K; i++) h_A[i] = __float2half(0.1f);
    for (int i = 0; i < K * N; i++) h_B[i] = __float2half(0.2f);

    cudaMemcpy(d_A, h_A, M * K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(half), cudaMemcpyHostToDevice);

    // Warmup
    for (int i = 0; i < 10; i++) {
        launch_matmul_optimized(d_A, d_B, d_C, M, N, K, 0);
    }
    cudaDeviceSynchronize();

    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int iterations = 100;
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        launch_matmul_optimized(d_A, d_B, d_C, M, N, K, 0);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float avg_time = milliseconds / iterations;

    printf("MatMul Performance:\n");
    printf("  Average time: %.3f ms\n", avg_time);
    printf("  TFLOPS: %.2f\n", (2.0 * M * N * K * 1e-9) / (avg_time * 1e-3));

    // 清理
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B);

    return 0;
}
