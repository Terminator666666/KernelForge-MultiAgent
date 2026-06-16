
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>

// 导入优化的wrapper函数
void launch_rmsnorm_optimized(
    const half* input, half* output,
    const half* weight, int batch_size, cudaStream_t stream
);

int main() {
    const int batch = 1024, hidden_dim = 4096;

    half *d_input, *d_output, *d_weight;
    size_t data_size = batch * hidden_dim * sizeof(half);
    size_t param_size = hidden_dim * sizeof(half);

    cudaMalloc(&d_input, data_size);
    cudaMalloc(&d_output, data_size);
    cudaMalloc(&d_weight, param_size);

    // 初始化数据
    half *h_input = (half*)malloc(data_size);
    half *h_weight = (half*)malloc(param_size);
    for (int i = 0; i < batch * hidden_dim; i++) {
        h_input[i] = __float2half((float)(rand() % 100) / 100.0f);
    }
    for (int i = 0; i < hidden_dim; i++) {
        h_weight[i] = __float2half(1.0f);
    }
    cudaMemcpy(d_input, h_input, data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, h_weight, param_size, cudaMemcpyHostToDevice);

    // Warmup
    for (int i = 0; i < 10; i++) {
        launch_rmsnorm_optimized(d_input, d_output, d_weight, batch, 0);
    }
    cudaDeviceSynchronize();

    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int iterations = 100;
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        launch_rmsnorm_optimized(d_input, d_output, d_weight, batch, 0);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float avg_time = milliseconds / iterations;

    printf("RMSNorm Performance:\n");
    printf("  Average time: %.3f ms\n", avg_time);
    printf("  Throughput: %.2f GB/s\n", (2.0 * batch * hidden_dim * sizeof(half) / 1e9) / (avg_time / 1000));

    cudaFree(d_input); cudaFree(d_output); cudaFree(d_weight);
    free(h_input); free(h_weight);
    return 0;
}
