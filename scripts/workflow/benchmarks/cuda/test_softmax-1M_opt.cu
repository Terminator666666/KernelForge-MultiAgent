
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// 导入优化的wrapper函数
void launch_softmax_optimized(
    const float* d_input, float* d_output,
    int batch_size, int seq_len, cudaStream_t stream
);

int main() {
    const int batch = 1, seq_len = 1000000;  // 1M elements

    float *d_input, *d_output;
    size_t data_size = batch * seq_len * sizeof(float);

    cudaMalloc(&d_input, data_size);
    cudaMalloc(&d_output, data_size);

    // 初始化数据
    float *h_input = (float*)malloc(data_size);
    for (int i = 0; i < batch * seq_len; i++) {
        h_input[i] = (float)(rand() % 100) / 100.0f;
    }
    cudaMemcpy(d_input, h_input, data_size, cudaMemcpyHostToDevice);

    // Warmup
    for (int i = 0; i < 10; i++) {
        launch_softmax_optimized(d_input, d_output, batch, seq_len, 0);
    }
    cudaDeviceSynchronize();

    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int iterations = 100;
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        launch_softmax_optimized(d_input, d_output, batch, seq_len, 0);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float avg_time = milliseconds / iterations;

    printf("Softmax Performance:\n");
    printf("  Average time: %.3f ms\n", avg_time);
    printf("  Throughput: %.2f GB/s\n", (2.0 * batch * seq_len * sizeof(float) / 1e9) / (avg_time / 1000));

    cudaFree(d_input); cudaFree(d_output);
    free(h_input);
    return 0;
}
