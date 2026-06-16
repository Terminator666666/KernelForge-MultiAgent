
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// 导入优化的wrapper函数
void launch_layernorm_optimized(
    const float* input, float* output,
    const float* gamma, const float* beta,
    int batch_size, int hidden_size,
    float eps, cudaStream_t stream
);

int main() {
    const int batch = 1024, hidden_dim = 4096;
    const float eps = 1e-5f;

    float *d_input, *d_output, *d_gamma, *d_beta;
    size_t data_size = batch * hidden_dim * sizeof(float);
    size_t param_size = hidden_dim * sizeof(float);

    cudaMalloc(&d_input, data_size);
    cudaMalloc(&d_output, data_size);
    cudaMalloc(&d_gamma, param_size);
    cudaMalloc(&d_beta, param_size);

    cudaMemset(d_input, 0, data_size);
    cudaMemset(d_gamma, 0, param_size);
    cudaMemset(d_beta, 0, param_size);

    // Warmup
    for (int i = 0; i < 10; i++) {
        launch_layernorm_optimized(d_input, d_output, d_gamma, d_beta, batch, hidden_dim, eps, 0);
    }
    cudaDeviceSynchronize();

    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int iterations = 100;
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        launch_layernorm_optimized(d_input, d_output, d_gamma, d_beta, batch, hidden_dim, eps, 0);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float avg_time = milliseconds / iterations;

    printf("LayerNorm Performance:\n");
    printf("  Average time: %.3f ms\n", avg_time);

    cudaFree(d_input); cudaFree(d_output);
    cudaFree(d_gamma); cudaFree(d_beta);
    return 0;
}
