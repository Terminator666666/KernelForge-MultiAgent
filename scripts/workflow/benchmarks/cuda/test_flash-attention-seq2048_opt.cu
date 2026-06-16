
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// 导入优化的wrapper函数
void launch_flash_attention(
    const half* Q, const half* K, const half* V, half* O,
    int batch_size, int num_heads, int seq_len, int head_dim, cudaStream_t stream
);

int main() {
    const int batch = 2, heads = 16, seq_len = 2048, head_dim = 64;

    half *d_Q, *d_K, *d_V, *d_O;
    size_t qkv_size = batch * heads * seq_len * head_dim * sizeof(half);

    cudaMalloc(&d_Q, qkv_size);
    cudaMalloc(&d_K, qkv_size);
    cudaMalloc(&d_V, qkv_size);
    cudaMalloc(&d_O, qkv_size);

    // 初始化数据
    cudaMemset(d_Q, 0, qkv_size);
    cudaMemset(d_K, 0, qkv_size);
    cudaMemset(d_V, 0, qkv_size);

    // Warmup
    for (int i = 0; i < 10; i++) {
        launch_flash_attention(d_Q, d_K, d_V, d_O, batch, heads, seq_len, head_dim, 0);
    }
    cudaDeviceSynchronize();

    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int iterations = 100;
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        launch_flash_attention(d_Q, d_K, d_V, d_O, batch, heads, seq_len, head_dim, 0);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float avg_time = milliseconds / iterations;

    printf("FlashAttention Performance:\n");
    printf("  Average time: %.3f ms\n", avg_time);

    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
    return 0;
}
