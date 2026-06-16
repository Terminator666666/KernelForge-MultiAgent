#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>

// 从原始kernel导入
extern "C" void launch_flash_attention(
    const half* Q, const half* K, const half* V, half* O,
    int batch_size, int num_heads, int seq_len, int head_dim, cudaStream_t stream);

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA错误: %s\n", cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

void init_random_half(half* data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = __float2half((float)rand() / RAND_MAX * 2.0f - 1.0f);
    }
}

int main() {
    const int batch_size = 1;
    const int num_heads = 8;
    const int seq_len = 2048;
    const int head_dim = 64;
    
    printf("=== 原始Kernel性能测试 ===\n");
    printf("配置: batch=%d, heads=%d, seq_len=%d, head_dim=%d\n\n",
           batch_size, num_heads, seq_len, head_dim);
    
    const int qkv_size = batch_size * num_heads * seq_len * head_dim;
    const size_t bytes = qkv_size * sizeof(half);
    
    half *h_Q = (half*)malloc(bytes);
    half *h_K = (half*)malloc(bytes);
    half *h_V = (half*)malloc(bytes);
    half *h_O = (half*)malloc(bytes);
    
    srand(42);
    init_random_half(h_Q, qkv_size);
    init_random_half(h_K, qkv_size);
    init_random_half(h_V, qkv_size);
    
    half *d_Q, *d_K, *d_V, *d_O;
    CHECK_CUDA(cudaMalloc(&d_Q, bytes));
    CHECK_CUDA(cudaMalloc(&d_K, bytes));
    CHECK_CUDA(cudaMalloc(&d_V, bytes));
    CHECK_CUDA(cudaMalloc(&d_O, bytes));
    
    CHECK_CUDA(cudaMemcpy(d_Q, h_Q, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_K, h_K, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_V, h_V, bytes, cudaMemcpyHostToDevice));
    
    // 预热
    for (int i = 0; i < 10; i++) {
        launch_flash_attention(d_Q, d_K, d_V, d_O, 
                              batch_size, num_heads, seq_len, head_dim, 0);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // 性能测试
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    const int num_iters = 100;
    CHECK_CUDA(cudaEventRecord(start));
    
    for (int i = 0; i < num_iters; i++) {
        launch_flash_attention(d_Q, d_K, d_V, d_O,
                              batch_size, num_heads, seq_len, head_dim, 0);
    }
    
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    
    float ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    float avg_ms = ms / num_iters;
    
    printf("平均延迟: %.4f ms\n", avg_ms);
    
    CHECK_CUDA(cudaFree(d_Q));
    CHECK_CUDA(cudaFree(d_K));
    CHECK_CUDA(cudaFree(d_V));
    CHECK_CUDA(cudaFree(d_O));
    
    free(h_Q); free(h_K); free(h_V); free(h_O);
    
    return 0;
}
