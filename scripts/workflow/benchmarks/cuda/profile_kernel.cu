#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>

extern "C" void launch_flash_attention(
    const half* Q, const half* K, const half* V, half* O,
    int batch_size, int num_heads, int seq_len, int head_dim, cudaStream_t stream);

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA错误 %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

void init_random_half(half* data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = __float2half((float)rand() / RAND_MAX * 2.0f - 1.0f);
    }
}

int main() {
    // 配置参数
    const int batch_size = 1;
    const int num_heads = 8;
    const int seq_len = 2048;
    const int head_dim = 64;
    
    printf("=== Flash Attention 详细性能分析 ===\n");
    printf("配置: batch=%d, heads=%d, seq_len=%d, head_dim=%d\n\n",
           batch_size, num_heads, seq_len, head_dim);
    
    // 计算数据大小
    const int qkv_size = batch_size * num_heads * seq_len * head_dim;
    const size_t bytes = qkv_size * sizeof(half);
    
    // 理论性能计算
    // FLOPs = batch * heads * (2 * seq^2 * head_dim + 2 * seq * head_dim)
    //       ≈ batch * heads * 2 * seq^2 * head_dim (主要是QK^T和Softmax@V)
    long long flops_per_call = 2LL * batch_size * num_heads * seq_len * seq_len * head_dim;
    flops_per_call += 2LL * batch_size * num_heads * seq_len * seq_len;  // softmax
    
    // 内存访问量 = 读取Q,K,V + 写入O = 4 * qkv_size * sizeof(half)
    long long memory_bytes = 4LL * qkv_size * sizeof(half);
    
    // 计算强度 (FLOPs/Byte)
    double compute_intensity = (double)flops_per_call / memory_bytes;
    
    printf("理论分析:\n");
    printf("  FLOPs/call: %.2f GFLOPs\n", flops_per_call / 1e9);
    printf("  Memory/call: %.2f MB\n", memory_bytes / 1024.0 / 1024.0);
    printf("  计算强度: %.2f FLOPs/Byte\n", compute_intensity);
    printf("  (>10: compute-bound, <10: memory-bound)\n\n");
    
    // 获取GPU属性
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    
    printf("GPU信息:\n");
    printf("  名称: %s\n", prop.name);
    printf("  计算能力: %d.%d\n", prop.major, prop.minor);
    printf("  SM数量: %d\n", prop.multiProcessorCount);
    printf("  全局内存: %.2f GB\n", prop.totalGlobalMem / 1024.0 / 1024.0 / 1024.0);
    printf("  Shared Memory/Block: %.2f KB\n", prop.sharedMemPerBlock / 1024.0);
    printf("  峰值内存带宽: %.2f GB/s\n", 
           2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1e6);
    printf("\n");
    
    // 分配内存
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
    
    // 计算实际性能
    double actual_tflops = (flops_per_call / (avg_ms / 1000.0)) / 1e12;
    double actual_bandwidth = (memory_bytes / (avg_ms / 1000.0)) / 1e9;
    double peak_bandwidth = 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1e6;
    double bandwidth_utilization = (actual_bandwidth / peak_bandwidth) * 100.0;
    
    printf("=== 性能结果 ===\n");
    printf("平均延迟: %.4f ms\n", avg_ms);
    printf("吞吐量: %.2f TFLOPS\n", actual_tflops);
    printf("实际带宽: %.2f GB/s\n", actual_bandwidth);
    printf("带宽利用率: %.2f%%\n", bandwidth_utilization);
    printf("\n");
    
    // 瓶颈分析
    printf("=== 瓶颈分析 ===\n");
    if (compute_intensity > 10) {
        printf("计算强度: %.2f FLOPs/Byte -> Compute-bound\n", compute_intensity);
        printf("建议优化方向:\n");
        printf("  1. 使用Tensor Core加速矩阵乘法\n");
        printf("  2. 提高SM占用率\n");
        printf("  3. 减少warp divergence\n");
    } else {
        printf("计算强度: %.2f FLOPs/Byte -> Memory-bound\n", compute_intensity);
        printf("建议优化方向:\n");
        printf("  1. 向量化内存访问\n");
        printf("  2. 增大tile size以提高数据重用\n");
        printf("  3. 使用异步内存拷贝隐藏延迟\n");
    }
    
    if (bandwidth_utilization < 50) {
        printf("\n带宽利用率偏低 (%.2f%%)，主要瓶颈在内存访问\n", bandwidth_utilization);
    } else if (bandwidth_utilization > 80) {
        printf("\n带宽利用率很高 (%.2f%%)，已接近内存带宽上限\n", bandwidth_utilization);
    } else {
        printf("\n带宽利用率中等 (%.2f%%)，有优化空间\n", bandwidth_utilization);
    }
    
    // 清理
    CHECK_CUDA(cudaFree(d_Q));
    CHECK_CUDA(cudaFree(d_K));
    CHECK_CUDA(cudaFree(d_V));
    CHECK_CUDA(cudaFree(d_O));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    
    free(h_Q);
    free(h_K);
    free(h_V);
    free(h_O);
    
    return 0;
}
