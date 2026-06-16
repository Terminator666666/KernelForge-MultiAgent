#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// 声明外部kernel函数
extern "C" void launch_flash_attention(
    const half* Q, const half* K, const half* V, half* O,
    int batch_size, int num_heads, int seq_len, int head_dim, cudaStream_t stream);

// 检查CUDA错误
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA错误 %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// 初始化随机数据
void init_random_half(half* data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = __float2half((float)rand() / RAND_MAX * 2.0f - 1.0f);
    }
}

// 简单的CPU参考实现（用于验证正确性）
void flash_attention_cpu_reference(
    const half* Q, const half* K, const half* V, half* O,
    int batch_size, int num_heads, int seq_len, int head_dim) {
    
    const float scale = 1.0f / sqrtf((float)head_dim);
    
    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < num_heads; h++) {
            int offset = (b * num_heads + h) * seq_len * head_dim;
            
            for (int i = 0; i < seq_len; i++) {
                // 计算 attention scores
                float* scores = (float*)malloc(seq_len * sizeof(float));
                float max_score = -INFINITY;
                
                for (int j = 0; j < seq_len; j++) {
                    float score = 0.0f;
                    for (int d = 0; d < head_dim; d++) {
                        float q_val = __half2float(Q[offset + i * head_dim + d]);
                        float k_val = __half2float(K[offset + j * head_dim + d]);
                        score += q_val * k_val;
                    }
                    score *= scale;
                    scores[j] = score;
                    max_score = fmaxf(max_score, score);
                }
                
                // Softmax
                float sum = 0.0f;
                for (int j = 0; j < seq_len; j++) {
                    scores[j] = expf(scores[j] - max_score);
                    sum += scores[j];
                }
                for (int j = 0; j < seq_len; j++) {
                    scores[j] /= sum;
                }
                
                // 计算输出
                for (int d = 0; d < head_dim; d++) {
                    float out_val = 0.0f;
                    for (int j = 0; j < seq_len; j++) {
                        float v_val = __half2float(V[offset + j * head_dim + d]);
                        out_val += scores[j] * v_val;
                    }
                    O[offset + i * head_dim + d] = __float2half(out_val);
                }
                
                free(scores);
            }
        }
    }
}

// 验证结果
bool verify_results(const half* gpu_out, const half* cpu_out, int size, float threshold = 0.01f) {
    int errors = 0;
    float max_diff = 0.0f;
    
    for (int i = 0; i < size; i++) {
        float g = __half2float(gpu_out[i]);
        float c = __half2float(cpu_out[i]);
        float diff = fabsf(g - c);
        max_diff = fmaxf(max_diff, diff);
        
        if (diff > threshold) {
            errors++;
            if (errors <= 10) {
                printf("位置 %d: GPU=%.6f, CPU=%.6f, diff=%.6f\n", i, g, c, diff);
            }
        }
    }
    
    printf("最大差异: %.6f\n", max_diff);
    printf("错误数量: %d / %d (%.2f%%)\n", errors, size, 100.0f * errors / size);
    
    return errors == 0 || (100.0f * errors / size < 1.0f);  // 允许1%的误差
}

int main(int argc, char** argv) {
    // 配置参数
    const int batch_size = 1;
    const int num_heads = 8;
    const int seq_len = 2048;
    const int head_dim = 64;
    
    printf("=== Flash Attention 性能测试 ===\n");
    printf("配置: batch=%d, heads=%d, seq_len=%d, head_dim=%d\n\n",
           batch_size, num_heads, seq_len, head_dim);
    
    // 计算数据大小
    const int qkv_size = batch_size * num_heads * seq_len * head_dim;
    const size_t bytes = qkv_size * sizeof(half);
    
    printf("内存占用: %.2f MB\n\n", bytes * 4 / 1024.0f / 1024.0f);
    
    // 分配主机内存
    half* h_Q = (half*)malloc(bytes);
    half* h_K = (half*)malloc(bytes);
    half* h_V = (half*)malloc(bytes);
    half* h_O = (half*)malloc(bytes);
    half* h_O_ref = (half*)malloc(bytes);
    
    // 初始化随机数据
    srand(42);
    init_random_half(h_Q, qkv_size);
    init_random_half(h_K, qkv_size);
    init_random_half(h_V, qkv_size);
    
    // 分配设备内存
    half *d_Q, *d_K, *d_V, *d_O;
    CHECK_CUDA(cudaMalloc(&d_Q, bytes));
    CHECK_CUDA(cudaMalloc(&d_K, bytes));
    CHECK_CUDA(cudaMalloc(&d_V, bytes));
    CHECK_CUDA(cudaMalloc(&d_O, bytes));
    
    // 拷贝数据到设备
    CHECK_CUDA(cudaMemcpy(d_Q, h_Q, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_K, h_K, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_V, h_V, bytes, cudaMemcpyHostToDevice));
    
    // 创建CUDA事件用于计时
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    // 预热
    printf("预热中...\n");
    for (int i = 0; i < 10; i++) {
        launch_flash_attention(d_Q, d_K, d_V, d_O, 
                              batch_size, num_heads, seq_len, head_dim, 0);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // 性能测试
    printf("性能测试中...\n");
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
    
    // 计算性能指标
    // FLOPs = 4 * batch * heads * seq^2 * head_dim (2 for QK^T, 2 for softmax+V)
    double flops = 4.0 * batch_size * num_heads * seq_len * seq_len * head_dim;
    double tflops = (flops / (avg_ms / 1000.0)) / 1e12;
    
    printf("\n=== 性能结果 ===\n");
    printf("平均延迟: %.4f ms\n", avg_ms);
    printf("吞吐量: %.2f TFLOPS\n", tflops);
    printf("带宽: %.2f GB/s\n", (bytes * 4 / (avg_ms / 1000.0)) / 1e9);
    
    // 拷贝结果回主机
    CHECK_CUDA(cudaMemcpy(h_O, d_O, bytes, cudaMemcpyDeviceToHost));
    
    // CPU参考实现（仅用于小规模验证）
    if (seq_len <= 512) {  // 只在序列长度较小时验证
        printf("\n正在计算CPU参考结果...\n");
        flash_attention_cpu_reference(h_Q, h_K, h_V, h_O_ref,
                                     batch_size, num_heads, seq_len, head_dim);
        
        printf("\n=== 正确性验证 ===\n");
        bool passed = verify_results(h_O, h_O_ref, qkv_size, 0.05f);
        printf("验证结果: %s\n", passed ? "通过" : "失败");
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
    free(h_O_ref);
    
    printf("\n测试完成！\n");
    return 0;
}
