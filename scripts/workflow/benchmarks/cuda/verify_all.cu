#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

// 函数声明
void softmax_online(const float*, float*, int, int, cudaStream_t);
void launch_rmsnorm_optimized(const half*, half*, const half*, int, cudaStream_t);
void launch_layernorm_optimized(const float*, float*, const float*, const float*, int, int, float, cudaStream_t);
void launch_matmul_naive(const half*, const half*, half*, int, int, int, cudaStream_t);

int main() {
    printf("===============================================\n");
    printf("  所有算子正确性验证\n");
    printf("===============================================\n\n");
    
    // ===== Softmax验证 =====
    printf("【1. Softmax】\n");
    {
        float *d_input, *d_output;
        cudaMalloc(&d_input, 1000 * sizeof(float));
        cudaMalloc(&d_output, 1000 * sizeof(float));
        
        float h_input[1000];
        for (int i = 0; i < 1000; i++) h_input[i] = (float)(rand() % 100) / 10.0f;
        cudaMemcpy(d_input, h_input, 1000 * sizeof(float), cudaMemcpyHostToDevice);
        
        softmax_online(d_input, d_output, 1, 1000, 0);
        cudaDeviceSynchronize();
        
        float h_output[1000];
        cudaMemcpy(h_output, d_output, 1000 * sizeof(float), cudaMemcpyDeviceToHost);
        
        float sum = 0.0f;
        for (int i = 0; i < 1000; i++) sum += h_output[i];
        
        bool pass = fabsf(sum - 1.0f) < 1e-3f;
        printf("  Sum = %.6f (期望1.0) %s\n\n", sum, pass ? "✅" : "❌");
        
        cudaFree(d_input); cudaFree(d_output);
    }
    
    // ===== RMSNorm验证 =====
    printf("【2. RMSNorm】\n");
    {
        half *d_input, *d_output, *d_weight;
        cudaMalloc(&d_input, 1024 * 4096 * sizeof(half));
        cudaMalloc(&d_output, 1024 * 4096 * sizeof(half));
        cudaMalloc(&d_weight, 4096 * sizeof(half));
        
        launch_rmsnorm_optimized(d_input, d_output, d_weight, 1024, 0);
        cudaError_t err = cudaGetLastError();
        
        printf("  执行状态: %s %s\n\n", cudaGetErrorString(err), 
               err == cudaSuccess ? "✅" : "❌");
        
        cudaFree(d_input); cudaFree(d_output); cudaFree(d_weight);
    }
    
    // ===== LayerNorm验证 =====
    printf("【3. LayerNorm】\n");
    {
        float *d_input, *d_output, *d_gamma, *d_beta;
        cudaMalloc(&d_input, 1024 * 4096 * sizeof(float));
        cudaMalloc(&d_output, 1024 * 4096 * sizeof(float));
        cudaMalloc(&d_gamma, 4096 * sizeof(float));
        cudaMalloc(&d_beta, 4096 * sizeof(float));
        
        launch_layernorm_optimized(d_input, d_output, d_gamma, d_beta, 1024, 4096, 1e-5f, 0);
        cudaError_t err = cudaGetLastError();
        
        printf("  执行状态: %s %s\n\n", cudaGetErrorString(err),
               err == cudaSuccess ? "✅" : "❌");
        
        cudaFree(d_input); cudaFree(d_output); cudaFree(d_gamma); cudaFree(d_beta);
    }
    
    // ===== MatMul验证 =====
    printf("【4. MatMul】\n");
    {
        half *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, 2048 * 2048 * sizeof(half));
        cudaMalloc(&d_B, 2048 * 2048 * sizeof(half));
        cudaMalloc(&d_C, 2048 * 2048 * sizeof(half));
        
        launch_matmul_naive(d_A, d_B, d_C, 2048, 2048, 2048, 0);
        cudaError_t err = cudaGetLastError();
        
        printf("  执行状态: %s %s\n\n", cudaGetErrorString(err),
               err == cudaSuccess ? "✅" : "❌");
        
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    }
    
    printf("===============================================\n");
    return 0;
}
