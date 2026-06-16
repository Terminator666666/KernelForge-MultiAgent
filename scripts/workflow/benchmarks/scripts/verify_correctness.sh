#!/bin/bash

echo "==============================================="
echo "  算子正确性和精度验证"
echo "==============================================="
echo ""

cd /mnt/d/Agent/KernelForge-MultiAgent

# ============================================
# Softmax验证
# ============================================
echo "【1. Softmax正确性验证】"
echo "-------------------------------------------"

cat > /tmp/verify_softmax.cu << 'EOF'
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

void launch_softmax_naive(const float*, float*, int, int, cudaStream_t);
void launch_softmax_optimized(const float*, float*, int, int, cudaStream_t);

int main() {
    const int batch = 1, seq_len = 1000000;

    float *d_input, *d_output_naive, *d_output_opt;
    float *h_input, *h_output_naive, *h_output_opt;

    size_t size = batch * seq_len * sizeof(float);

    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output_naive, size);
    cudaMalloc(&d_output_opt, size);

    h_input = (float*)malloc(size);
    h_output_naive = (float*)malloc(size);
    h_output_opt = (float*)malloc(size);

    // 生成测试数据
    for (int i = 0; i < batch * seq_len; i++) {
        h_input[i] = (float)(rand() % 100) / 10.0f - 5.0f;
    }
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    // 运行naive
    launch_softmax_naive(d_input, d_output_naive, batch, seq_len, 0);
    cudaDeviceSynchronize();

    // 运行optimized
    launch_softmax_optimized(d_input, d_output_opt, batch, seq_len, 0);
    cudaDeviceSynchronize();

    // 拷贝结果
    cudaMemcpy(h_output_naive, d_output_naive, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output_opt, d_output_opt, size, cudaMemcpyDeviceToHost);

    // 验证
    int error_count = 0;
    float max_error = 0.0f;
    float tolerance = 1e-5f;

    for (int i = 0; i < batch * seq_len; i++) {
        float diff = fabsf(h_output_naive[i] - h_output_opt[i]);
        float rel_error = diff / (fabsf(h_output_naive[i]) + 1e-10f);

        if (rel_error > tolerance) {
            if (error_count < 5) {
                printf("  [%d] Naive=%.6e, Opt=%.6e, RelErr=%.6e\n",
                       i, h_output_naive[i], h_output_opt[i], rel_error);
            }
            error_count++;
            max_error = fmaxf(max_error, rel_error);
        }
    }

    if (error_count > 0) {
        printf("  ❌ 验证失败: %d/%d 错误, 最大相对误差: %.6e\n",
               error_count, batch * seq_len, max_error);
        return 1;
    }

    printf("  ✅ 验证通过: 所有元素误差 < %.1e\n", tolerance);

    free(h_input); free(h_output_naive); free(h_output_opt);
    cudaFree(d_input); cudaFree(d_output_naive); cudaFree(d_output_opt);
    return 0;
}
EOF

echo "编译验证程序..."
nvcc -arch=sm_120 -O3 softmax/softmax_naive.cu softmax/softmax_final.cu /tmp/verify_softmax.cu -o /tmp/verify_softmax 2>&1 | grep -i error || /tmp/verify_softmax

echo ""
echo "==============================================="
echo "  验证完成"
echo "==============================================="
