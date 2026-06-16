#!/bin/bash

# ============================================================================
# 真实 Naive Baseline 性能测试脚本
# 目的: 重新计算基于真正 naive 实现的加速比
# ============================================================================

echo "=================================="
echo "编译真实 Naive Baseline 测试程序"
echo "=================================="

# 检测 GPU 架构
GPU_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
echo "检测到 GPU 架构: sm_${GPU_ARCH}"

# 编译测试程序
nvcc -O3 --use_fast_math -arch=sm_${GPU_ARCH} \
    benchmark_true_naive.cu \
    -o benchmark_true_naive \
    2>&1 | tee compile_true_naive.log

if [ $? -ne 0 ]; then
    echo "❌ 编译失败，请检查 compile_true_naive.log"
    exit 1
fi

echo "✅ 编译成功"
echo ""

echo "=================================="
echo "运行性能测试"
echo "=================================="

./benchmark_true_naive | tee true_naive_results.txt

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 测试完成，结果已保存到 true_naive_results.txt"
else
    echo ""
    echo "❌ 测试失败"
    exit 1
fi
