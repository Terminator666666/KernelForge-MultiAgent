#!/bin/bash

echo "========================================"
echo "  MatMul Kernel 版本性能对比"
echo "========================================"
echo ""

echo "编译所有版本..."
echo ""

# 编译V1
echo "[1/2] 编译 V1..."
nvcc -O3 -arch=sm_89 -use_fast_math -std=c++17 -lineinfo \
    matmul-2048x2048x2048_opt.cu matmul_test.cu -o matmul_test_v1 2>&1 | grep -v "warning"

if [ $? -eq 0 ]; then
    echo "✓ V1 编译成功"
else
    echo "✗ V1 编译失败"
fi

# 编译V2 (Final)
echo "[2/2] 编译 V2 (Final)..."
nvcc -O3 -arch=sm_89 -use_fast_math -std=c++17 -lineinfo \
    matmul-2048x2048x2048_opt_v2.cu matmul_test.cu -o matmul_test_v2 2>&1 | grep -v "warning"

if [ $? -eq 0 ]; then
    echo "✓ V2 编译成功"
else
    echo "✗ V2 编译失败"
fi

echo ""
echo "========================================"
echo "  性能测试"
echo "========================================"
echo ""

# 测试V1
if [ -f "./matmul_test_v1" ]; then
    echo "--- V1 性能 ---"
    ./matmul_test_v1 2>&1 | grep -E "平均时间|性能|效率"
    echo ""
fi

# 测试V2
if [ -f "./matmul_test_v2" ]; then
    echo "--- V2 性能 ---"
    ./matmul_test_v2 2>&1 | grep -E "平均时间|性能|效率"
    echo ""
fi

echo "========================================"
echo "  对比总结"
echo "========================================"
echo ""
echo "详细报告: optimization_report.md"
