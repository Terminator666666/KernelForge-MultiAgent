#!/bin/bash

# MatMul Kernel 优化测试脚本

echo "==================================="
echo "  MatMul Kernel 优化与性能测试"
echo "==================================="
echo ""

# 检测CUDA
if ! command -v nvcc &> /dev/null; then
    echo "错误: 找不到 nvcc，请确保安装了 CUDA Toolkit"
    exit 1
fi

CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
echo "CUDA 版本: $CUDA_VERSION"
echo ""

# 检测GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "错误: 找不到 nvidia-smi"
    exit 1
fi

echo "检测到的GPU:"
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader
echo ""

# 编译参数
ARCH="sm_89"  # RTX 5070 (Ada Lovelace)
NVCC_FLAGS="-O3 -arch=$ARCH -use_fast_math -std=c++17"
NVCC_FLAGS="$NVCC_FLAGS -Xcompiler -fopenmp"
NVCC_FLAGS="$NVCC_FLAGS -lineinfo"

echo "编译优化kernel..."
echo "编译参数: $NVCC_FLAGS"
echo ""

# 编译
nvcc $NVCC_FLAGS \
    matmul-2048x2048x2048_opt.cu \
    matmul_test.cu \
    -o matmul_test

if [ $? -ne 0 ]; then
    echo "编译失败!"
    exit 1
fi

echo "编译成功!"
echo ""

# 运行测试
echo "==================================="
echo "  运行性能测试"
echo "==================================="
echo ""

./matmul_test

if [ $? -ne 0 ]; then
    echo ""
    echo "测试运行失败!"
    exit 1
fi

echo ""
echo "==================================="
echo "  测试完成"
echo "==================================="
