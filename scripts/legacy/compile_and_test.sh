#!/bin/bash

echo "=== Flash Attention 算子优化测试 ==="
echo ""

# 检查CUDA环境
if ! command -v nvcc &> /dev/null; then
    echo "错误: 未找到nvcc编译器"
    exit 1
fi

echo "CUDA编译器版本:"
nvcc --version
echo ""

# 检查GPU
echo "检测到的GPU:"
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
echo ""

# 编译原始kernel
echo "=== 编译原始kernel ==="
nvcc -o flash_attention_original \
    flash-attention-seq2048_kernel.cu \
    -arch=sm_89 \
    -O3 \
    -use_fast_math \
    --expt-relaxed-constexpr \
    2>&1 | tee compile_original.log

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "✓ 原始kernel编译成功"
else
    echo "✗ 原始kernel编译失败，查看 compile_original.log"
fi
echo ""

# 编译优化kernel
echo "=== 编译优化kernel ==="
nvcc -o flash_attention_optimized \
    flash-attention-seq2048_opt.cu \
    test_flash_attention.cu \
    -arch=sm_89 \
    -O3 \
    -use_fast_math \
    --expt-relaxed-constexpr \
    -lcudart \
    2>&1 | tee compile_optimized.log

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "✓ 优化kernel编译成功"
else
    echo "✗ 优化kernel编译失败，查看 compile_optimized.log"
    exit 1
fi
echo ""

# 运行测试
echo "=== 运行性能测试 ==="
./flash_attention_optimized

echo ""
echo "=== 测试完成 ==="
