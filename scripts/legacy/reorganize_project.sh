#!/bin/bash

# 创建算子文件夹
mkdir -p matmul softmax rmsnorm layernorm

echo "重组项目结构..."

# MatMul
echo "处理MatMul..."
# 暂时跳过，因为没有明确的迭代版本文件

# Softmax
echo "处理Softmax..."
if [ -f "softmax-1M_kernel.cu" ]; then
    cp softmax-1M_kernel.cu softmax/softmax_baseline.cu
fi
if [ -f "softmax-1M_opt.cu" ]; then
    cp softmax-1M_opt.cu softmax/softmax_final.cu
fi

# RMSNorm  
echo "处理RMSNorm..."
if [ -f "rmsnorm-4096_kernel.cu" ]; then
    cp rmsnorm-4096_kernel.cu rmsnorm/rmsnorm_baseline.cu
fi
if [ -f "rmsnorm-4096_opt.cu" ]; then
    cp rmsnorm-4096_opt.cu rmsnorm/rmsnorm_final.cu
fi

# LayerNorm
echo "处理LayerNorm..."
if [ -f "layernorm-4096_kernel.cu" ]; then
    cp layernorm-4096_kernel.cu layernorm/layernorm_baseline.cu
fi
if [ -f "layernorm-4096_opt.cu" ]; then
    cp layernorm-4096_opt.cu layernorm/layernorm_final.cu
fi

# MatMul (现在处理)
echo "处理MatMul..."
if [ -f "matmul-2048x2048x2048_opt.cu" ]; then
    cp matmul-2048x2048x2048_opt.cu matmul/matmul_final.cu
fi

echo "项目重组完成！"
ls -la matmul/ softmax/ rmsnorm/ layernorm/ 2>/dev/null | head -50
