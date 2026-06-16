#!/bin/bash
# 真实性能测试 - 主脚本

set -e

echo "========================================================================"
echo "  KernelForge 真实性能测试"
echo "========================================================================"
echo ""

# 激活环境
echo "步骤 1: 激活 conda 环境..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lite
echo "✓ 环境激活成功"
echo ""

# 检查环境
echo "步骤 2: 检查环境..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
echo ""

# 创建输出目录
mkdir -p benchmark_results
echo "✓ 创建输出目录: benchmark_results/"
echo ""

# 运行 baseline
echo "步骤 3: 运行 Baseline (Self-Attention)..."
python baseline/self_attention_baseline.py \
    --batch 2 \
    --heads 16 \
    --seq-len 1024 \
    --head-dim 64 \
    --dtype float16 \
    --warmup 20 \
    --iterations 100 \
    --save benchmark_results/baseline_self_attention.json

echo ""
echo "========================================================================"
echo "  Baseline 测试完成！"
echo "========================================================================"
echo ""
echo "查看结果:"
echo "  cat benchmark_results/baseline_self_attention.json"
echo ""
echo "下一步:"
echo "  1. 检查生成的 CUDA 代码是否可编译"
echo "  2. 创建 PyTorch extension wrapper"
echo "  3. 编译优化的 kernel"
echo "  4. 运行对比测试"
echo ""
