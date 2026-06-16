# FlashInfer-Bench 验证指南

## 概述

**所有生成的算子代码必须通过 FlashInfer-Bench 验证才能被认为是成功的。**

FlashInfer-Bench 是 FlashInfer 项目的官方基准测试套件，用于验证：
- 相对于参考实现的正确性
- 在真实工作负载上的性能表现
- 符合生产级内核标准

## 验证环境

### FlashInfer-Bench 位置
```
D:/Agent/flashinfer-bench-main/flashinfer-bench-main
```

### FlashInfer-Trace 数据集
FlashInfer-Trace 是官方数据集，包含来自真实 AI 系统部署环境的内核和工作负载：
- HuggingFace 地址: https://huggingface.co/datasets/flashinfer-ai/flashinfer-trace
- 数据集包含：定义、参考测试、基线解决方案、工作负载、评估 traces

## 验证步骤

### 1. 安装 FlashInfer-Bench

```bash
# 在 Linux 环境中执行
pip install flashinfer-bench
```

### 2. 克隆 FlashInfer-Trace 数据集

```bash
# 只克隆 Git LFS 指针文件（大型张量文件按需下载）
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/flashinfer-ai/flashinfer-trace

# 或者使用已有的数据集目录
cd D:/Agent/flashinfer-bench-main/flashinfer-bench-main
```

### 3. 准备待验证的内核

将生成的 CUDA 内核代码按照 FlashInfer-Bench 的解决方案（Solution）格式准备：

```python
# 示例：将优化后的 kernel 封装为 Solution
from flashinfer_bench.data import Solution

solution = Solution(
    name="my_optimized_kernel",
    definition_name="target_operation",
    op_type="attention",  # 或 gemm, normalization, moe 等
    implementation_path="path/to/optimized_kernel.cu",
    language="cuda"
)
```

### 4. 运行验证

```bash
cd D:/Agent/flashinfer-bench-main/flashinfer-bench-main

# 对本地数据集运行基准测试
flashinfer-bench run --local flashinfer-trace

# 或针对特定算子类型
flashinfer-bench run --local flashinfer-trace --op-type attention

# 或针对特定定义
flashinfer-bench run --local flashinfer-trace --definition my_kernel_definition
```

### 5. 检查验证结果

FlashInfer-Bench 会生成验证报告，包括：

- **正确性检查**: 与参考实现的输出比较
- **性能指标**: 延迟、吞吐量、GPU 利用率
- **工作负载覆盖**: 在多种输入配置下的表现
- **数据类型支持**: FP16、BF16、FP8 等

成功的验证要求：
- ✅ 所有正确性测试通过
- ✅ 性能不低于基线解决方案
- ✅ 在所有相关工作负载上稳定运行

## 支持的算子类型

FlashInfer-Trace 数据集包含以下算子类型：

| 算子类型 | 描述 | 示例 |
|---------|------|------|
| `attention` | 注意力机制 | GQA, MLA, FlashAttention |
| `gemm` | 矩阵乘法 | Dense GEMM, Block-scaled GEMM |
| `normalization` | 归一化 | RMSNorm, LayerNorm |
| `moe` | 专家混合 | Fused MoE, Grouped MoE |
| `rope` | 旋转位置编码 | RoPE |
| `sampling` | 采样 | Top-k, Top-p sampling |
| `gated-delta-net` | GatedDeltaNet | GDN 融合投影 |

每个算子类型在 `flashinfer-trace/definitions/{op_type}/` 下有多个定义。

## 工作负载配置

FlashInfer-Trace 数据集包含多种真实场景的工作负载配置：

- **批大小**: 1, 4, 8, 16, 32, 64, 128, 256
- **序列长度**: 128, 256, 512, 1024, 2048, 4096, 8192, 16384
- **张量形状**: 根据模型架构变化
- **数据类型**: FP16, BF16, FP8, FP4
- **并行配置**: TP (Tensor Parallelism), EP (Expert Parallelism)

## 验证标准

### 正确性要求

1. **数值精度**: 输出与参考实现的误差在容差范围内
   - FP32: `atol=1e-5, rtol=1e-3`
   - FP16/BF16: `atol=1e-3, rtol=1e-2`
   - FP8: `atol=1e-2, rtol=5e-2`

2. **边界情况**: 在极端输入下不崩溃
   - 空批次（batch_size=0）
   - 最小序列长度
   - 最大序列长度

3. **内存安全**: 无越界访问、无内存泄漏

### 性能要求

1. **基线对比**: 性能不低于基线解决方案（通常是 FlashInfer 的原生实现）

2. **稳定性**: 多次运行的性能方差 < 5%

3. **可扩展性**: 在不同工作负载下保持合理性能

### 完整性要求

1. **工作负载覆盖**: 通过数据集中该算子的所有工作负载测试

2. **数据类型覆盖**: 支持定义中声明的所有数据类型

3. **约束满足**: 满足定义中的所有约束条件

## 内部开发工具与 FlashInfer-Bench 的关系

### 内部工具（仅用于开发）

```bash
# 快速冒烟测试 - 检查编译和基本运行
python verify.py --cuda --arch sm_120

# 内部基准测试 - 快速性能对比
nvcc scripts/workflow/benchmarks/cuda/benchmark_true_naive.cu -O3 -lineinfo -arch=sm_120 -o scripts/workflow/build/benchmark_true_naive
scripts/workflow/build/benchmark_true_naive
```

这些内部测试：
- ✅ 帮助快速发现明显错误
- ✅ 在开发迭代中提供快速反馈
- ❌ **不构成验收标准**
- ❌ 不覆盖真实工作负载
- ❌ 不验证与 FlashInfer 生态的兼容性

### FlashInfer-Bench（最终验收）

```bash
# 最终验收测试
flashinfer-bench run --local flashinfer-trace
```

FlashInfer-Bench 验证：
- ✅ 在真实工作负载上的正确性
- ✅ 在生产环境配置下的性能
- ✅ 与 FlashInfer API 的兼容性
- ✅ 符合社区标准

## 故障排除

### 验证失败：正确性错误

```
FAILED: Correctness check failed for workload X
Expected: [...]
Got: [...]
Max error: 0.05 (threshold: 0.001)
```

**解决方案**：
1. 检查数值计算精度
2. 验证 shared memory 同步
3. 检查边界条件处理
4. 对比参考实现逻辑

### 验证失败：性能不达标

```
FAILED: Performance below baseline
Baseline: 1.23 ms
Current: 2.45 ms
```

**解决方案**：
1. 使用 Nsight Compute 分析瓶颈（见 `skills/ncu-report-skill`）
2. 检查 GPU 利用率
3. 优化 memory access pattern
4. 参考 `skills/KernelWiki` 中的优化技术

### 验证失败：工作负载不兼容

```
FAILED: Kernel launch failed for workload X
Shape: (batch=128, seq=8192, heads=32)
```

**解决方案**：
1. 检查 kernel 的资源限制（shared memory, registers）
2. 添加动态配置支持
3. 实现 shape-specific 优化路径

## 集成到工作流

### 在优化循环中使用

```python
# 伪代码：集成 FlashInfer-Bench 验证
for iteration in range(max_iterations):
    # 1. 生成优化候选
    generate_optimized_kernel()
    
    # 2. 内部快速测试（可选）
    if not run_smoke_test():
        continue
    
    # 3. FlashInfer-Bench 验证（必需）
    result = run_flashinfer_bench_validation()
    
    if result.correctness_passed and result.performance_acceptable:
        accept_candidate()
        break
    else:
        analyze_failure_and_iterate()
```

### 在 CI/CD 中使用

```yaml
# GitHub Actions 示例
- name: Validate with FlashInfer-Bench
  run: |
    pip install flashinfer-bench
    flashinfer-bench run --local flashinfer-trace --op-type ${{ matrix.op_type }}
```

## 参考资源

- [FlashInfer 文档](https://docs.flashinfer.ai)
- [FlashInfer-Bench GitHub](https://github.com/flashinfer-ai/flashinfer-bench)
- [FlashInfer-Trace 数据集](https://huggingface.co/datasets/flashinfer-ai/flashinfer-trace)
- [FlashInfer-Trace 定义模式](https://bench.flashinfer.ai/docs/flashinfer-trace/definition)
- [算子类型参考](https://bench.flashinfer.ai/docs/op-types/)

## 总结

记住关键原则：

1. **内部测试 = 开发工具**：快速迭代，但不是验收标准
2. **FlashInfer-Bench = 验收标准**：必须通过才能认为优化成功
3. **数据集驱动**：验证基于真实工作负载，不是合成数据
4. **社区标准**：确保与 FlashInfer 生态兼容

**只有通过 FlashInfer-Bench 验证的内核才能被认为是成功的优化成果。**
