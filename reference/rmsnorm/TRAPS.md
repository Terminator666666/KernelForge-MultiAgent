# RMSNorm 优化陷阱记录

本文件记录 RMSNorm 优化过程中遇到的陷阱、无声错误和教训。

---

## 🚨 已知陷阱

### 陷阱 1: 数值稳定性问题
**类型**: 正确性错误

**症状**:
- 在某些 hidden_size 下输出 NaN
- 大的 batch_size 时数值溢出

**根本原因**:
- FP16 精度不足导致平方和溢出
- epsilon 值太小

**解决方案**:
```cuda
// ❌ 错误：直接平方求和可能溢出
float sum = 0.0f;
for (int i = 0; i < hidden_size; ++i) {
    sum += x[i] * x[i];  // 可能溢出
}

// ✅ 正确：使用更稳定的计算方式
float sum = 0.0f;
float max_val = 0.0f;
// 先找最大值进行归一化
for (int i = 0; i < hidden_size; ++i) {
    max_val = fmaxf(max_val, fabsf(x[i]));
}
// 归一化后计算
for (int i = 0; i < hidden_size; ++i) {
    float norm_x = x[i] / max_val;
    sum += norm_x * norm_x;
}
```

**预防措施**:
- 总是使用 `rsqrt` 而非 `1.0 / sqrt`
- epsilon 不要小于 1e-6
- FP16 模式下考虑使用 FP32 accumulator

---

### 陷阱 2: Warp Divergence
**类型**: 性能问题

**症状**:
- 理论上 memory-bound 但实际 compute 占用高
- 小 batch_size 性能差

**根本原因**:
- 不同 warp 处理不同长度的序列导致分支
- 边界处理逻辑不一致

**解决方案**:
```cuda
// ❌ 错误：条件分支导致 divergence
if (tid < hidden_size) {
    // 处理逻辑
}

// ✅ 正确：使用 warp-uniform 控制流
int items_per_thread = (hidden_size + blockDim.x - 1) / blockDim.x;
for (int i = 0; i < items_per_thread; ++i) {
    int idx = tid + i * blockDim.x;
    if (idx < hidden_size) {
        // 所有 warp 执行相同次数的循环
    }
}
```

---

### 陷阱 3: Shared Memory Bank Conflict
**类型**: 性能问题

**症状**:
- 理论带宽利用率低于预期
- NCU 显示高 shared memory conflict

**根本原因**:
- Reduction 过程中多个线程访问相同 bank

**解决方案**:
```cuda
// ❌ 错误：所有线程写入连续地址
__shared__ float sdata[256];
sdata[tid] = value;  // Bank conflict

// ✅ 正确：使用 padding 避免 conflict
__shared__ float sdata[256 + 32];  // 32 = warpSize
sdata[tid] = value;
```

---

### 陷阱 4: 忽略 Fused Add RMSNorm
**类型**: 优化机会错失

**症状**:
- 性能达到瓶颈，难以进一步提升

**根本原因**:
- 未考虑 residual connection 融合
- 多次 memory access

**解决方案**:
- 实现 Fused Add RMSNorm 变体
- 一次 kernel 完成 `hidden + residual` 和 normalization
- 减少 memory 传输

---

## 📊 性能陷阱

### 陷阱 5: 过度优化小 batch
**类型**: 优化方向错误

**症状**:
- 在 batch=1 上性能很好
- 在 batch=256 上性能反而下降

**根本原因**:
- 针对小 batch 的优化（如单 warp）不适合大 batch
- 未考虑并行度

**教训**:
- 先优化最常见的 batch_size（32-256）
- 小 batch 可以用特化路径

---

### 陷阱 6: 盲目使用 TMA
**类型**: 硬件特性误用

**症状**:
- 启用 TMA 后性能反而下降
- NCU 显示 TMA stall 增加

**根本原因**:
- RMSNorm 计算量小，TMA overhead 明显
- 数据重用少，TMA 优势不明显

**教训**:
- TMA 适合大 tile、高重用的场景
- RMSNorm 可能更适合传统 load/store

---

## 🔍 调试陷阱

### 陷阱 7: 误判基线性能
**类型**: 基准测试错误

**症状**:
- 优化后加速比异常高（如 10x+）
- 难以复现

**根本原因**:
- 基线未 warmup
- 基线未使用 CUDA events 计时
- 基线使用了错误的配置

**预防措施**:
```python
# ✅ 正确的基准测试
# 1. Warmup
for _ in range(10):
    baseline_kernel()
torch.cuda.synchronize()

# 2. CUDA events 计时
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
for _ in range(100):
    baseline_kernel()
end.record()
torch.cuda.synchronize()
baseline_time = start.elapsed_time(end) / 100
```

---

## 📝 文档陷阱

### 陷阱 8: 变体未记录父子关系
**类型**: 记忆丢失

**症状**:
- 不知道某个优化是基于哪个版本
- 难以回滚到之前的好版本

**解决方案**:
- 严格维护 `solutions.jsonl` 的 parent 字段
- 每个变体记录父 ID 和修改描述

---

## 🎯 优化建议

### ✅ 推荐策略
1. **先正确，再快速** - Phase 1 专注正确性
2. **基于证据优化** - 每个优化有 NCU 支持
3. **记录所有尝试** - 包括失败的

### ❌ 避免策略
1. **不要盲目复制代码** - 理解再实现
2. **不要跳过验证** - 每次修改后验证
3. **不要过早优化** - 先建立基线

---

## 📚 相关资源

- KernelWiki RMSNorm 页面
- FlashInfer RMSNorm 参考实现
- NCU 分析示例: `profile/rmsnorm-*/`

---

**最后更新**: 2026-06-15  
**贡献者**: KernelForge-MultiAgent  
**版本**: v1.0

---

## 如何使用本文件

### 在优化前
1. 阅读所有已知陷阱
2. 在 draft.md 中标注风险点
3. 设计预防措施

### 遇到问题时
1. 检查症状是否匹配已知陷阱
2. 应用对应的解决方案
3. 如果是新陷阱，记录到本文件

### 优化完成后
1. 回顾是否触发了陷阱
2. 补充新发现的陷阱
3. 更新预防措施
