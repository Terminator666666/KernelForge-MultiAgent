# Repository Notes

- Keep final CUDA operator source under `kernels/operators/`.
- Use `prompts/` for phase-based optimization guidance.
- Use `verify.py` for release-layout checks and smoke testing.
- Treat `skills/KernelWiki` and `skills/ncu-report-skill` as required local
  skills for Blackwell/B200 optimization and Nsight Compute analysis.
- **执行环境规则**：禁止在 Windows 主环境运行代码；但如果当前在 WSL / Linux 环境，
  则允许执行构建、验证、benchmark、Nsight Compute profile，并以真实运行结果作为闭环证据。
- **Supported operators**: 10 types aligned with FlashInfer-Bench - `gemm`, 
  `gqa_paged`, `gqa_ragged`, `mla_paged`, `dsa_paged`, `moe`, `rmsnorm`, `rope`, 
  `sampling`, `gdn`. See `docs/SUPPORTED_OPERATORS.md` for details.
- **Acceptance criterion**: All generated operator code must pass FlashInfer-Bench
  validation (`D:/Agent/flashinfer-bench-main`) to be considered successful.
  Internal benchmarks are for development only.
- Reject any candidate that relies on a broken baseline or a false speedup.
- **加速比口径（强制）**：只认「相对官方 baseline」的加速比 `sol/base`
  （官方 baseline = 数据集 `solutions/baseline/...` 里的 flashinfer 实现）。
  「vs 参考实现 PyTorch」的加速比（`sol/ref`）只能用于确认正确性，**不得**作为成绩，
  因为打过朴素 PyTorch 对带宽瓶颈算子毫无含金量。
- **闭环强制要求**：每一轮优化必须有本机真实 NCU 数据驱动，禁止凭空猜测优化方向。
- **NCU 版本强制要求**：RTX 5070 / sm_120 的 Nsight Compute 采样只能使用
  `/usr/local/NVIDIA-Nsight-Compute-2025.2/ncu`。禁止使用 `/usr/local/cuda/bin/ncu`，
  因为后者在当前 WSL 环境会报 `LibraryNotLoaded`，不能作为有效证据。
- **决策依据强制要求**：每一轮的 ACCEPT / REJECT 都必须同时引用：
  1. 本轮真实 NCU 报告（solution + 官方 baseline 各一份）；
  2. `skills/KernelWiki` 中与本轮瓶颈对应的页面。
- **baseline NCU 复用规则**：官方 baseline 的 NCU 报告可以复用，但仅限
  `definition + batch_size + device + baseline_solution + NCU版本` 完全一致的情况；
  一旦其中任何一项变化，就必须重新采样。
- **KernelWiki 使用规则**：RTX 5070 / sm_120 属于 Blackwell 架构，因此每轮都必须先查
  `skills/KernelWiki` 再定方向；但要明确区分 **可迁移模式**（如 low-sm-utilization、
  memory-bound、vectorized-loads、register-budgeting）与 **仅限 SM100/B200 的特性**
  （如 CLC、部分 TMEM/tcgen05 持久化路径），禁止把只适用于数据中心 Blackwell 的结论
  直接照搬到 sm_120。

---

## 当前进度（2026-06-16）

### 环境（已打通，固定不要再动）
- **torch 锁定 `2.7.1+cu128`**，匹配本机 CUDA 12.8 驱动（RTX 5070 Laptop, sm_120,
  capability (12,0)）。**禁止升级 torch**，flashinfer-bench 虽声明 torch>=2.8 但实测可用。
- 已安装 `flashinfer-bench`(0.0.0.dev0) + `apache-tvm-ffi`(0.1.12)。
- 官方数据集已克隆到 `/mnt/d/Agent/flashinfer-trace`（国内镜像，HF 直连会超时）。
  含 190 个 definition；rmsnorm 家族有 `rmsnorm_h4096` 等 15 个，并带 5 家作者对照 solution。

### 验证工具链（处理 WSL + torch 2.7.1 的三个坑）
- `scripts/workflow/fib_run.py`：flashinfer-bench CLI 包装器。
  - 补占位 dtype `torch.float4_e2m1fn_x2`（torch 2.7.1 缺，bfloat16 算子用不到）。
  - 补 `LIBRARY_PATH`/`LD_LIBRARY_PATH` 指向 `/usr/lib/wsl/lib`，解决链接期 `-lcuda` 找不到。
- `scripts/workflow/fib_inproc_validate.py`：**单进程**正确性+性能验证器。
  - WSL 下官方多进程 persistent runner 报 `CUDA error: invalid resource handle`（CUDA IPC
    不兼容 WSL），故绕开多进程，直接在本进程内 build reference + build solution 并逐 workload 比对。
  - **后续 9 个算子统一用此脚本验证。**

### 已完成算子
- **rmsnorm（样板，接口已通，性能未达标 → 进入闭环优化中）**
  - kernel: `kernels/operators/rmsnorm/rmsnorm_h4096_tvmffi.cu`
    （bfloat16，bfloat162 向量化，warp + shared memory 两级归约，float32 累加）
  - solution: `flashinfer-trace/solutions/kernelforge/rmsnorm/rmsnorm_h4096/kernelforge_rmsnorm_h4096_cuda_v1.json`
  - 绑定 `tvm-ffi`，入口 `kernel.cu::rmsnorm_h4096`，DPS（输出在末位参数）。
  - **正确性：14/14 workload 通过。**
  - **性能（真实口径 sol/base = 0.893x，比官方 flashinfer 慢约 11%）→ 判定 REJECT。**
    - 小 batch（1~170）只有官方 0.64~0.88；大 batch（>8000）持平 ~1.0。
  - NCU 实采（本机 5070，`profile/rmsnorm/`）：batch=16 时我的内存吞吐仅 5.70%，
    官方 35.99%；grid=16 仅占满 1/3 SM。**根因：小 batch 并行度不足 + 访存事务不紧凑。**
  - 当前正按 NCU 结论做 Round 1（v2）优化。

## 闭环优化流程（真实闭环，强制执行）

不是「写一版就报数」，而是 NCU 数据驱动的多轮迭代。每个算子按下述 10 步循环推进，
全部基于本机真实数据，加速比一律用 `sol/base`。

```
1. derive    建 rounds/round-<N>/<family>/{src,profile,docs}，从锚点变体派生
2. brief     写 BRIEF.md：本轮目标(sol/base 提升)、方向(来自上轮NCU)、要避开的 TRAPS，
             并列出本轮必须参考的 KernelWiki 页面
3. optimize  按 NCU 结论改 kernel（禁止凭空猜方向），并记录哪些 KernelWiki 页面支持该方向
4. benchmark fib_inproc_validate.py 计时（本机 5070，CUDA events）
5. validate  同脚本逐 workload 正确性比对（atol/rtol=1e-2），不过则 REJECT
6. compare   以官方 baseline 为锚点算 sol/base（唯一成绩口径）
7. decide    只有在“本轮真实 NCU 证据 + KernelWiki 依据”都完整时才允许做决策；
             正确性失败 → REJECT；sol/base ≥ 1.05 → ACCEPT；否则 REJECT/继续
8. document  ACCEPT 才更新 reference/<family>/（README + solutions.jsonl + variants/）
9. lessons   失败教训写 reference/<family>/TRAPS.md（下一轮 brief 自动注入）
10. plan     用本轮 NCU + KernelWiki 定位新瓶颈，规划下一轮方向；回到 1
```

终止条件：sol/base ≥ 1.05（ACCEPT 收敛）/ 连续 3 轮无改进 / 轮次上限。

### 闭环固定命令
```bash
# 验证 + 对官方 baseline 比（成绩口径）
python scripts/workflow/fib_inproc_validate.py \
  --dataset /mnt/d/Agent/flashinfer-trace \
  --definition rmsnorm_h4096 \
  --solution  <我的solution名> \
  --baseline  flashinfer_wrapper_2e27cd

# NCU 实采（本机），分别采我的 kernel 与官方 baseline
/usr/local/NVIDIA-Nsight-Compute-2025.2/ncu -f -o profile/<family>/sol_bs<N> \
  --set full <python绝对路径> scripts/workflow/ncu_driver.py ... --which sol --batch-size <N>
```

### 本轮决策前必须补齐的证据文件

每一轮工作区都必须补齐下面两个文件，否则 `./scripts/evaluate-round.sh` 会直接拒绝进入决策：

1. `rounds/round-<N>/<family>/profile/ncu_evidence.json`
   - 必须填写本轮 **solution** 与 **官方 baseline** 的真实 NCU 报告路径
   - 必须写明本轮瓶颈、决策驱动因素、为何产生当前优化方向
2. `rounds/round-<N>/<family>/docs/kernelwiki_evidence.json`
   - 必须列出本轮实际参考的 `skills/KernelWiki` 页面
   - 必须写明这些页面如何支撑本轮决策
   - 必须标注这些结论是否适用于 `sm_120`

禁止出现“只有 benchmark，没有 NCU”；也禁止出现“有 NCU，但没有 KernelWiki 依据”的轮次。

### 待办（其余 9 个算子，仍是占位空壳）
- 顺序建议：`gemm`(收益最高，数据集有现成 definition+官方CUDA对照) → `rope` → `sampling`
  → `gqa_paged` → `moe` → `gqa_ragged` → `mla_paged` → `dsa_paged` → `gdn`。
- 复用模板：每个算子 = 1 个 TVM-FFI kernel(`.cu`) + 1 个 solution.json + 用
  `fib_inproc_validate.py` 验证。
- 注意 dtype：rmsnorm 是 bfloat16；gemm 等量化算子若用到 float4/float8，需确认 torch 2.7.1
  是否支持，否则需另想办法（不可升级 torch）。
