# KernelForge-MultiAgent - Closed-Loop Automatic Optimization

KernelForge-MultiAgent 是一个**闭环自动优化**的 CUDA 内核优化系统，采用 Master-Sub 双智能体架构。

---

## 🎯 核心特性

### 闭环自动优化
- ✅ **Master Agent 编排**: 自动规划和执行优化轮次
- ✅ **10-Step Round Loop**: 标准化的优化循环
- ✅ **跨轮次记忆**: Reference 家族归档持久化
- ✅ **陷阱累积**: TRAPS.md 系统化避免失败
- ✅ **证据驱动决策**: 客观的 ACCEPT/REJECT 判断

### 双智能体架构
```
Master Agent (编排者)
  ↓ BRIEF.md (目标、约束、陷阱)
Sub Agent (执行者 + Humanize RLCR)
  ↓ 优化后的内核 + 性能数据
自动化评估
  ↓ benchmark + validate + compare
Master Agent 决策
  ↓ ACCEPT → 归档 | REJECT → 下一轮
```

### 工具集成
- **Humanize RLCR Loop**: Plan-Execute-Verify 执行引擎
- **KernelWiki**: 2179 PRs, 48 wiki 优化知识库
- **ncu-report-skill**: Nsight Compute 性能分析
- **FlashInfer-Bench**: 官方 benchmark / baseline / workload 验收来源

### 当前主线优化范围（强制）
- 本仓库当前**只主动优化 3 类高收益算子**：
  1. `dsa_sparse_attention`
  2. `gdn_prefill`
  3. `dsa_topk_indexer`
- 选择依据：
  - 对照 `mlsys2026-flashinfer-contest-main` 与 `AKO4X-main` 的真实结果，这几类算子更容易做出
    明显加速比；
  - `rmsnorm` / `gemm` 等算子在官方库中往往已经高度优化，继续深挖的平均收益更低。
- 说明：
  - FlashInfer-Bench 上游仍覆盖更多算子类型；
  - 但 **KernelForge-MultiAgent 当前闭环目标只限上述 3 类**；
  - 其他算子默认仅保留样板、验证或参考价值，除非用户单独要求重开。

### 当前主线实现方式（强制）
- 所有主线算子都必须在**独立工作区**里推进。
- 起点不是“直接改 FlashInfer 官方源码仓库里的内核”，而是：
  1. 读取官方 task / baseline 的语义与接口；
  2. 明确 shape contract、correctness contract、baseline 行为；
  3. 在自己的工作区里**自写新的 kernel variant**（Triton / CUDA / CuTe DSL）。
- 允许参考官方 baseline 的接口、行为和 profile 证据；
  但默认**不把“直接 patch 官方内核源码”作为主线优化方式**。

### 当前主线成绩口径（强制）
- 最终只认**相对 FlashInfer 官方 baseline / expert baseline 的提升**。
- 轮内允许做：
  - candidate vs parent
  - candidate vs 历史 anchor
  - candidate vs 参考实现
- 但这些都只是中间证据，**不能替代最终成绩口径**。
- 最终归档、README 结论、ACCEPT/REJECT，一律以官方 baseline / expert baseline 为分母。

---

## 🚀 快速开始

### 前置准备

1. **安装 Humanize 插件**（首次运行）
   ```bash
   /plugin marketplace add PolyArch/humanize
   /plugin install humanize@PolyArch
   ```

2. **设置数据集路径**
   ```bash
   export FIB_DATASET_PATH="D:/Agent/flashinfer-bench-main/flashinfer-bench-main/data/flashinfer-trace"
   ```

### 启动 Campaign

```bash
cd D:/Agent/KernelForge-MultiAgent

# 启动 10 轮优化 Campaign
./scripts/start-campaign.sh rmsnorm 10

# 这将:
# ✓ 初始化 reference/rmsnorm/ 归档
# ✓ 创建 README.md (优化历史)
# ✓ 创建 TRAPS.md (陷阱记录)
# ✓ 准备 Campaign 配置
```

### 运行轮次

```bash
# Round 0: 建立基线
./scripts/run-round.sh rmsnorm 0

# 这将:
# 1. derive: 创建 rounds/round-0/rmsnorm/
# 2. brief: 生成 BRIEF.md
# 3-10: 提示你执行后续步骤
```

### Sub Agent 优化流程

```bash
cd rounds/round-0/rmsnorm

# 1. 阅读 BRIEF.md
cat BRIEF.md

# 2. 查看陷阱
cat ../../reference/rmsnorm/TRAPS.md

# 3. 编写实现计划
vim docs/draft.md

# 4. 在 Claude 中生成详细计划
/humanize:gen-plan

# 5. 在 Claude 中启动 RLCR 循环
/humanize:start-rlcr-loop
# → Research: 使用 KernelWiki 查找优化技术
# → Learn: 理解技术细节
# → Code: 编写/修改代码
# → Review: 验证正确性

# 6. 评估结果
cd ../../..
./scripts/evaluate-round.sh rmsnorm 0
```

### 持续优化

```bash
# Round 1, 2, 3...
./scripts/run-round.sh rmsnorm 1
# ... 重复 Sub Agent 流程
./scripts/evaluate-round.sh rmsnorm 1

# 查看进度
cat reference/rmsnorm/README.md      # 优化历史
cat reference/rmsnorm/TRAPS.md       # 累积的陷阱
cat reference/rmsnorm/solutions.jsonl # 变体 DAG
```

---

## 📁 项目结构

```
KernelForge-MultiAgent/
├── master/                  # Master Agent 指南
│   └── MASTER.md
├── reference/               # 家族归档（跨轮次记忆）
│   └── <family>/
│       ├── README.md        # 优化历史、变体树
│       ├── TRAPS.md         # 陷阱记录 ⭐
│       ├── baseline.json    # 性能基准
│       ├── solutions.jsonl  # 变体 DAG
│       └── variants/        # 所有历史变体
├── rounds/                  # 轮次工作区
│   └── round-<N>/<family>/
│       ├── BRIEF.md         # Master 生成的本轮指导
│       ├── docs/draft.md    # Sub Agent 实现计划
│       ├── src/             # 内核代码
│       ├── profile/         # NCU 报告
│       ├── benchmark.csv    # 性能追踪
│       └── solutions.jsonl  # 本轮候选
├── scripts/                 # 自动化脚本
│   ├── start-campaign.sh    # 启动 Campaign
│   ├── run-round.sh         # 运行单轮
│   └── evaluate-round.sh    # 评估轮次
├── docs/                    # 文档
│   ├── CLOSED_LOOP.md       # 闭环流程详解
│   ├── STRUCTURED_WORKFLOW.md
│   ├── HUMANIZE_INTEGRATION.md
│   └── ...
├── skills/                  # 知识库
│   ├── KernelWiki/          # 优化知识（2179 PRs）
│   └── ncu-report-skill/    # 性能分析
├── kernels/operators/       # 内核实现目录（当前主线聚焦 3 类算子）
└── prompts/                 # 三阶段提示词
```

---

## 🔄 真实闭环迭代流程（强制执行）

本项目的闭环**不是「写一版 kernel 就报个数」**，而是 **NCU 真实硬件数据驱动 + KernelWiki 证据约束**
的多轮迭代。每一轮都必须基于本机 GPU 实采数据决策，不允许凭空猜测优化方向，也不允许用虚的加速比口径。

### 三条铁律

1. **baseline 捕获阶段只记录 latency**：baseline 只需要有真实正确性、真实时间和真实 workload 覆盖率，
   不强调任何加速比。
2. **闭环阶段加速比唯一口径 = `sol/base`**：只看「我的实现 vs 当前不可变官方 baseline / expert baseline」。
   这里的 baseline 只能是 `baseline.json` 里锁定的 `baseline_solution`；`anchor`、`parent`、
   历史自研候选都只是派生关系或中间证据，**不能被重命名成 baseline**。
   vs 朴素 PyTorch 参考实现（`sol/ref`）的加速比**没有意义**，仅作正确性旁证，不作为成绩。
3. **优化方向必须来自 NCU**：每一轮改 kernel 前，必须先在本机用 Nsight Compute 采集
   我的 kernel 与官方 baseline 的真实指标，定位瓶颈后再动手。禁止拍脑袋优化。
   对 RTX 5070 / sm_120，**只允许使用** `/usr/local/NVIDIA-Nsight-Compute-2025.2/ncu`；
   `/usr/local/cuda/bin/ncu` 在当前 WSL 环境会报 `LibraryNotLoaded`，不能作为有效证据。
4. **快迭代 workload 口径固定为 3 个代表点**：每个主线算子默认只测
   低 / 中 / 高各 1 个真实 safetensors workload，并将其写入该 family 的
   `selected_workloads_file`。除非用户明确要求扩展覆盖，否则闭环 benchmark / validate /
   决策都只按这 3 个代表 workload 执行。
5. **决策必须参考 KernelWiki**：每一轮做 ACCEPT / REJECT 决策前，必须记录并引用
   `skills/KernelWiki` 中与当前瓶颈匹配的页面，说明这些页面如何支撑本轮优化方向。RTX 5070 属于
   Blackwell 架构，因此必须使用 KernelWiki；但对仅限 SM100/B200 的特性要明确标注“不适用于 sm_120”。
6. **baseline 必须 immutable 且可追溯**：`baseline_solution`、`comparison_denominator`、
   `baseline_source_kind` 必须视为锁定字段；轮次配置不得私自覆盖。每个 family 都必须记录
   baseline provenance，至少包含官方 solution 名、来源 json 路径、source kind 和锁定语义。
7. **正确性必须复用官方 harness 语义，且显式保留 invalid-value 拒绝**：
   正确性判定优先复用 FlashInfer-Bench 官方 `compute_error_stats` / 容差逻辑；
   同时必须显式拒绝 `NaN`、`Inf` 等 invalid values，不能只做 atol/rtol 比较。
8. **新轮次必须基于官方 baseline 源码派生**：后续优化轮次不再允许直接从旧的 rejected 自研候选起步，
   必须先读取数据集 `solutions/baseline/...` 中的官方 baseline 源码，再基于它派生新的 `kernel.cu` 起点。
   工作区里的 `src/official_baseline/` 是强制参考目录。
   但派生时必须同时继承上一轮已经验证的 NCU 结论、KernelWiki 页面和 `TRAPS.md` 陷阱记录，
   不允许只看新 baseline 源码就把历史证据全部丢掉。
9. **证据驱动 ACCEPT/REJECT**：正确性不过 → REJECT；`sol/base < 1.05` → REJECT/继续迭代；
   `sol/base ≥ 1.05` 且正确，并且本轮 NCU + KernelWiki 证据齐全 → ACCEPT 并归档。
10. **官方 baseline 的 NCU 可复用，但不能乱复用**：只有在 `definition + batch_size + device +
   baseline_solution + NCU版本` 完全一致时，才能复用同一份 baseline NCU 报告；否则必须重跑。
11. **verifier / acceptance 角色不能反向改实现**：验证者只能读取候选、日志和证据，
   不能在 verification prompt 里要求 verifier 直接补功能，更不能让 acceptance 角色修改实现代码后再给通过。
12. **每轮必须留下 audit log**：必须记录 writer / verifier / acceptance 各自读写了哪些文件。
   其中 verifier / acceptance 的 `modified_files` 必须为空；否则该轮直接视为角色隔离失效。
13. **归档加速比必须可回溯到原始日志**：凡是写入 `reference/<family>/solutions.jsonl` 的
   `sol_vs_base`，都必须能在同仓库里回溯到对应轮次的 `decision.json`、`validate.log`
   和必要的 `ncu_evidence.json`。如果只有 reference 里的摘要数字、没有原始日志与决策文件，
   则该条目只能算“待补证据”，不能视为完全审计通过的真实归档成绩。

### 10 步循环

```
1. derive    建 rounds/round-<N>/<family>/{src,profile,docs}，先展开官方 baseline 源码，再从其派生工作副本
2. brief     写 BRIEF.md：本轮目标(sol/base 提升幅度)、优化方向(来自上一轮 NCU)、要避开的 TRAPS，
             并列出本轮必须参考的 KernelWiki 页面
3. optimize  按 NCU 结论修改 kernel（方向必须有 NCU 依据），并明确记录它相对官方 baseline 源码的派生关系
4. benchmark scripts/workflow/fib_inproc_validate.py 在本机 5070 计时（默认 warmup=3, iters=10）
5. validate  同脚本逐 workload 正确性比对（复用官方 harness 语义 + 显式 NaN/Inf 拒绝，
             atol/rtol=1e-2）；只统计真实 safetensors 文件完整存在的 workload，并记录覆盖率，不过即 REJECT
6. compare   以官方 baseline 为锚点计算 sol/base —— 唯一成绩口径
7. decide    只有在“本轮真实 NCU 证据 + KernelWiki 依据”齐全时才允许 REJECT / ACCEPT
8. audit     补齐 writer / verifier / acceptance 的文件读写审计；verifier/acceptance 不得写实现
9. document  仅 ACCEPT 才更新 reference/<family>/（README + solutions.jsonl + variants/）
10. lessons  把失败教训写入 reference/<family>/TRAPS.md（下一轮 brief 自动注入，避免重复踩坑）
11. plan     用本轮 NCU 数据 + KernelWiki 页面定位新瓶颈，规划下一轮优化方向；回到第 1 步
```

终止条件：`sol/base ≥ 1.05`（收敛 ACCEPT）/ 连续 3 轮无改进 / 达到轮次上限。

### 固定命令（每一轮都用这两条）

```bash
# (A) 验证正确性 + 对官方 baseline 比性能（成绩口径 sol/base）
python scripts/workflow/fib_inproc_validate.py \
  --dataset /mnt/d/Agent/flashinfer-trace \
  --definition rmsnorm_h4096 \
  --solution  <我的 solution 名> \
  --baseline  flashinfer_wrapper_2e27cd      # 官方 FlashInfer baseline

# (B) NCU 本机实采（分别采我的 kernel 与官方 baseline，对比定位瓶颈）
/usr/local/NVIDIA-Nsight-Compute-2025.2/ncu -f -o profile/<family>/sol_bs<N> \
  --set full <python 绝对路径> scripts/workflow/ncu_driver.py \
  --dataset /mnt/d/Agent/flashinfer-trace --definition rmsnorm_h4096 \
  --solution <我的 solution 名> --batch-size <N> --which sol
```

### 每轮评估前必须补齐的证据文件

从现在开始，每个轮次目录都必须有下面两个文件，否则 `./scripts/evaluate-round.sh` 会直接失败：

1. `rounds/round-<N>/<family>/profile/ncu_evidence.json`
   - 记录本轮 solution 的真实 NCU 报告
   - 记录官方 baseline 的真实 NCU 报告；若复用历史报告，必须满足同 `definition + batch_size + device + baseline_solution + NCU版本`
   - 写明本轮瓶颈、关键指标、决策驱动因素
2. `rounds/round-<N>/<family>/docs/kernelwiki_evidence.json`
   - 记录本轮参考的 `skills/KernelWiki` 页面
   - 说明这些页面如何支持本轮优化方向
   - 标注哪些结论适用于 `sm_120`

> 注：本机为 WSL2 + RTX 5070(sm_120)。WSL 不支持 CUDA IPC，故用单进程验证器
> `fib_inproc_validate.py` 替代官方多进程 runner；NCU 需用 Nsight Compute 2025.2
> （2025.1 在本机 WSL 报 LibraryNotLoaded）。详见 `CLAUDE.md`。

---

## 📊 当前聚焦的 3 类算子

KernelForge-MultiAgent 当前的主动闭环资源只投向 3 类高收益算子。

| 算子家族 | 代表 definition / 场景 | 当前优先级 | 说明 |
|-----|------|--------|------|
| **DSA Sparse Attention** | `dsa_sparse_attention_*` | P0 | 竞赛型高收益主战场 |
| **GDN Prefill** | `gdn_prefill_*` | P0 | 变长 prefill，结构空间大 |
| **DSA Top-k Indexer** | `dsa_topk_indexer_*` | P1 | 索引/访存/图捕获优化空间大 |

### 非主线算子说明

- `rmsnorm`：保留为样板与流程验证参考，但默认不再作为主线优化对象。
- `gemm`、`rope`、`sampling`、`gqa_ragged` 等：当前不进入主动闭环排期。
- 若用户明确指定，仍可单独重开某个非主线算子。

### 快速启动优化

```bash
# DSA Sparse Attention
./scripts/start-campaign.sh dsa_sparse_attention 10

# GDN Prefill
./scripts/start-campaign.sh gdn_prefill 10

# DSA Top-k Indexer
./scripts/start-campaign.sh dsa_topk_indexer 10

说明：
- 上面是当前主线建议入口。
- `rmsnorm` 仍可用于流程联调，但默认不再作为性能主战场。
- 当前聚焦算子说明见 [`docs/SUPPORTED_OPERATORS.md`](docs/SUPPORTED_OPERATORS.md)。

---

## 📚 核心文档

| 文档 | 用途 |
|-----|------|
| [`docs/CLOSED_LOOP.md`](docs/CLOSED_LOOP.md) | **闭环优化流程详解** ⭐ |
| [`master/MASTER.md`](master/MASTER.md) | **Master Agent 指南** ⭐ |
| [`docs/HUMANIZE_INTEGRATION.md`](docs/HUMANIZE_INTEGRATION.md) | Humanize 使用指南 |
| [`docs/STRUCTURED_WORKFLOW.md`](docs/STRUCTURED_WORKFLOW.md) | 整体工作流 |
| [`docs/SUPPORTED_OPERATORS.md`](docs/SUPPORTED_OPERATORS.md) | 当前聚焦的 3 类算子说明 |
| [`docs/FLASHINFER_BENCH_VALIDATION.md`](docs/FLASHINFER_BENCH_VALIDATION.md) | 验收标准 |
| [`reference/<family>/README.md`](reference/rmsnorm/README.md) | 家族归档示例 |
| [`reference/<family>/TRAPS.md`](reference/rmsnorm/TRAPS.md) | 陷阱记录示例 |

---

## 🎓 核心概念

### Reference 家族归档

**跨轮次持久化记忆**，包含：
- **README.md**: 优化历史、当前最优、变体树
- **TRAPS.md**: 陷阱记录（核心创新！）⭐
- **baseline.json**: 家族级状态摘要（当前锚点、最新结果、下一轮方向）
- **solutions.jsonl**: 完整变体 DAG
- **variants/**: 所有历史实现归档

### TRAPS.md 陷阱记录

系统化记录优化陷阱：
```markdown
## 陷阱 1: 数值稳定性问题
**症状**: NaN, 溢出
**原因**: FP16 精度不足
**解决方案**: [具体代码]
**预防措施**: [检查清单]
```

**自动注入**: Master Agent 在每轮的 BRIEF.md 中自动注入已知陷阱

### 证据驱动决策

客观标准：
```python
if correctness == False:
    return "REJECT"
if speedup >= anchor * 1.05:
    return "ACCEPT"  # 显著改进
else:
    return "REJECT"  # 改进不足
```

---

## ✅ 验收标准

**所有优化必须通过 FlashInfer-Bench 验证**：

```bash
cd D:/Agent/flashinfer-bench-main/flashinfer-bench-main
flashinfer-bench run --local flashinfer-trace --op-type <family>
```

验证内容：
- ✅ 正确性（数值精度）
- ✅ 性能（真实工作负载）
- ✅ 稳定性（方差 < 5%）

---

## 🔧 Skills 和工具

### Humanize Plugin ⭐⭐⭐⭐⭐
**Plan-Execute-Verify 循环**

安装：
```bash
/plugin marketplace add PolyArch/humanize
/plugin install humanize@PolyArch
```

命令：
- `/humanize:gen-plan` - 生成详细计划
- `/humanize:start-rlcr-loop` - 启动 RLCR 循环
- `/humanize:status` - 查看进度

### KernelWiki ⭐⭐⭐⭐⭐
**优化知识库**

- 位置: `skills/KernelWiki/`
- 规模: 2179 PRs, 48 wiki 页面
- 覆盖: Blackwell/Hopper, TMA, TMEM, tcgen05, FP8
- 用法要求：每一轮闭环都必须在工作区记录本轮实际参考的页面与适用性判断，不能只“看过不记”

### ncu-report-skill ⭐⭐⭐⭐⭐
**性能分析**

- 位置: `skills/ncu-report-skill/`
- 功能: 瓶颈识别、指标解释、优化建议

---

## 📈 Campaign 终止条件

自动终止当满足以下任一条件：
1. 达到最大轮次
2. 达到目标加速比
3. 连续 3 轮无改进
4. 所有优化方向探索完毕
5. 用户手动终止

---

## 🎯 示例 Campaign

```bash
# 启动 RMSNorm 优化 Campaign
./scripts/start-campaign.sh rmsnorm 10 2.0

# Round 0: 建立基线
# → 结果: 1.23ms, 1.0x, ACCEPT

# Round 1: Warp reduction
# → 结果: 0.95ms, 1.29x, ACCEPT

# Round 2: Shared memory (失败)
# → 结果: 0.98ms, 1.25x, REJECT
# → 发现陷阱: Bank conflict
# → 更新 TRAPS.md

# Round 3: Register blocking
# → 读取 TRAPS.md, 避免 shared memory
# → 结果: 0.82ms, 1.50x, ACCEPT

# Round 4-7: 持续优化...
# → 最终: 0.57ms, 2.15x, 达到目标！
```

---

## 🌟 核心优势

1. **自动化** - Master Agent 编排，减少人工介入
2. **记忆化** - 跨轮次持久化，不重复错误
3. **系统化** - 10-step loop 标准化流程
4. **证据驱动** - 客观决策，可重复验证
5. **陷阱规避** - TRAPS.md 系统化避免失败

---

## 📝 License

MIT License

---

**最后更新**: 2026-06-15  
**版本**: v2.0 - Closed-Loop Automatic Optimization  
**灵感**: AKO4X + Humanize + KDA
