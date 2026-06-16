# Blackwell Kernel Optimization Knowledge Base — Build Plan

## Goal Description

Build a comprehensive, one-time knowledge base of GPU kernel optimization techniques for NVIDIA Blackwell (SM100) and Hopper (SM90) architectures. The KB is optimized for LLM agent retrieval: structured YAML frontmatter, canonical IDs, cross-referenced indices, and terse English prose. It covers PRs from 5 major repositories (CUTLASS, SGLang, vLLM, FlashInfer, PyTorch) from Jan 2025 to Apr 2026, two kernel competitions (GPU Mode NVFP4 Hackathon, FlashInfer MLSys 2026), and optimization work from DeepSeek and Qwen teams. Scope is single-device kernel optimization only (no distributed system topics). Implementations are preserved as completely as possible; files exceeding 1000 lines use key code excerpts with detailed documentation.

## Acceptance Criteria

- AC-1: Three-layer directory structure exists with correct layout
  - Positive Tests:
    - `sources/prs/{cutlass,sglang,vllm,flashinfer,pytorch}/` directories exist and contain PR files
    - `sources/contests/`, `sources/docs/`, `sources/blogs/` exist and contain content
    - `wiki/{hardware,techniques,patterns,kernels,languages,migration}/` exist and contain pages
    - `queries/by-{problem,technique,hardware-feature,repo,kernel-type,language}.md` exist
  - Negative Tests:
    - No `wiki/systems/` directory (excluded per scope decision)
    - No empty directories without content

- AC-2: All source pages have valid YAML frontmatter matching their page-type schema
  - Positive Tests:
    - Every `sources/prs/*/PR-*.md` has required fields: id, repo, pr, title, author, date, url, source_category, architectures, tags, techniques, hardware_features, kernel_types, languages, captured_at, status
    - Every `sources/docs/*.md` has: id, title, url, source_category, architectures, tags, retrieved_at
    - Every `sources/blogs/*.md` has: id, title, author, url, source_category, architectures, tags, retrieved_at
    - Every `sources/contests/*/*.md` has: id, title, source_category, architectures, tags
  - Negative Tests:
    - A source file missing required `id` field is rejected by validator
    - A source file with tags not in `data/tags.yaml` is flagged

- AC-3: All wiki pages have valid frontmatter matching their page-type schema
  - Positive Tests:
    - Every `wiki/techniques/*.md` has: id, title, type=technique, architectures, tags, confidence, reproducibility (>= snippet), prerequisites, related, sources
    - Every `wiki/kernels/*.md` has: id, title, type=kernel, architectures, tags, confidence, reproducibility (>= snippet), kernel_types, languages, related, sources, performance_claims
    - Every `wiki/hardware/*.md` has: id, title, type=hardware, architectures, tags, confidence, related, sources, aliases
    - Every `wiki/patterns/*.md` has: id, title, type=pattern, tags, symptoms, candidate_techniques, related, sources
    - Every `wiki/languages/*.md` has: id, title, type=language, tags, related, sources, reproducibility (>= snippet)
    - Every `wiki/migration/*.md` has: id, title, type=migration, from_arch, to_arch, tags, related, sources, blackwell_relevance
  - Negative Tests:
    - A technique page with `reproducibility: concept` is rejected (must be >= snippet)
    - A migration page without `blackwell_relevance` field is flagged
    - A wiki page referencing a non-existent source id is flagged

- AC-4: Query index pages are generated from frontmatter metadata
  - Positive Tests:
    - `queries/by-problem.md` contains entries derived from `wiki/patterns/*.md` symptoms fields
    - `queries/by-technique.md` lists all technique pages with their tags and source counts
    - `queries/by-hardware-feature.md` maps hardware features to wiki pages using them
    - `queries/by-repo.md` lists all source PRs grouped by repository
    - `queries/by-kernel-type.md` maps kernel types to relevant wiki pages
    - `queries/by-language.md` maps DSL/language tags to relevant pages
    - A `scripts/generate-indices.py` script reproduces these files from frontmatter
  - Negative Tests:
    - Manually editing a query file and re-running the generator overwrites the manual edit
    - The generator ignores files outside `sources/` and `wiki/`

- AC-5: Coverage meets scope requirements
  - AC-5.1: At least 5 Blackwell-related PRs ingested per repository (CUTLASS, SGLang, vLLM, FlashInfer, PyTorch)
    - Positive: `sources/prs/cutlass/` contains >= 5 PR files with sm100/blackwell tags
    - Negative: A PR file tagged only `sm90` without `blackwell_relevance` is not counted
  - AC-5.2: Both competitions have dedicated source pages
    - Positive: `sources/contests/gpu-mode-nvfp4/` has pages for all 4 problems
    - Positive: `sources/contests/flashinfer-mlsys26/` has pages for all 3 tracks
    - Negative: A contest page without problem definition or technique analysis is incomplete
  - AC-5.3: DeepSeek and Qwen kernel work is covered
    - Positive: Wiki kernel pages exist for DeepGEMM, FlashMLA, NSA, GatedDeltaNet
    - Negative: DeepEP, DualPipe, EPLB do NOT have wiki pages (system-level, excluded)
  - AC-5.4: Core Blackwell hardware features all have wiki pages
    - Positive: `wiki/hardware/` has pages for tcgen05-mma, tmem, clc, tma, 2sm-cooperative, nvfp4, pdl-gdc
    - Negative: No hardware page exists without at least one source reference

- AC-6: Technique and kernel pages include code implementations
  - Positive Tests:
    - Every `wiki/techniques/*.md` has a "Key Code" or "Implementation" section with compilable code or inline PTX
    - Every `wiki/kernels/*.md` has code showing the kernel's core loop or key optimization
    - Code snippets include language annotation (cuda, python, ptx)
  - Negative Tests:
    - A technique page with only prose description and no code is rejected (reproducibility < snippet)

- AC-7: CLAUDE.md, index.md, and data files provide complete navigation schema
  - Positive Tests:
    - `CLAUDE.md` documents all page types, required fields, navigation flow, and references `data/*.yaml`
    - `index.md` has curated top-level navigation linking to all wiki sections and query pages
    - `data/tags.yaml` defines all valid tags with categories
    - `data/aliases.yaml` maps all known aliases (tcgen05=UMMA, TMEM=tensor memory, etc.)
    - `data/schemas.yaml` defines required fields per page type
  - Negative Tests:
    - A tag used in a source/wiki file that is not in `data/tags.yaml` is invalid

- AC-8: Performance claims include environment metadata
  - Positive Tests:
    - Every `performance_claims` entry in kernel pages specifies: gpu, dtype, shape, metric, value, source_id
    - FlashAttention-4 claim "1605 TFLOPS" links to its source with B200, BF16 context
  - Negative Tests:
    - A bare "2x speedup" without GPU SKU and baseline reference is incomplete

- AC-9: Confidence levels follow defined evidence rules
  - Positive Tests:
    - `verified` pages have evidence_basis with both official-doc and upstream-code entries
    - `source-reported` pages cite at least one authoritative source
    - `experimental` is used for undocumented PTX tricks
  - Negative Tests:
    - A page marked `verified` with only a blog post source is invalid

- AC-10: First-class DSL coverage for CuTe DSL, CUDA C++, PTX, and Triton
  - Positive Tests:
    - `wiki/languages/` has dedicated pages for cute-dsl, cuda-cpp, ptx, triton
    - Each language page has Blackwell-specific examples at snippet level or above
  - Negative Tests:
    - TileLang and cuTile do NOT have dedicated wiki/languages pages (secondary coverage)

## Path Boundaries

### Upper Bound (Maximum Acceptable Scope)
Complete ingestion of all Blackwell-related PRs from all 5 repos (100+ PRs), full contest solution analysis with performance data, every technique page with runnable examples, comprehensive pattern diagnosis pages covering all known Blackwell performance bottlenecks, and a fully automated validator + index generator pipeline.

### Lower Bound (Minimum Acceptable Scope)
At minimum 5 PRs per repo (25+ total), both contests covered at problem-definition level, core hardware pages for all 7 SM100 features, 8+ technique pages with code snippets, 5+ pattern diagnosis pages, 4+ kernel case studies, 4 language pages, and working index generation from frontmatter.

### Allowed Choices
- Can use: Python (PyYAML) for tooling scripts, Markdown with YAML frontmatter for all content, `gh` CLI for PR data collection, web search/fetch for blog and contest data
- Can use: CuTe DSL, CUDA C++, PTX, Triton for code examples
- Cannot use: Database backends, static site generators, or external search engines
- Cannot use: Non-English for wiki content body (English canonical per user decision)
- Fixed: Karpathy LLM Wiki three-layer pattern (sources → wiki → queries)
- Fixed: One source file per PR, topic-level synthesis in wiki

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only.

### Conceptual Approach

```
Phase 1: Skeleton
  Create CLAUDE.md, data/*.yaml, directory structure, scripts/

Phase 2: Data Collection (parallelizable across repos)
  For each repo in [cutlass, sglang, vllm, flashinfer, pytorch]:
    gh search prs --repo {repo} "{keyword}" --limit 100
    For each relevant PR:
      Extract metadata → sources/prs/{repo}/PR-{N}.md
  For contests:
    Fetch problem definitions, solutions, leaderboards
  For blogs/docs:
    Fetch and summarize key articles

Phase 3: Wiki Synthesis
  For each hardware feature: synthesize from sources → wiki/hardware/
  For each technique: synthesize from sources → wiki/techniques/
  For each kernel case study: synthesize → wiki/kernels/
  For each problem pattern: synthesize → wiki/patterns/
  For each language: synthesize → wiki/languages/
  For migration patterns: synthesize → wiki/migration/

Phase 4: Index Generation
  Run scripts/generate-indices.py → queries/*.md
  Run scripts/validate.py → check all schemas

Phase 5: Review
  Verify link integrity, tag consistency, reproducibility coverage
  Curate index.md top-level navigation
```

### Relevant References
- `research-deepseek-qwen.md` — Already-collected DeepSeek/Qwen research data
- `draft.md` — Original design draft with architecture details and collected information
- Karpathy LLM Wiki pattern: `https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f`
- NVIDIA Blackwell Tuning Guide: `https://docs.nvidia.com/cuda/blackwell-tuning-guide/`
- tcgen05 tutorial: `https://gau-nernst.github.io/tcgen05/`
- Colfax CUTLASS tutorials: `https://research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/`
- FlashAttention-4 paper: `https://arxiv.org/abs/2603.05451`
- GPU Mode NVFP4 Hackathon: `https://forums.developer.nvidia.com/t/join-us-for-the-blackwell-nvfp4-kernel-hackathon-with-nvidia-and-gpu-mode/350092`
- FlashInfer MLSys 2026 Contest: `https://mlsys26.flashinfer.ai/`

## Dependencies and Sequence

### Milestones

1. **Skeleton**: Repository structure, schemas, and tooling
   - Create CLAUDE.md with complete schema documentation
   - Create `data/tags.yaml`, `data/aliases.yaml`, `data/schemas.yaml`
   - Create directory structure
   - Create `scripts/validate.py` and `scripts/generate-indices.py`

2. **Data Collection**: Parallel ingestion from all sources
   - Phase A: PR collection from 5 repos (fully parallel)
   - Phase B: Contest data collection (parallel with Phase A)
   - Phase C: Blog/doc collection (parallel with Phase A)
   - Phase D: Process already-collected DeepSeek/Qwen data into source pages

3. **Wiki Synthesis**: Build knowledge pages from collected sources
   - Depends on: Milestone 2 (needs source data)
   - Phase A: Hardware feature pages (7 core features)
   - Phase B: Technique pages (10+ techniques)
   - Phase C: Kernel case studies (8+ kernels)
   - Phase D: Pattern diagnosis pages (5+ patterns)
   - Phase E: Language/DSL pages (4 first-class)
   - Phase F: Migration pages (Hopper→Blackwell patterns)

4. **Index Generation and Validation**: Automated finishing
   - Depends on: Milestone 3 (needs wiki pages)
   - Run generate-indices.py to build query pages
   - Run validate.py to check all schemas
   - Curate index.md
   - Fix any validation errors

## Task Breakdown

| Task ID | Description | Target AC | Tag | Depends On |
|---------|-------------|-----------|-----|------------|
| task1 | Create directory structure, CLAUDE.md, data/*.yaml, scripts/ | AC-1, AC-7 | coding | - |
| task2 | Create `scripts/validate.py` (YAML frontmatter validator) | AC-2, AC-3 | coding | task1 |
| task3 | Create `scripts/generate-indices.py` (query page generator) | AC-4 | coding | task1 |
| task4 | Collect CUTLASS Blackwell PRs → sources/prs/cutlass/ | AC-2, AC-5.1 | coding | task1 |
| task5 | Collect SGLang Blackwell PRs → sources/prs/sglang/ | AC-2, AC-5.1 | coding | task1 |
| task6 | Collect vLLM Blackwell PRs → sources/prs/vllm/ | AC-2, AC-5.1 | coding | task1 |
| task7 | Collect FlashInfer Blackwell PRs → sources/prs/flashinfer/ | AC-2, AC-5.1 | coding | task1 |
| task8 | Collect PyTorch Blackwell PRs → sources/prs/pytorch/ | AC-2, AC-5.1 | coding | task1 |
| task9 | Collect GPU Mode NVFP4 Hackathon data → sources/contests/ | AC-2, AC-5.2 | coding | task1 |
| task10 | Collect FlashInfer MLSys 2026 Contest data → sources/contests/ | AC-2, AC-5.2 | coding | task1 |
| task11 | Collect official docs and blog summaries → sources/docs/, sources/blogs/ | AC-2 | coding | task1 |
| task12 | Process research-deepseek-qwen.md into source pages | AC-2, AC-5.3 | coding | task1 |
| task13 | Write wiki/hardware/ pages (tcgen05, tmem, clc, tma, 2sm, nvfp4, pdl-gdc) | AC-3, AC-5.4 | coding | task4,task11 |
| task14 | Write wiki/techniques/ pages (warp-spec, persistent, swizzling, pipeline, etc.) | AC-3, AC-6 | coding | task4-task12 |
| task15 | Write wiki/kernels/ pages (DeepGEMM, FlashMLA, FlashAttn4, NSA, GatedDeltaNet, etc.) | AC-3, AC-5.3, AC-6, AC-8 | coding | task4-task12 |
| task16 | Write wiki/patterns/ pages (low-SM-util, memory-bound, register-pressure, etc.) | AC-3 | coding | task14,task15 |
| task17 | Write wiki/languages/ pages (cute-dsl, cuda-cpp, ptx, triton) | AC-3, AC-10 | coding | task4-task12 |
| task18 | Write wiki/migration/ pages (Hopper→Blackwell patterns) | AC-3, AC-9 | coding | task13,task14 |
| task19 | Run generate-indices.py → queries/*.md | AC-4 | coding | task13-task18 |
| task20 | Run validate.py, fix errors, curate index.md | AC-2,AC-3,AC-7 | coding | task19 |
| task21 | Verify confidence levels and evidence basis across all wiki pages | AC-9 | analyze | task20 |
| task22 | Verify performance claims have complete environment metadata | AC-8 | analyze | task20 |
| task23 | Final coverage audit: check all AC-5 sub-criteria met | AC-5 | analyze | task20 |

Tasks 4-12 can execute fully in parallel. Tasks 13-18 can partially overlap but depend on source data.

## Claude-Codex Deliberation

### Agreements
- Metadata-first architecture with generated query indices is the correct approach over hand-curated indices
- Blackwell-first with Hopper content admitted only for explicit transfer value
- English as canonical wiki language
- Per-PR source files with topic-level wiki synthesis above them (correct abstraction boundary)
- Confidence field with formalized evidence rules is necessary
- Performance claims require environment metadata (GPU SKU, dtype, shape, measurement context)
- Page-type-specific schemas with different required fields per wiki section
- Reproducibility ladder: technique and kernel pages must be >= snippet level
- Controlled vocabulary stored as machine-readable data files (`data/*.yaml`)

### Resolved Disagreements

- **Index generator implementation language**: Claude proposed shell script, Codex argued shell is poor for YAML parsing. Resolution: Python with PyYAML. Rationale: YAML frontmatter parsing requires a real parser; Python is available and appropriate.

- **`source_type` semantics**: Codex noted "reliability tiers" was imprecise — the enum is a taxonomy, not a confidence model. Resolution: Renamed to `source_category` (taxonomy); wiki pages use separate `confidence` + `evidence_basis` fields for reliability. Rationale: separating taxonomy from confidence avoids confusion.

- **Sources "immutability"**: Codex noted local .md summaries may need correction, so "immutable" is misleading. Resolution: The external URL/SHA/date are the immutable anchors; the local summary can be corrected with a `captured_at` update. Rationale: practical accuracy > theoretical purity.

- **`by-problem.md` generation source**: Codex argued `wiki-pattern` should be canonical for problem diagnosis, not scattered `symptoms` fields on technique pages. Resolution: `wiki/patterns/` pages are the canonical source for `by-problem.md`. Technique/kernel pages may have optional `symptoms` backlinks. Rationale: single source of truth for diagnosis flow.

- **Multi-value languages**: Claude had scalar `language`, Codex required array. Resolution: `languages: [cute-dsl, ptx]` array field. Rationale: most real PRs span multiple languages.

- **Generator scan scope**: Codex required narrowing to `sources/**/*.md` and `wiki/**/*.md` only. Resolution: accepted. Rationale: prevents ingesting CLAUDE.md, planning docs, etc.

### Convergence Status
- Final Status: `converged`
- Rounds: 2 (all REQUIRED_CHANGES addressed, no remaining high-impact DISAGREE)

## Pending User Decisions

- DEC-1: Primary consumer
  - Claude Position: LLM agents primarily
  - Codex Position: Agreed
  - Decision Status: `LLM agents primarily` (user confirmed)

- DEC-2: Reproducibility level
  - Claude Position: Compilable snippet (< 50 lines)
  - Codex Position: Snippet-first is correct default
  - Decision Status: `Save complete implementations when possible; >1000 lines → key code + detailed docs` (user override)

- DEC-3: Distributed system topics
  - Claude Position: Include in separate wiki/systems/ track
  - Codex Position: Reasonable to split
  - Decision Status: `Excluded — kernel-only scope` (user chose "No, kernel-only")

- DEC-4: First-class DSL coverage
  - Claude Position: All DSLs with varying depth
  - Codex Position: No strong opinion
  - Decision Status: `CuTe DSL, CUDA C++, PTX, Triton — first-class. TileLang, cuTile, JAX Pallas — secondary (mentioned but no dedicated pages)` (user confirmed)

- DEC-5: Wiki language
  - Claude Position: English canonical
  - Codex Position: Agreed
  - Decision Status: `English` (user confirmed)

- DEC-6: Maintenance model
  - Claude Position: Continuously maintained with ingest workflow
  - Codex Position: No strong opinion
  - Decision Status: `One-time comprehensive build` (user chose — no ingest workflow needed, no log.md)

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as "AC-", "Milestone", "Step", "Phase", or similar workflow markers
- These terms are for plan documentation only, not for the resulting codebase
- Use descriptive, domain-appropriate naming in code instead

### Source Page Frontmatter Schema (PR)
```yaml
---
id: pr-cutlass-2145
repo: NVIDIA/cutlass
pr: 2145
title: "Add SM100 tcgen05 GEMM support"
author: username
date: 2025-08-15
url: https://github.com/NVIDIA/cutlass/pull/2145
source_category: upstream-code
architectures: [sm100]
tags: [tcgen05, gemm, bf16, warp-specialization]
techniques: [warp-specialization, tmem-double-buffering]
hardware_features: [tcgen05, tmem, clc]
kernel_types: [gemm]
languages: [cute-dsl, cuda-cpp]
captured_at: 2026-04-16
status: merged
merge_sha: abc123
---
```

### Wiki Page Frontmatter Schema (Technique)
```yaml
---
id: technique-warp-specialization
title: "Warp Specialization on Blackwell"
type: technique
architectures: [sm100, sm90]
tags: [warp-specialization, tcgen05, tmem]
confidence: verified
reproducibility: snippet
prerequisites: [hw-tmem, hw-tcgen05-mma]
related: [technique-persistent-kernels, technique-pipeline-stages]
sources: [pr-cutlass-2145, blog-tcgen05-tutorial]
aliases: [warp specialization, warp-spec]
---
```

### Wiki Page Frontmatter Schema (Pattern — Diagnosis Flow)
```yaml
---
id: pattern-low-sm-utilization
title: "Low SM Utilization"
type: pattern
tags: [performance, occupancy]
symptoms: [low-sm-utilization, tail-effect, load-imbalance]
candidate_techniques: [technique-clc, technique-persistent-kernels, technique-pipeline-stages]
related: [pattern-tail-effect, pattern-load-imbalance]
sources: [pr-cutlass-xxxx, blog-tcgen05-tutorial]
---
## Symptom
...
## Likely Causes
...
## Candidate Techniques (table)
...
## Examples
...
## Caveats
...
```

### Wiki Page Frontmatter Schema (Kernel)
```yaml
---
id: kernel-flash-attention-4
title: "FlashAttention-4"
type: kernel
architectures: [sm100]
tags: [attention, bf16, tcgen05, tmem, 2sm-cooperative]
confidence: verified
reproducibility: snippet
kernel_types: [attention]
languages: [cute-dsl]
related: [technique-warp-specialization, technique-2sm-cooperative, hw-tmem]
sources: [blog-flash-attention-4, paper-flash-attention-4]
performance_claims:
  - gpu: B200
    dtype: bf16
    shape: "seqlen=8192, headdim=128"
    metric: TFLOPS
    value: 1605
    utilization: "71%"
    source_id: paper-flash-attention-4
---
```

### Confidence Evidence Rules
- `verified`: Requires evidence_basis with >= 1 `official-doc` AND >= 1 `upstream-code`
- `source-reported`: Requires >= 1 authoritative source (paper, official blog, major repo)
- `inferred`: Synthesized from multiple sources, not directly confirmed by any single one
- `experimental`: Undocumented behavior, PTX tricks, version-sensitive. Must note CUDA version.

### Controlled Vocabulary Files
- `data/tags.yaml`: All valid tags grouped by category (hardware_features, techniques, kernel_types, languages, architectures)
- `data/aliases.yaml`: Canonical → alias mappings (e.g., tcgen05: [UMMA, tensor_core_gen05, "tensor core generation 5"])
- `data/schemas.yaml`: Required and optional fields per page type, consumed by `scripts/validate.py`

### Search Keywords for PR Collection
```
blackwell, sm100, sm_100, sm_100a, tcgen05, tmem, tensor memory,
cuda 13, B200, B100, nvfp4, fp4, fp8, cutile, umma, clc, 2sm,
warp specialization, persistent kernel, fp8_moe, moe, deepseek,
sparse attention, gated delta net, qwen, grouped gemm, tma,
block scale, microscaling, triton sm100, hopper blackwell migration
```

--- Original Design Draft Start ---

# Blackwell Kernel Wiki — 调研与建设方案草案

## 一、项目目标

在本地仓库中建立一个完整的 Blackwell (SM100) 和 Hopper (SM90) GPU Kernel 优化知识库，用于指导 LLM Agent 编写 B200 机器上的高性能 kernel。

知识库需要：
- 保存大量原始信息（PR、比赛数据、博客、文档）
- 通过多层文档作为入口，方便 LLM 查询
- 支持按问题、技巧、硬件特性、kernel 类型、语言等多维度交叉索引

## 二、调研范围

### 代码仓库 PR（2025-01 至今）
- **NVIDIA/cutlass** — CUTLASS 4.x Blackwell 支持，CuTe SM100 atoms
- **sgl-project/sglang** — SGLang Blackwell kernel 集成
- **vllm-project/vllm** — vLLM Blackwell 支持
- **flashinfer-ai/flashinfer** — FlashInfer Blackwell attention/MoE kernel
- **pytorch/pytorch** — PyTorch/TorchInductor Blackwell 后端

### 比赛
- **GPU Mode NVFP4 Blackwell Hackathon**（NVIDIA + GPU Mode，2025 Nov - 2026 Feb）
  - Problem 1: NVFP4 Batched GEMV
  - Problem 2: NVFP4 GEMM
  - Problem 3: Gated Dual GEMM
  - Problem 4: Grouped GEMM
- **FlashInfer AI Kernel Generation Contest**（MLSys 2026）
  - Track A: Fused MoE (FP8)
  - Track B: DeepSeek V3.2 Sparse Attention
  - Track C: Gated Delta Net (Qwen3-Next)

### 团队优化工作
- **DeepSeek** — MoE kernel、sparse attention、FP8 训练/推理
- **Qwen** — Gated Delta Net、Blackwell 适配

### 搜索关键词
blackwell, sm100, sm_100, sm_100a, tcgen05, tmem, cuda 13, B200, B100,
nvfp4, fp4, fp8, cutile, umma, clc, 2sm, warp specialization, persistent kernel,
fp8_moe, moe, deepseek, sparse attention, gated delta net, qwen,
grouped gemm, tma, block scale, microscaling, triton, tilelang, cute

## 三、知识库架构（参考 Karpathy LLM Wiki）

采用三层架构：原始数据层 → LLM 维护的知识层 → 交叉索引查询层。

```
blackwell-kernel-wiki/
├── CLAUDE.md                  # Schema：LLM 操作指南、约定、工作流
├── index.md                   # 主入口：按类别组织的所有页面索引
├── log.md                     # 时间线：所有 ingest 操作记录
│
├── sources/                   # 第一层：原始信息（不可修改）
│   ├── prs/                   # PR 原始数据
│   │   ├── cutlass/           # 每个 PR 一个文件
│   │   ├── sglang/
│   │   ├── vllm/
│   │   ├── flashinfer/
│   │   └── pytorch/
│   ├── contests/              # 比赛信息
│   │   ├── gpu-mode-nvfp4/
│   │   └── flashinfer-mlsys26/
│   ├── docs/                  # 官方文档摘要
│   └── blogs/                 # 社区博客/教程
│
├── wiki/                      # 第二层：LLM 维护的知识页面
│   ├── hardware/              # 硬件特性（TMEM, tcgen05, CLC, TMA...）
│   ├── techniques/            # 优化技巧（warp specialization, pipelining...）
│   ├── patterns/              # 问题→解决方案映射（SM利用率低→...）
│   ├── kernels/               # 具体 kernel 案例分析
│   └── languages/             # 语言/DSL 指南（CuTe, Triton, Tilelang...）
│
└── queries/                   # 第三层：交叉索引入口
    ├── by-problem.md          # 按问题类型查询
    ├── by-technique.md        # 按优化技巧查询
    ├── by-hardware-feature.md # 按硬件特性查询
    ├── by-repo.md             # 按来源仓库查询
    ├── by-kernel-type.md      # 按 kernel 类型查询
    └── by-language.md         # 按编程语言查询
```

### 导航流程
1. `index.md` → 按类别找到页面
2. `queries/by-problem.md` → 有具体性能问题时定位方案
3. `queries/by-technique.md` → 了解某个技巧的所有示例
4. 深入 `wiki/` 页面 → 详细解释和代码示例
5. 跟随 `[source]` 链接 → `sources/` 原始数据和代码

### Source 页面格式
```markdown
---
repo: NVIDIA/cutlass
pr: 1234
title: "PR title"
author: username
date: 2025-06-15
url: https://github.com/NVIDIA/cutlass/pull/1234
tags: [sm100, tcgen05, gemm, fp8, warp-specialization]
techniques: [warp-specialization, tmem-double-buffering]
hardware_features: [tcgen05, tmem, clc]
language: cute-dsl
---
## Summary
...
## Problem
...
## Solution / Techniques
...
## Key Code
...
## Performance
...
```

### Wiki 页面格式
```markdown
---
title: "Page Title"
tags: [tag1, tag2]
related: [link1.md, link2.md]
sources: [PR-123.md, PR-456.md]
---
## Overview
...
## How It Works
...
## When To Use
...
## Examples
...
## Related
- [Link](path) — description
```

### 标签体系
- **硬件特性**: sm100, sm90, tcgen05, tmem, tma, clc, 2sm-cooperative, pdl, gdc, nvfp4, fp8, block-scale, wgmma, cluster
- **优化技巧**: warp-specialization, persistent-kernel, swizzling, pipeline-stages, double-buffering, register-reuse, shared-memory-optimization, tma-multicast, epilogue-fusion, tile-scheduling, communication-overlap
- **Kernel 类型**: gemm, attention, moe, sparse-attention, gemv, grouped-gemm, gated-delta-net, fused-kernel, decode, prefill, quantization
- **编程语言**: cuda-cpp, cute-dsl, triton, tilelang, cutile, ptx, python

## 四、已收集的关键架构信息

### Blackwell SM100 核心变化（vs Hopper SM90）

| 方面 | Hopper SM90 | Blackwell SM100 |
|---|---|---|
| MMA 指令 | wgmma.mma_async（warp group 128 线程） | tcgen05.mma（单线程发射） |
| 累加器 | 寄存器 | TMEM（专用 256KB） |
| 最大 MMA 形状 | m64×n256×k16 | m128×n256×k16 (1SM), m256×n256×k16 (2SM) |
| 吞吐量 | 基线 | BF16 2×, FP4 4× |
| Tile 调度 | 静态/手动 | CLC 硬件动态调度 |
| Shared Memory | 228 KB | 228 KB |
| L2 Cache | 50 MB (H100) | 126 MB (B200) |
| TMEM | 无 | 256 KB/SM (128 rows × 512 cols) |

### 关键新特性
1. **tcgen05.mma** — 7 种变体，支持 TF32/FP16/INT8/FP8/FP6/FP4/NVFP4
2. **Tensor Memory (TMEM)** — 专用累加器内存，消除寄存器压力
3. **2-SM Cooperative** — 两个 SM 协作执行 256×256 MMA
4. **CLC** — 硬件级动态 tile 调度
5. **NVFP4** — 原生 4-bit 浮点，block scale
6. **PDL 默认开启** — kernel 间依赖执行重叠

### 性能优化路径（tcgen05 tutorial 数据）
```
Naive (17%) → Swizzling (46%) → Pipelining (62%) → Warp Specialization (80%)
→ 2-SM MMA (86%) → Persistent/CLC (98% of cuBLAS)
```

### 比赛信息摘要

**GPU Mode NVFP4 Hackathon**:
- 4 个问题，奖品 RTX 5080/5090，Grand prize GB300
- Problem 1 (NVFP4 Batched GEMV) 参赛者从 2000μs 优化到 22.3μs
- 关键技巧：memory coalescing, FP4/FP8 decode intrinsics, PTX assembly, ILP

**FlashInfer MLSys 2026 Contest**:
- 3 个 track 均在 B200 上评测
- 支持 CuTe DSL, CUDA, Tilelang, Triton, cuTile
- 允许人写、AI 生成、或混合提交
- 2026-04-24 截止提交

## 五、调研执行策略

### 并行方式
7 个独立研究任务可完全并行：
1. CUTLASS PRs → `sources/prs/cutlass/`
2. SGLang PRs → `sources/prs/sglang/`
3. vLLM PRs → `sources/prs/vllm/`
4. FlashInfer PRs → `sources/prs/flashinfer/`
5. PyTorch PRs → `sources/prs/pytorch/`
6. 比赛数据 → `sources/contests/`
7. DeepSeek/Qwen 优化 + 社区博客 → `sources/blogs/`

### 每个 PR 的提取模板
- PR number, title, author, date, URL
- 解决什么问题
- 使用了什么技巧
- 利用了哪些硬件特性
- 编程语言
- 关键代码片段
- 性能提升数据

### 后处理
调研完成后：
1. 从 sources 合成 wiki 页面
2. 建立交叉索引
3. 验证链接完整性
4. 确保每个技巧/问题都有足够的案例支撑

## 六、待讨论的设计决策

1. **PR 粒度** — 每个 PR 一个文件 vs. 按主题合并？重要 PR 保存多少代码？
2. **Hopper 内容范围** — 全面收录 vs. 只保留对 Blackwell 有启发的？
3. **语言** — 知识库正文用英文还是中文？
4. **索引粒度** — 一个 kernel 解决多个问题时，在所有相关索引中都出现？
5. **更新机制** — ingest 工作流的具体步骤？
6. **是否需要额外的可视化**（如架构图、性能对比图）？

## 七、参考资料

- [Karpathy LLM Wiki](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f)
- [NVIDIA Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/)
- [tcgen05 for dummies (Gau Nernst)](https://gau-nernst.github.io/tcgen05/)
- [Colfax CUTLASS Blackwell Tutorial](https://research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/)
- [CUDA 13.0 Blog](https://developer.nvidia.com/blog/whats-new-and-important-in-cuda-toolkit-13-0/)
- [CUDA 13.1 Blog](https://developer.nvidia.com/blog/nvidia-cuda-13-1-powers-next-gen-gpu-programming-with-nvidia-cuda-tile-and-performance-gains/)
- [GPU Mode NVFP4 Hackathon](https://forums.developer.nvidia.com/t/join-us-for-the-blackwell-nvfp4-kernel-hackathon-with-nvidia-and-gpu-mode/350092)
- [FlashInfer MLSys 2026 Contest](https://mlsys26.flashinfer.ai/)
- [Blackwell NVFP4 Hackathon Journey (Yue Zhang)](https://yue-zhang-2025.github.io/2025/12/02/blackwell-nvfp4-kernel-hackathon-journey.html)
- [Modular: Matrix Multiplication on Blackwell](https://www.modular.com/blog/matrix-multiplication-on-nvidias-blackwell-part-1-introduction)

--- Original Design Draft End ---
