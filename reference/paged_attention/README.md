# paged_attention 归档

当前主线 family：`paged_attention`

## 默认目标

- 当前默认 `definition`: `gqa_paged_decode_h32_kv8_d128_ps1`
- 当前默认 `baseline_solution`: `flashinfer_wrapper_a9588f`
- `comparison_denominator`: `expert_baseline`
- `baseline_dataset_group`: `gqa_paged`

## 支持的子目标

1. `gqa_paged_decode_h32_kv8_d128_ps1`
   - `baseline_solution`: `flashinfer_wrapper_a9588f`
   - `baseline_dataset_group`: `gqa_paged`
2. `mla_paged_decode_h16_ckv512_kpe64_ps1`
   - `baseline_solution`: `flashinfer_wrapper_03f7b0`
   - `baseline_dataset_group`: `mla_paged`
3. `mla_paged_prefill_causal_h16_ckv512_kpe64_ps1`
   - `baseline_solution`: `flashinfer_wrapper_ea3787`
   - `baseline_dataset_group`: `mla_paged`

## baseline 来源说明

上面三个 baseline 都来自数据集 `flashinfer-trace/solutions/baseline/...`，当前看到的是官方/专家分母方案文件。它们在数据集里都是 `python wrapper` 形态，`entry_point` 都是 `main.py::run`。

这意味着：
- 是官方 baseline / expert baseline 方案，能作为正确分母
- 但不等于“你现在已经拿到了最终底层 CUDA kernel 文件”
- 若后续要改底层 kernel，需要继续沿 wrapper 追到底层 FlashInfer 实现

## 当前状态

- 仅完成 baseline 归档初始化
- 当前默认子目标为 `gqa_paged_decode_h32_kv8_d128_ps1`
- 切换到 MLA decode / MLA prefill 时，必须同步修改 `baseline.json`

## 归档文件

- `baseline.json`
- `TRAPS.md`
- `solutions.jsonl`
- `variants/`
