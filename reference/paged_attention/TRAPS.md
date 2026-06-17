# paged_attention Traps

## 当前状态

尚未进入真实 round，暂无已验证陷阱。

## 预置检查项

1. GQA 与 MLA 子目标不能混用 baseline 分母
2. 切换子目标时必须同步更新 `definition`、`op_type`、`baseline_solution`、`baseline_dataset_group`
3. 不允许拿 `gqa` 的 expert baseline 去评 `mla`
4. 所有结论必须基于真实 NCU 与真实 benchmark
