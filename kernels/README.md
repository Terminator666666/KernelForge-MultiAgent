# Kernels

This directory contains only the kernel source files kept for the current mainline operator families.

## Layout

- `operators/dsa_paged`: shared source bucket for the DSA mainline families
  (`dsa_sparse_attention` and `dsa_topk_indexer`).
- `operators/gdn`: source bucket for the `gdn_prefill` mainline family.
- `generated/all_operators.cu`: convenience include file for the current mainline-only source set.

Non-mainline operator source directories were intentionally removed from `kernels/operators/`
to keep the published tree aligned with the repository's active optimization scope.
Compiled objects, `.so` files, NCU reports, logs, and executable binaries are intentionally excluded.
