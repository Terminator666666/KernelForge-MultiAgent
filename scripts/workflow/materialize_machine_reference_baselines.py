#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为当前 5070/WSL 环境生成“官方参考实现”基线。

设计目标：
1. 不再依赖 FlashInfer 的专用推理后端（xqa / trtllm-gen / cute-dsl 等）；
2. 直接使用官方 definition 中的 reference 代码，或基于官方 reference 结构
   补齐缺失 target 的 reference 定义；
3. 为三个主线 family 生成可被现有 workflow 直接消费的 baseline solution；
4. 对缺失的 definition / workload 文本索引进行补齐；未真实执行的结果禁止写入 traces/。

注意：
- 这里的“无后端”指不走 FlashInfer 专用算子后端，仍然会通过 PyTorch 基础算子
  在 CUDA 上执行 matmul / softmax / pointwise kernel。
- 本脚本只写 definition / workload / solution 文本索引，不下载任何新数据。
- 对未实际 benchmark 产生的结果，禁止伪造 traces/ 记录，否则会污染 TraceSet。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from safetensors import safe_open


PROJECT_ROOT = Path("/mnt/d/agent/KernelForge-MultiAgent").resolve()
DATASET_ROOT = Path("/mnt/d/Agent/flashinfer-trace").resolve()


@dataclass(frozen=True)
class BaselineSpec:
    family: str
    definition: str
    baseline_dataset_group: str
    baseline_solution: str
    baseline_source_kind: str
    comparison_denominator: str


BASELINE_SPECS = {
    "dsa_sparse_attention": BaselineSpec(
        family="dsa_sparse_attention",
        definition="dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64",
        baseline_dataset_group="dsa",
        baseline_solution="official_reference_dsa_sparse_attention_v1",
        baseline_source_kind="official_baseline",
        comparison_denominator="official_baseline",
    ),
    "gdn_prefill": BaselineSpec(
        family="gdn_prefill",
        definition="gdn_prefill_qk4_v8_d128_k_last",
        baseline_dataset_group="gdn",
        baseline_solution="official_reference_gdn_prefill_v1",
        baseline_source_kind="official_baseline",
        comparison_denominator="official_baseline",
    ),
    "dsa_topk_indexer": BaselineSpec(
        family="dsa_topk_indexer",
        definition="dsa_topk_indexer_fp8_h64_d128_topk2048_ps64",
        baseline_dataset_group="dsa",
        baseline_solution="official_reference_dsa_topk_indexer_v1",
        baseline_source_kind="official_baseline",
        comparison_denominator="official_baseline",
    ),
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def remove_file_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()


def extract_uuid_from_filename(path: Path, definition: str) -> str:
    stem = path.stem
    prefix = f"{definition}_"
    if not stem.startswith(prefix):
        raise ValueError(f"无法从文件名提取 uuid: {path}")
    return stem[len(prefix):]


def load_safetensor_meta(path: Path) -> dict[str, tuple[tuple[int, ...], str]]:
    meta: dict[str, tuple[tuple[int, ...], str]] = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            meta[key] = (tuple(tensor.shape), str(tensor.dtype).replace("torch.", ""))
    return meta


def build_gdn_decode_qk4_definition() -> dict[str, Any]:
    scale = 1.0 / math.sqrt(128.0)
    reference = f"""import math
import torch
import torch.nn.functional as F


def matmul(a: torch.Tensor, b: torch.Tensor):
    \"\"\"Float32 matmul for numerical stability.\"\"\"
    return a.float() @ b.float()


@torch.no_grad()
def run(q, k, v, state, A_log, a, dt_bias, b, scale):
    \"\"\"
    Gated Delta Net decode reference implementation (k-last layout).
    State layout: [B, H, V, K]，其中 H=8, K=V=128。
    \"\"\"
    B, T, num_q_heads, K = q.shape
    _, _, num_k_heads, _ = k.shape
    _, _, num_v_heads, V = v.shape
    device = q.device

    assert T == 1
    assert num_q_heads == 4
    assert num_k_heads == 4
    assert num_v_heads == 8
    assert K == 128 and V == 128

    if scale is None or scale == 0.0:
        scale = {scale}

    x = a.float() + dt_bias.float()
    g = torch.exp(-torch.exp(A_log.float()) * F.softplus(x))
    beta = torch.sigmoid(b.float())

    q_f32 = q.squeeze(1).float()
    k_f32 = k.squeeze(1).float()
    v_f32 = v.squeeze(1).float()
    g_f32 = g.squeeze(1).float()
    beta_f32 = beta.squeeze(1).float()

    if state is not None:
        state_f32 = state.float()
    else:
        state_f32 = torch.zeros(B, num_v_heads, V, K, dtype=torch.float32, device=device)

    q_exp = q_f32.repeat_interleave(num_v_heads // num_q_heads, dim=1)
    k_exp = k_f32.repeat_interleave(num_v_heads // num_k_heads, dim=1)

    new_state = torch.zeros_like(state_f32)
    output = torch.zeros(B, num_v_heads, V, dtype=torch.float32, device=device)

    for b_idx in range(B):
        for h_idx in range(num_v_heads):
            q_h = q_exp[b_idx, h_idx]
            k_h = k_exp[b_idx, h_idx]
            v_h = v_f32[b_idx, h_idx]
            h_state = state_f32[b_idx, h_idx].clone().transpose(-1, -2)
            g_val = g_f32[b_idx, h_idx]
            beta_val = beta_f32[b_idx, h_idx]

            old_state = g_val * h_state
            old_v = k_h @ old_state
            new_v = beta_val * v_h + (1 - beta_val) * old_v
            state_remove = k_h.unsqueeze(1) @ old_v.unsqueeze(0)
            state_update = k_h.unsqueeze(1) @ new_v.unsqueeze(0)
            h_state = old_state - state_remove + state_update

            output[b_idx, h_idx] = scale * (q_h @ h_state)
            new_state[b_idx, h_idx] = h_state.transpose(-1, -2)

    output = output.unsqueeze(1).to(torch.bfloat16)
    return output, new_state
"""
    return {
        "name": "gdn_decode_qk4_v8_d128_k_last",
        "description": "Gated Delta Net decode with GVA configuration and k-last state layout. Single-token generation with recurrent state update. Captured from Qwen3 Next linear attention layers (q=4, v=8).",
        "op_type": "gdn_decode",
        "tags": [
            "stage:decode",
            "status:machine-baseline",
            "model:qwen3-next",
            "layout:k-last",
        ],
        "axes": {
            "batch_size": {
                "type": "var",
                "description": "并发 decode 的序列条数。"
            },
            "seq_len": {
                "type": "const",
                "value": 1,
                "description": "decode 单步长度固定为 1。"
            },
            "num_q_heads": {"type": "const", "value": 4},
            "num_k_heads": {"type": "const", "value": 4},
            "num_v_heads": {"type": "const", "value": 8},
            "head_size": {"type": "const", "value": 128},
        },
        "constraints": [
            "num_v_heads >= num_q_heads",
            "num_v_heads % num_q_heads == 0",
            "num_k_heads == num_q_heads",
        ],
        "inputs": {
            "q": {"shape": ["batch_size", "seq_len", "num_q_heads", "head_size"], "dtype": "bfloat16"},
            "k": {"shape": ["batch_size", "seq_len", "num_k_heads", "head_size"], "dtype": "bfloat16"},
            "v": {"shape": ["batch_size", "seq_len", "num_v_heads", "head_size"], "dtype": "bfloat16"},
            "state": {
                "shape": ["batch_size", "num_v_heads", "head_size", "head_size"],
                "dtype": "float32",
                "optional": True,
            },
            "A_log": {"shape": ["num_v_heads"], "dtype": "float32"},
            "a": {"shape": ["batch_size", "seq_len", "num_v_heads"], "dtype": "bfloat16"},
            "dt_bias": {"shape": ["num_v_heads"], "dtype": "float32"},
            "b": {"shape": ["batch_size", "seq_len", "num_v_heads"], "dtype": "bfloat16"},
            "scale": {"shape": None, "dtype": "float32"},
        },
        "outputs": {
            "output": {"shape": ["batch_size", "seq_len", "num_v_heads", "head_size"], "dtype": "bfloat16"},
            "new_state": {"shape": ["batch_size", "num_v_heads", "head_size", "head_size"], "dtype": "float32"},
        },
        "reference": reference,
    }


def build_gdn_prefill_qk4_definition() -> dict[str, Any]:
    scale = 1.0 / math.sqrt(128.0)
    reference = f"""import math
import torch
import torch.nn.functional as F


def matmul(a: torch.Tensor, b: torch.Tensor):
    \"\"\"Float32 matmul for numerical stability.\"\"\"
    return a.float() @ b.float()


@torch.no_grad()
def run(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
    \"\"\"
    Gated Delta Net prefill reference implementation (k-last layout).
    State layout: [N, H, V, K]，其中 H=8, K=V=128。
    \"\"\"
    total_seq_len, num_q_heads, head_size = q.shape
    num_v_heads = v.shape[1]
    num_k_heads = k.shape[1]
    num_sab_heads = max(num_q_heads, num_v_heads)
    num_seqs = cu_seqlens.size(0) - 1
    device = q.device

    assert num_q_heads == 4
    assert num_k_heads == 4
    assert num_v_heads == 8
    assert head_size == 128

    if scale is None or scale == 0.0:
        scale = {scale}

    x = a.float() + dt_bias.float()
    g = torch.exp(-torch.exp(A_log.float()) * F.softplus(x))
    beta = torch.sigmoid(b.float())

    q_exp = q.repeat_interleave(num_v_heads // num_q_heads, dim=1)
    k_exp = k.repeat_interleave(num_v_heads // num_k_heads, dim=1)

    output = torch.zeros((total_seq_len, num_sab_heads, head_size), dtype=torch.bfloat16, device=device)
    new_state = torch.zeros((num_seqs, num_sab_heads, head_size, head_size), dtype=torch.float32, device=device)

    for seq_idx in range(num_seqs):
        seq_start = int(cu_seqlens[seq_idx].item())
        seq_end = int(cu_seqlens[seq_idx + 1].item())
        seq_len = seq_end - seq_start

        if seq_len <= 0:
            continue

        if state is not None:
            state_hkv = state[seq_idx].clone().float().transpose(-1, -2)
        else:
            state_hkv = torch.zeros((num_sab_heads, head_size, head_size), dtype=torch.float32, device=device)

        for i in range(seq_len):
            t = seq_start + i
            q_h1k = q_exp[t].unsqueeze(1).float()
            k_h1k = k_exp[t].unsqueeze(1).float()
            v_h1v = v[t].unsqueeze(1).float()
            g_h11 = g[t].unsqueeze(1).unsqueeze(2)
            beta_h11 = beta[t].unsqueeze(1).unsqueeze(2)

            old_state_hkv = g_h11 * state_hkv
            old_v_h1v = matmul(k_h1k, old_state_hkv)
            new_v_h1v = beta_h11 * v_h1v + (1 - beta_h11) * old_v_h1v
            state_remove = torch.einsum('hkl,hlv->hkv', k_h1k.transpose(-1, -2), old_v_h1v)
            state_update = torch.einsum('hkl,hlv->hkv', k_h1k.transpose(-1, -2), new_v_h1v)
            state_hkv = old_state_hkv - state_remove + state_update

            o_h1v = scale * matmul(q_h1k, state_hkv)
            output[t] = o_h1v.squeeze(1).to(torch.bfloat16)

        new_state[seq_idx] = state_hkv.transpose(-1, -2)

    return output, new_state
"""
    return {
        "name": "gdn_prefill_qk4_v8_d128_k_last",
        "description": "Gated Delta Net prefill with GVA configuration and k-last state layout. Captured from Qwen3 Next linear attention layers (q=4, v=8).",
        "op_type": "gdn_prefill",
        "tags": [
            "stage:prefill",
            "status:machine-baseline",
            "model:qwen3-next",
            "layout:k-last",
        ],
        "axes": {
            "total_seq_len": {"type": "var"},
            "num_seqs": {"type": "var"},
            "num_q_heads": {"type": "const", "value": 4},
            "num_k_heads": {"type": "const", "value": 4},
            "num_v_heads": {"type": "const", "value": 8},
            "head_size": {"type": "const", "value": 128},
            "len_cu_seqlens": {
                "type": "var",
                "description": "cu_seqlens 数组长度（num_seqs + 1）。"
            },
        },
        "constraints": [
            "len_cu_seqlens == num_seqs + 1",
            "total_seq_len == cu_seqlens[-1].item()",
        ],
        "inputs": {
            "q": {"shape": ["total_seq_len", "num_q_heads", "head_size"], "dtype": "bfloat16"},
            "k": {"shape": ["total_seq_len", "num_k_heads", "head_size"], "dtype": "bfloat16"},
            "v": {"shape": ["total_seq_len", "num_v_heads", "head_size"], "dtype": "bfloat16"},
            "state": {"shape": ["num_seqs", "num_v_heads", "head_size", "head_size"], "dtype": "float32", "optional": True},
            "A_log": {"shape": ["num_v_heads"], "dtype": "float32"},
            "a": {"shape": ["total_seq_len", "num_v_heads"], "dtype": "bfloat16"},
            "dt_bias": {"shape": ["num_v_heads"], "dtype": "float32"},
            "b": {"shape": ["total_seq_len", "num_v_heads"], "dtype": "bfloat16"},
            "cu_seqlens": {"shape": ["len_cu_seqlens"], "dtype": "int64"},
            "scale": {"shape": None, "dtype": "float32"},
        },
        "outputs": {
            "output": {"shape": ["total_seq_len", "num_v_heads", "head_size"], "dtype": "bfloat16"},
            "new_state": {"shape": ["num_seqs", "num_v_heads", "head_size", "head_size"], "dtype": "float32"},
        },
        "reference": reference,
    }


def build_dsa_topk_indexer_topk2048_definition() -> dict[str, Any]:
    reference = """import torch


def dequant_fp8_kv_cache(k_index_cache_fp8):
    \"\"\"从 deep_gemm 打包格式反量化 FP8 KV cache。\"\"\"
    k_index_cache_fp8 = k_index_cache_fp8.view(torch.uint8)
    num_pages, page_size, num_heads, head_dim_sf = k_index_cache_fp8.shape
    head_dim = head_dim_sf - 4

    kv_flat = k_index_cache_fp8.view(num_pages, page_size * head_dim_sf)
    fp8_bytes = kv_flat[:, :page_size * head_dim].contiguous()
    fp8_tensor = fp8_bytes.view(num_pages, page_size, head_dim).view(torch.float8_e4m3fn)
    fp8_float = fp8_tensor.to(torch.float32)

    scale_bytes = kv_flat[:, page_size * head_dim:].contiguous()
    scale = scale_bytes.view(num_pages, page_size, 4).view(torch.float32)

    return fp8_float * scale


@torch.no_grad()
def run(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table):
    batch_size, num_index_heads, index_head_dim = q_index_fp8.shape
    num_pages, page_size, _, _ = k_index_cache_fp8.shape
    topk = 2048

    assert num_index_heads == 64
    assert index_head_dim == 128
    assert page_size == 64

    device = q_index_fp8.device

    q = q_index_fp8.to(torch.float32)
    k_all = dequant_fp8_kv_cache(k_index_cache_fp8)

    topk_indices = torch.full((batch_size, topk), -1, dtype=torch.int32, device=device)
    max_num_pages = block_table.shape[1]

    for b_idx in range(batch_size):
        seq_len = int(seq_lens[b_idx].item())
        if seq_len == 0:
            continue

        num_pages_for_seq = (seq_len + page_size - 1) // page_size
        page_indices = block_table[b_idx, :num_pages_for_seq].to(torch.long)

        k_paged = k_all[page_indices]
        k = k_paged.reshape(-1, index_head_dim)[:seq_len]
        q_b = q[b_idx]

        scores = q_b @ k.T
        scores_relu = torch.relu(scores)
        w = weights[b_idx]
        weighted_scores = scores_relu * w[:, None]
        final_scores = weighted_scores.sum(dim=0)

        actual_topk = min(topk, seq_len)
        _, topk_idx = torch.topk(final_scores, actual_topk)

        page_idx_per_token = topk_idx // page_size
        offset_per_token = topk_idx % page_size
        global_page_idx = page_indices[page_idx_per_token]
        topk_tokens = global_page_idx * page_size + offset_per_token

        topk_indices[b_idx, :actual_topk] = topk_tokens.to(torch.int32)

    return (topk_indices,)
"""
    return {
        "name": "dsa_topk_indexer_fp8_h64_d128_topk2048_ps64",
        "description": "Native Sparse Attention (DSA) TopK indexer with FP8 quantization for DeepSeek-V3. Computes sparse attention scores using ReLU activation and learned weights, then selects top-K KV cache indices. Page size 64, topk=2048.",
        "op_type": "dsa_topk_indexer",
        "tags": [
            "stage:indexer",
            "status:machine-baseline",
            "model:deepseek-v3",
            "sparse:topk",
            "quant:fp8",
        ],
        "axes": {
            "batch_size": {"type": "var"},
            "num_index_heads": {"type": "const", "value": 64},
            "index_head_dim": {"type": "const", "value": 128},
            "page_size": {"type": "const", "value": 64},
            "topk": {"type": "const", "value": 2048},
            "max_num_pages": {"type": "var"},
            "num_pages": {"type": "var"},
            "kv_cache_num_heads": {"type": "const", "value": 1},
            "head_dim_with_scale": {"type": "const", "value": 132},
        },
        "constraints": [
            "topk <= max_num_pages * page_size"
        ],
        "inputs": {
            "q_index_fp8": {"shape": ["batch_size", "num_index_heads", "index_head_dim"], "dtype": "float8_e4m3fn"},
            "k_index_cache_fp8": {
                "shape": ["num_pages", "page_size", "kv_cache_num_heads", "head_dim_with_scale"],
                "dtype": "int8",
            },
            "weights": {"shape": ["batch_size", "num_index_heads"], "dtype": "float32"},
            "seq_lens": {"shape": ["batch_size"], "dtype": "int32"},
            "block_table": {"shape": ["batch_size", "max_num_pages"], "dtype": "int32"},
        },
        "outputs": {
            "topk_indices": {"shape": ["batch_size", "topk"], "dtype": "int32"},
        },
        "reference": reference,
    }


def build_gdn_decode_qk4_workloads() -> list[dict[str, Any]]:
    definition = "gdn_decode_qk4_v8_d128_k_last"
    blob_dir = DATASET_ROOT / "blob/workloads/gdn" / definition
    rows: list[dict[str, Any]] = []
    for path in sorted(blob_dir.glob("*.safetensors")):
        meta = load_safetensor_meta(path)
        q_shape, _ = meta["q"]
        k_shape, _ = meta["k"]
        v_shape, _ = meta["v"]
        state_shape, _ = meta["state"]
        uuid = extract_uuid_from_filename(path, definition)
        scale = 1.0 / math.sqrt(128.0)
        rows.append(
            {
                "definition": definition,
                "solution": None,
                "workload": {
                    "uuid": uuid,
                    "axes": {
                        "batch_size": q_shape[0],
                    },
                    "inputs": {
                        "q": {"type": "safetensors", "path": f"./blob/workloads/gdn/{definition}/{path.name}", "tensor_key": "q"},
                        "k": {"type": "safetensors", "path": f"./blob/workloads/gdn/{definition}/{path.name}", "tensor_key": "k"},
                        "v": {"type": "safetensors", "path": f"./blob/workloads/gdn/{definition}/{path.name}", "tensor_key": "v"},
                        "state": {"type": "safetensors", "path": f"./blob/workloads/gdn/{definition}/{path.name}", "tensor_key": "state"},
                        "A_log": {"type": "safetensors", "path": f"./blob/workloads/gdn/{definition}/{path.name}", "tensor_key": "A_log"},
                        "a": {"type": "safetensors", "path": f"./blob/workloads/gdn/{definition}/{path.name}", "tensor_key": "a"},
                        "dt_bias": {"type": "safetensors", "path": f"./blob/workloads/gdn/{definition}/{path.name}", "tensor_key": "dt_bias"},
                        "b": {"type": "safetensors", "path": f"./blob/workloads/gdn/{definition}/{path.name}", "tensor_key": "b"},
                        "scale": {"type": "scalar", "value": scale},
                    },
                },
                "evaluation": None,
            }
        )
        assert q_shape[2] == 4 and k_shape[2] == 4 and v_shape[2] == 8 and state_shape[1] == 8
    return rows


def build_gdn_prefill_qk4_workloads() -> list[dict[str, Any]]:
    definition = "gdn_prefill_qk4_v8_d128_k_last"
    blob_dir = DATASET_ROOT / "blob/workloads/gdn" / definition
    rows: list[dict[str, Any]] = []
    for path in sorted(blob_dir.glob("*.safetensors")):
        meta = load_safetensor_meta(path)
        q_shape, _ = meta["q"]
        state_shape, _ = meta["state"]
        cu_shape, _ = meta["cu_seqlens"]
        uuid = extract_uuid_from_filename(path, definition)
        scale = 1.0 / math.sqrt(128.0)
        rows.append(
            {
                "definition": definition,
                "solution": None,
                "workload": {
                    "uuid": uuid,
                    "axes": {
                        "total_seq_len": q_shape[0],
                        "num_seqs": state_shape[0],
                        "len_cu_seqlens": cu_shape[0],
                    },
                    "inputs": {
                        "q": {"type": "safetensors", "path": f"./blob/workloads/gdn/{definition}/{path.name}", "tensor_key": "q"},
                        "k": {"type": "safetensors", "path": f"./blob/workloads/gdn/{definition}/{path.name}", "tensor_key": "k"},
                        "v": {"type": "safetensors", "path": f"./blob/workloads/gdn/{definition}/{path.name}", "tensor_key": "v"},
                        "state": {"type": "safetensors", "path": f"./blob/workloads/gdn/{definition}/{path.name}", "tensor_key": "state"},
                        "A_log": {"type": "safetensors", "path": f"./blob/workloads/gdn/{definition}/{path.name}", "tensor_key": "A_log"},
                        "a": {"type": "safetensors", "path": f"./blob/workloads/gdn/{definition}/{path.name}", "tensor_key": "a"},
                        "dt_bias": {"type": "safetensors", "path": f"./blob/workloads/gdn/{definition}/{path.name}", "tensor_key": "dt_bias"},
                        "b": {"type": "safetensors", "path": f"./blob/workloads/gdn/{definition}/{path.name}", "tensor_key": "b"},
                        "cu_seqlens": {"type": "safetensors", "path": f"./blob/workloads/gdn/{definition}/{path.name}", "tensor_key": "cu_seqlens"},
                        "scale": {"type": "scalar", "value": scale},
                    },
                },
                "evaluation": None,
            }
        )
    return rows


def build_dsa_topk_indexer_topk2048_workloads() -> list[dict[str, Any]]:
    definition = "dsa_topk_indexer_fp8_h64_d128_topk2048_ps64"
    blob_dir = DATASET_ROOT / "blob/workloads/dsa_paged" / definition
    rows: list[dict[str, Any]] = []
    for path in sorted(blob_dir.glob("*.safetensors")):
        meta = load_safetensor_meta(path)
        block_table_shape, _ = meta["block_table"]
        seq_lens_shape, _ = meta["seq_lens"]
        with safe_open(path, framework="pt", device="cpu") as f:
            block_table = f.get_tensor("block_table")
        batch_size = block_table_shape[0]
        max_num_pages = block_table_shape[1]
        num_pages = int(block_table.max().item()) + 1 if block_table.numel() else 0
        uuid = extract_uuid_from_filename(path, definition)
        rows.append(
            {
                "definition": definition,
                "solution": None,
                "workload": {
                    "uuid": uuid,
                    "axes": {
                        "batch_size": batch_size,
                        "max_num_pages": max_num_pages,
                        "num_pages": num_pages,
                    },
                    "inputs": {
                        "q_index_fp8": {"type": "random"},
                        "k_index_cache_fp8": {"type": "random"},
                        "weights": {"type": "random"},
                        "seq_lens": {"type": "safetensors", "path": f"./blob/workloads/dsa_paged/{definition}/{path.name}", "tensor_key": "seq_lens"},
                        "block_table": {"type": "safetensors", "path": f"./blob/workloads/dsa_paged/{definition}/{path.name}", "tensor_key": "block_table"},
                    },
                },
                "evaluation": None,
            }
        )
        assert seq_lens_shape[0] == batch_size
    return rows


def make_solution_payload(
    definition: dict[str, Any],
    solution_name: str,
    description: str,
    target_hardware: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "name": solution_name,
        "definition": definition["name"],
        "author": "baseline",
        "spec": {
            "language": "python",
            "target_hardware": target_hardware or ["NVIDIA RTX 5070", "WSL2"],
            "entry_point": "main.py::run",
            "dependencies": ["torch"],
            "destination_passing_style": False,
        },
        "description": description,
        "sources": [
            {
                "path": "main.py",
                "content": definition["reference"],
            }
        ],
    }


def upsert_reference_baseline(family: str, spec: BaselineSpec) -> None:
    baseline_path = PROJECT_ROOT / "reference" / family / "baseline.json"
    payload = read_json(baseline_path)
    payload["baseline_solution"] = spec.baseline_solution
    payload["baseline_source_kind"] = spec.baseline_source_kind
    payload["comparison_denominator"] = spec.comparison_denominator
    for target in payload.get("supported_targets", []):
        if target.get("definition") == spec.definition:
            target["baseline_solution"] = spec.baseline_solution
            target["baseline_source_kind"] = spec.baseline_source_kind
            if "compare_against" in target:
                target["compare_against"] = spec.comparison_denominator
    write_json(baseline_path, payload)

    solutions_jsonl = PROJECT_ROOT / "reference" / family / "solutions.jsonl"
    rows = []
    if solutions_jsonl.exists():
        with solutions_jsonl.open("r", encoding="utf-8") as f:
            for raw in f:
                if raw.strip():
                    rows.append(json.loads(raw))
    found = False
    for row in rows:
        if row.get("id") == "official-baseline-v0":
            row["solution_name"] = spec.baseline_solution
            row["definition"] = spec.definition
            row["description"] = "官方 reference 基线；来源于 definition reference / 官方参考实现，适配当前本机环境。"
            found = True
    if not found:
        rows.insert(
            0,
            {
                "id": "official-baseline-v0",
                "parent": None,
                "solution_name": spec.baseline_solution,
                "definition": spec.definition,
                "decision": "BASELINE",
                "description": "官方 reference 基线；来源于 definition reference / 官方参考实现，适配当前本机环境。",
            },
        )
    write_jsonl(solutions_jsonl, rows)


def update_operator_families(specs: dict[str, BaselineSpec]) -> None:
    path = PROJECT_ROOT / "configs/operator_families.json"
    payload = read_json(path)
    family_cfg = payload["families"]

    family_cfg["dsa_sparse_attention"]["default_baseline_solution"] = specs["dsa_sparse_attention"].baseline_solution
    family_cfg["dsa_sparse_attention"]["targets"][0]["baseline_solution"] = specs["dsa_sparse_attention"].baseline_solution

    family_cfg["gdn_prefill"]["default_baseline_solution"] = specs["gdn_prefill"].baseline_solution
    family_cfg["gdn_prefill"]["targets"][0]["baseline_solution"] = specs["gdn_prefill"].baseline_solution

    family_cfg["dsa_topk_indexer"]["default_baseline_solution"] = specs["dsa_topk_indexer"].baseline_solution
    family_cfg["dsa_topk_indexer"]["targets"][0]["baseline_solution"] = specs["dsa_topk_indexer"].baseline_solution

    write_json(path, payload)


def ensure_definition(path: Path, payload: dict[str, Any]) -> None:
    write_json(path, payload)


def ensure_solution(spec: BaselineSpec, definition_payload: dict[str, Any]) -> None:
    solution_path = (
        DATASET_ROOT
        / "solutions"
        / "baseline"
        / spec.baseline_dataset_group
        / spec.definition
        / f"{spec.baseline_solution}.json"
    )
    description = f"官方参考实现 baseline（不走 FlashInfer 专用推理后端），定义为 {spec.definition}。"
    payload = make_solution_payload(definition_payload, spec.baseline_solution, description)
    write_json(solution_path, payload)


def main() -> int:
    # 1) 补齐缺失 definition
    gdn_prefill_def = build_gdn_prefill_qk4_definition()
    dsa_topk_def = build_dsa_topk_indexer_topk2048_definition()

    ensure_definition(DATASET_ROOT / "definitions/gdn/gdn_prefill_qk4_v8_d128_k_last.json", gdn_prefill_def)
    ensure_definition(DATASET_ROOT / "definitions/dsa_paged/dsa_topk_indexer_fp8_h64_d128_topk2048_ps64.json", dsa_topk_def)

    # 2) 生成 workload 索引；真实 execution trace 必须由后续 benchmark 产生
    gdn_prefill_rows = build_gdn_prefill_qk4_workloads()
    dsa_topk_rows = build_dsa_topk_indexer_topk2048_workloads()

    write_jsonl(DATASET_ROOT / "workloads/gdn/gdn_prefill_qk4_v8_d128_k_last.jsonl", gdn_prefill_rows)
    write_jsonl(DATASET_ROOT / "workloads/dsa_paged/dsa_topk_indexer_fp8_h64_d128_topk2048_ps64.jsonl", dsa_topk_rows)
    remove_file_if_exists(DATASET_ROOT / "traces/gdn/gdn_prefill_qk4_v8_d128_k_last.jsonl")
    remove_file_if_exists(DATASET_ROOT / "traces/dsa_paged/dsa_topk_indexer_fp8_h64_d128_topk2048_ps64.jsonl")

    # 3) 生成三个 machine baseline solution
    definition_map = {
        "dsa_sparse_attention": read_json(DATASET_ROOT / "definitions/dsa_paged/dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.json"),
        "gdn_prefill": gdn_prefill_def,
        "dsa_topk_indexer": dsa_topk_def,
    }
    for family, spec in BASELINE_SPECS.items():
        ensure_solution(spec, definition_map[family])

    # 4) 回写 family baseline 配置
    update_operator_families(BASELINE_SPECS)
    for family, spec in BASELINE_SPECS.items():
        upsert_reference_baseline(family, spec)

    print("machine reference baselines 已生成")
    for family, spec in BASELINE_SPECS.items():
        print(f"- {family}: {spec.baseline_solution}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
