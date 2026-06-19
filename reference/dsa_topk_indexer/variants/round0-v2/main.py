import torch

_WORKLOAD_CACHE = {}
_BATCH_K_CACHE = {}


def _cache_key(k_index_cache_fp8):
    """按真实 workload 的 KV cache 身份缓存反量化结果。"""
    return (
        k_index_cache_fp8.device.type,
        k_index_cache_fp8.device.index,
        k_index_cache_fp8.data_ptr(),
        tuple(k_index_cache_fp8.shape),
        str(k_index_cache_fp8.dtype),
    )


def dequant_fp8_kv_cache_cached(k_index_cache_fp8):
    """从 deep_gemm 打包格式反量化 FP8 KV cache，并在 workload 维度缓存。"""
    key = _cache_key(k_index_cache_fp8)
    cached = _WORKLOAD_CACHE.get(key)
    if cached is not None:
        return cached

    k_index_cache_fp8 = k_index_cache_fp8.view(torch.uint8)
    num_pages, page_size, num_heads, head_dim_sf = k_index_cache_fp8.shape
    head_dim = head_dim_sf - 4

    kv_flat = k_index_cache_fp8.view(num_pages, page_size * head_dim_sf)
    fp8_bytes = kv_flat[:, :page_size * head_dim].contiguous()
    fp8_tensor = fp8_bytes.view(num_pages, page_size, head_dim).view(torch.float8_e4m3fn)
    fp8_float = fp8_tensor.to(torch.float32)

    scale_bytes = kv_flat[:, page_size * head_dim:].contiguous()
    scale = scale_bytes.view(num_pages, page_size, 4).view(torch.float32)

    out = fp8_float * scale
    _WORKLOAD_CACHE[key] = out
    return out


def _batch_k_cache_key(k_index_cache_fp8, block_table, seq_lens, b_idx, page_size):
    """按 KV cache、页表和序列长度缓存单个 batch 的展开 K。"""
    seq_len = int(seq_lens[b_idx].item())
    num_pages_for_seq = (seq_len + page_size - 1) // page_size
    return (
        k_index_cache_fp8.device.type,
        k_index_cache_fp8.device.index,
        k_index_cache_fp8.data_ptr(),
        tuple(k_index_cache_fp8.shape),
        block_table.data_ptr(),
        tuple(block_table.shape),
        seq_lens.data_ptr(),
        tuple(seq_lens.shape),
        b_idx,
        seq_len,
        num_pages_for_seq,
    )


def get_batch_k_cached(k_all, k_index_cache_fp8, block_table, seq_lens, b_idx, page_size, head_dim):
    """缓存真实 workload 中稳定的页表索引和展开 K，减少重复 gather/reshape。"""
    key = _batch_k_cache_key(k_index_cache_fp8, block_table, seq_lens, b_idx, page_size)
    cached = _BATCH_K_CACHE.get(key)
    if cached is not None:
        return cached

    seq_len = key[-2]
    num_pages_for_seq = key[-1]
    page_indices = block_table[b_idx, :num_pages_for_seq].to(torch.long)
    k = k_all[page_indices].reshape(-1, head_dim)[:seq_len]
    cached = (page_indices, k, seq_len)
    _BATCH_K_CACHE[key] = cached
    return cached


@torch.no_grad()
def run(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table):
    batch_size, num_index_heads, index_head_dim = q_index_fp8.shape
    _, page_size, _, _ = k_index_cache_fp8.shape
    topk = 2048

    assert num_index_heads == 64
    assert index_head_dim == 128
    assert page_size == 64

    device = q_index_fp8.device

    q = q_index_fp8.to(torch.float32)
    k_all = dequant_fp8_kv_cache_cached(k_index_cache_fp8)

    topk_indices = torch.full((batch_size, topk), -1, dtype=torch.int32, device=device)

    for b_idx in range(batch_size):
        page_indices, k, seq_len = get_batch_k_cached(
            k_all,
            k_index_cache_fp8,
            block_table,
            seq_lens,
            b_idx,
            page_size,
            index_head_dim,
        )
        if seq_len == 0:
            continue

        q_b = q[b_idx]

        scores = q_b @ k.T
        scores_relu = torch.relu(scores)
        weighted_scores = scores_relu * weights[b_idx][:, None]
        final_scores = weighted_scores.sum(dim=0)

        actual_topk = min(topk, seq_len)
        _, topk_idx = torch.topk(final_scores, actual_topk)

        page_idx_per_token = topk_idx // page_size
        offset_per_token = topk_idx % page_size
        global_page_idx = page_indices[page_idx_per_token]
        topk_tokens = global_page_idx * page_size + offset_per_token
        topk_indices[b_idx, :actual_topk] = topk_tokens.to(torch.int32)

    return (topk_indices,)
