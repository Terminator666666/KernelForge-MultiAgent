# 鑷姩鎵归噺浼樺寲浣跨敤鎸囧崡

## 馃殌 蹇€熷紑濮?
### 1. 閰嶇疆 API Key

缂栬緫 `config/api_config.json`锛?
```json
{
  "api_key": "REPLACE_ME_LOCALLY",
  "base_url": "https://api.siliconflow.cn/v1",
  "model": "deepseek-ai/DeepSeek-V3"
}
```

### 2. 鍚姩鎵归噺浼樺寲

```bash
cd KernelForge-MultiAgent

# 鏂瑰紡 1: 浣跨敤鍚姩鍣紙鎺ㄨ崘锛?python scripts/start_batch_optimization.py

# 鏂瑰紡 2: 鐩存帴杩愯
python scripts/batch_optimize_12.py
```

---

## 馃搳 浼樺寲鐨?12 涓畻瀛?
### 楂樹紭鍏堢骇锛圧OI 鏈€楂橈級

1. **matmul-2048x2048x2048** - 鐭╅樀涔樻硶 (8-10脳 鍔犻€?
2. **layernorm-4096** - LayerNorm (6-8脳 鍔犻€?
3. **self-attention-h16-d64-seq1024** - Self-Attention (12-15脳 鍔犻€?
4. **flash-attention-seq2048** - Flash Attention (10-20脳 鍔犻€?
5. **sparse-attention-topk2048** - Sparse Attention (10-30脳 鍔犻€?

### 涓紭鍏堢骇

6. **softmax-1M** - Softmax (3-6脳 鍔犻€?
7. **rmsnorm-4096** - RMSNorm (6-8脳 鍔犻€?
8. **gelu-activation-1M** - GELU 婵€娲?(2-4脳 鍔犻€?
9. **moe-fp8-experts8** - MoE (5-10脳 鍔犻€?
10. **batched-gemm-b32-1024x1024x1024** - Batched GEMM (3-6脳 鍔犻€?

### 浣庝紭鍏堢骇

11. **conv2d-resnet-style** - Conv2D (3-8脳 鍔犻€?
12. **fp8-gemm-2048x2048x2048** - FP8 閲忓寲 GEMM (2-4脳 鍔犻€?

---

## 馃攧 宸ヤ綔娴佺▼

### 姣忎釜绠楀瓙鐨勪紭鍖栨祦绋?
```
1. 鍒涘缓 baseline kernel
   鈫?2. 杩愯 NCU profiling
   鈫?3. 璋冪敤 LLM API 鍒嗘瀽鐡堕
   鈫?4. LLM 鍔ㄦ€佺敓鎴愪紭鍖栨柟妗?   鈫?5. 淇濆瓨浼樺寲浠ｇ爜
   鈫?6. 杩愯娴嬭瘯锛堟ā鎷熸垨鐪熷疄锛?   鈫?7. 璁板綍缁撴灉
   鈫?8. 杩涘叆涓嬩竴杞紙鎴栦笅涓€涓畻瀛愶級
```

### 瀹屾暣鎵归噺浼樺寲娴佺▼

```
鍒濆鍖?LLM 瀹㈡埛绔?  鈫?娴嬭瘯 API 杩炴帴
  鈫?閫愪釜浼樺寲 12 涓畻瀛愶紙姣忎釜 6-15 杞級
  鈫?鐢熸垚鎬荤粨鎶ュ憡
  鈫?淇濆瓨鎵€鏈夌粨鏋?```

---

## 馃摑 杈撳嚭鏂囦欢

### 姣忎釜绠楀瓙鐨勮緭鍑?
```
reference/<operator-name>/
鈹溾攢鈹€ kernel.py                        # Baseline kernel
鈹溾攢鈹€ kernel_iter1.cuda               # 绗?1 杞紭鍖栦唬鐮?鈹溾攢鈹€ kernel_iter2.cuda               # 绗?2 杞紭鍖栦唬鐮?鈹溾攢鈹€ ...
鈹溾攢鈹€ kernel_iterN.cuda               # 绗?N 杞紭鍖栦唬鐮?鈹溾攢鈹€ optimization_results.json       # 浼樺寲缁撴灉鎽樿
鈹斺攢鈹€ debug_iterX.txt                 # 璋冭瘯淇℃伅锛堝鏋滄湁閿欒锛?```

### 鍏ㄥ眬杈撳嚭

```
logs/
鈹溾攢鈹€ batch_optimization_20260602_120000.log   # 璇︾粏鏃ュ織
鈹斺攢鈹€ summary_20260602_130000.json             # 鎬荤粨鎶ュ憡
```

---

## 馃幆 LLM API 璋冪敤娴佺▼

### 1. 鍒嗘瀽 NCU 鏁版嵁

```python
# 杈撳叆
ncu_data = {
    "metrics": {
        "dram_throughput_pct": 75.0,
        "sm_utilization_pct": 45.0,
        "l2_hit_rate_pct": 60.0,
        ...
    }
}

# LLM 鍒嗘瀽
analysis = llm.analyze_bottleneck(ncu_data)
# 杈撳嚭: 涓昏鐡堕銆佹牴鍥犮€佺悊璁烘瀬闄愩€佸樊璺?```

### 2. 鍔ㄦ€佺敓鎴愪紭鍖栨柟鍚?
```python
# LLM 鐢熸垚锛堜笉闄愪簬棰勫畾涔夌瓥鐣ワ級
directions = llm.generate_directions(analysis)
# 杈撳嚭: 3-5 涓紭鍖栨柟鍚?# - 浼犵粺鏂瑰悜: Tiling, Vectorization, Tensor Core
# - 鍒涙柊鏂瑰悜: 绠楁硶閲嶆瀯, Warp Specialization, 鏂扮‖浠剁壒鎬?```

### 3. 鐢熸垚浼樺寲浠ｇ爜

```python
# LLM 鐢熸垚瀹屾暣浠ｇ爜
code = llm.generate_code(selected_direction, operator_info)
# 杈撳嚭: CUDA/Triton/C++ 浠ｇ爜锛屽甫璇︾粏涓枃娉ㄩ噴
```

---

## 鈿欙笍 閰嶇疆閫夐」

### API 閰嶇疆 (`config/api_config.json`)

```json
{
  "api_key": "REPLACE_ME_LOCALLY",
  "base_url": "https://api.siliconflow.cn/v1",
  "model": "deepseek-ai/DeepSeek-V3",
  "timeout": 60,
  "max_retries": 3,
  "temperature": 0.7
}
```

### 鎵归噺浼樺寲閰嶇疆 (`scripts/batch_optimize_12.py`)

淇敼浠ヤ笅鍙橀噺锛?- `SELECTED_OPERATORS`: 閫夋嫨瑕佷紭鍖栫殑绠楀瓙
- `GPU_NAME`: GPU 鍚嶇О
- `COMPUTE_CAPABILITY`: 璁＄畻鑳藉姏

---

## 馃敡 瀹為檯 NCU 闆嗘垚

### 鍚敤鐪熷疄 NCU profiling

鍦?`batch_optimize_12.py` 涓紝淇敼 `run_ncu_profiling` 鍑芥暟锛?
```python
def run_ncu_profiling(kernel_path: Path, operator_name: str) -> Optional[Dict]:
    """杩愯 NCU profiling 骞惰繑鍥炵粨鏋?""
    
    # 杩愯鐪熷疄鐨?NCU 鍛戒护
    ncu_command = [
        "ncu",
        "--set", "full",
        "--target-processes", "all",
        "--metrics",
        "dram__throughput.avg.pct_of_peak_sustained_elapsed,"
        "sm__throughput.avg.pct_of_peak_sustained_elapsed,"
        "l2_tex_read_hit_rate,"
        "sm__warps_active.avg.pct_of_peak_sustained_active",
        "--csv",
        "python", str(kernel_path)
    ]
    
    try:
        result = subprocess.run(
            ncu_command,
            capture_output=True,
            text=True,
            timeout=300
        )
        
        # 瑙ｆ瀽 NCU CSV 杈撳嚭
        ncu_data = parse_ncu_csv(result.stdout)
        return ncu_data
        
    except Exception as e:
        print(f"鉂?NCU 鎵ц澶辫触: {e}")
        return None
```

---

## 馃搳 棰勬湡缁撴灉

### 鏃堕棿棰勪及

- 姣忎釜绠楀瓙: 6-15 杞?脳 2-5 鍒嗛挓/杞?= 12-75 鍒嗛挓
- 12 涓畻瀛愭€昏: **绾?2-15 灏忔椂**

### 鍔犻€熸瘮棰勬湡

| 绠楀瓙绫诲瀷 | 棰勬湡鍔犻€?| 瀹為檯鍙兘 |
|---------|---------|---------|
| MatMul | 8-10脳 | 5-12脳 |
| Attention | 12-15脳 | 8-20脳 |
| LayerNorm | 6-8脳 | 4-10脳 |
| Softmax | 3-6脳 | 2-8脳 |
| MoE | 5-10脳 | 3-12脳 |

---

## 馃悰 鏁呴殰鎺掓煡

### API 璋冪敤澶辫触

```python
# 妫€鏌?API Key
python -c "from scripts.batch_optimize_12 import LLMClient; \
           client = LLMClient('your-key', 'url', 'model'); \
           print(client.chat([{'role': 'user', 'content': 'test'}]))"
```

### NCU 鎵句笉鍒?
```bash
# 妫€鏌?NCU 瀹夎
which ncu
ncu --version

# 娣诲姞鍒?PATH
export PATH=/path/to/ncu:$PATH
```

### 鍐呭瓨涓嶈冻

```python
# 鍦?batch_optimize_12.py 涓噺灏戝苟鍙?# 鎴栧鍔犳瘡杞箣闂寸殑浼戞伅鏃堕棿
time.sleep(5)  # 澧炲姞鍒?5 绉?```

---

## 馃摎 杩涢樁浣跨敤

### 鑷畾涔夌畻瀛愬垪琛?
淇敼 `SELECTED_OPERATORS`:

```python
SELECTED_OPERATORS = [
    {
        "name": "my-custom-operator",
        "type": "custom",
        "priority": 1,
        "max_rounds": 10,
        "expected_speedup": "5-8脳",
        "description": "鎴戠殑鑷畾涔夌畻瀛?
    },
    # ... 鏇村绠楀瓙
]
```

### 璋冩暣浼樺寲绛栫暐

淇敼 LLM prompt 涓殑鎸囦护锛?- 鏇存縺杩涚殑浼樺寲鏂瑰悜
- 鏇翠繚瀹堢殑鏂规
- 鐗瑰畾鐨勭‖浠剁壒鎬?- 鐗瑰畾鐨勪紭鍖栨ā寮?
---

## 鉁?妫€鏌ユ竻鍗?
寮€濮嬪墠纭繚锛?
- [ ] API Key 宸查厤缃湪 `config/api_config.json`
- [ ] NCU 宸插畨瑁呭苟鍦?PATH 涓?- [ ] CUDA 鐜宸查厤缃?- [ ] Python 渚濊禆宸插畨瑁?(`pip install -r requirements.txt`)
- [ ] 鍦?Linux 鐜涓繍琛岋紙濡傛灉闇€瑕佺湡瀹?NCU锛?- [ ] 鏈夎冻澶熺殑纾佺洏绌洪棿锛堟瘡涓畻瀛愮害 100 MB锛?
---

## 馃殌 寮€濮嬩紭鍖?
```bash
cd KernelForge-MultiAgent
python scripts/start_batch_optimization.py
```

閫夋嫨閫夐」 1 寮€濮嬫壒閲忎紭鍖栵紒

---

**棰勮鎬昏€楁椂**: 2-15 灏忔椂  
**棰勬湡鎬诲姞閫?*: 骞冲潎 5-10脳 姣忎釜绠楀瓙  
**杈撳嚭**: 12 涓紭鍖栧悗鐨?kernel + 璇︾粏鎶ュ憡
