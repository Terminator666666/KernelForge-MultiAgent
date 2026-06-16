# 蹇€熷紑濮嬫寚鍗?
**KernelForge-MultiAgent** 蹇€熶笂鎵嬫寚鍗?
## 馃殌 5 鍒嗛挓蹇€熷紑濮?
### 1. 鐜鍑嗗

```bash
# 鍏嬮殕鎴栧垏鎹㈠埌椤圭洰鐩綍
cd KernelForge-MultiAgent

# 鍦?Linux 鐜涓畨瑁呬緷璧?pip install -r requirements.txt

# 璁剧疆鐜鍙橀噺
export AKO_DATASET_PATH=/path/to/flashinfer-trace

# 鍙€? 涓嬭浇 FlashInfer 鏁版嵁闆?GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/flashinfer-ai/flashinfer-trace
```

### 2. 杩愯婕旂ず

```bash
# 杩愯绔埌绔紨绀?python demo/end_to_end_demo.py
```

浣犲皢鐪嬪埌锛?- 6 涓櫤鑳戒綋鍒濆鍖?- Phase 1: 鎺㈢储涓庡垵姝ヤ紭鍖?- Phase 2: 娣卞害浼樺寲涓庣簿璋?- Phase 3: 楠岃瘉涓庡綊妗?- 鏈€缁堜紭鍖栨€荤粨

### 3. 鍗曟鎵嬪姩浼樺寲 (Mode 1)

```bash
# 鍚姩 Master Agent
python master/master_agent.py \
  --mode manual \
  --family dsa-sparse-attention \
  --gpu RTX5070 \
  --backend local

# 鍦ㄧ敓鎴愮殑宸ヤ綔绌洪棿涓伐浣?cd ../kfma-run-dsa-sparse-attention-r1

# 鎵嬪姩涓?Claude 浜や簰浼樺寲
# (鍦ㄥ疄闄呯幆澧冧腑浣跨敤 claude CLI)
```

### 4. 鑷姩闂幆浼樺寲 (Mode 2)

```bash
# 鍚姩闂幆浼樺寲
python master/master_agent.py \
  --mode closed-loop \
  --family matmul \
  --gpu B200 \
  --backend local \
  --max-rounds 10

# 绯荤粺灏嗚嚜鍔ㄨ繍琛屽杞紭鍖?# 鏌ョ湅杩涘害: tail -f master/harness-ledger.md
```

### 5. 鍗忓悓婕斿寲妯″紡 (Mode 3)

```bash
# 鍚姩鍗忓悓婕斿寲
python master/master_agent.py \
  --mode co-evolution \
  --family gdn-prefill \
  --gpu B200 \
  --backend modal \
  --max-rounds 20

# 绯荤粺灏嗕紭鍖?Kernel 骞舵敼杩涘伐鍏烽摼
```

---

## 馃摎 璇︾粏鏁欑▼

### 鏁欑▼ 1: 浼樺寲鑷畾涔?MatMul

**鐩爣**: 灏嗕竴涓畝鍗曠殑 MatMul kernel 浼樺寲鍒?8脳 鍔犻€?
**姝ラ**:

1. **鍒涘缓鍩虹嚎 kernel**
```python
# solution/kernel.py
import torch

def matmul_baseline(A, B):
    return torch.matmul(A, B)
```

2. **鍚姩浼樺寲**
```bash
python master/master_agent.py \
  --mode closed-loop \
  --family custom-matmul \
  --gpu RTX5070 \
  --max-rounds 10
```

3. **鐩戞帶杩涘害**
```bash
# 鏌ョ湅瀹¤鏃ュ織
tail -f master/harness-ledger.md

# 鏌ョ湅褰撳墠鏈€浣崇増鏈?cat reference/custom-matmul/README.md
```

4. **鏌ョ湅缁撴灉**
```bash
# 鏈€浣崇増鏈?cat reference/custom-matmul/variants/r5-final/kernel.py

# 鎬ц兘鎶ュ憡
cat reference/custom-matmul/variants/r5-final/results.json
```

### 鏁欑▼ 2: 浣跨敤鑷畾涔夌瓥鐣?
**鐩爣**: 娣诲姞骞朵娇鐢ㄨ嚜宸辩殑浼樺寲绛栫暐

**姝ラ**:

1. **鍒涘缓绛栫暐鏂囨。**
```bash
mkdir -p skills/my-custom-strategy
cat > skills/my-custom-strategy/SKILL.md << 'EOF'
---
name: my-custom-strategy
description: 鎴戠殑鑷畾涔変紭鍖栫瓥鐣?---

# My Custom Strategy

## 鏍稿績鎬濇兂
...

## 瀹炵幇瑕佺偣
...
EOF
```

2. **鍦ㄩ厤缃腑鍚敤**
```toml
# config/default_config.toml
[skills]
enabled_skills = [
    "strategy-library",
    "my-custom-strategy"  # 娣诲姞杩欒
]
```

3. **浣跨敤绛栫暐**

绛栫暐浼氳嚜鍔ㄨ鍔犺浇鍒板瓙鏅鸿兘浣撶殑 SKILL 鐩綍涓€?
### 鏁欑▼ 3: 绉绘鑷畾涔夊熀鍑嗘祴璇?
**鐩爣**: 浣跨敤鑷繁鐨勫熀鍑嗘祴璇曟浛浠?flashinfer-bench

**姝ラ**:

1. **鍒涘缓閫傞厤鍣?*
```python
# scripts/my_benchmark_adapter.py
class MyBenchmarkAdapter:
    def run(self, kernel_path, workloads):
        # 瀹炵幇娴嬭瘯閫昏緫
        return {"latency": 10.5, "speedup": 2.3}
    
    def profile(self, kernel_path, workload):
        # 瀹炵幇 profiling 閫昏緫
        return {"metrics": {...}}
    
    # ... 鍏朵粬鏂规硶
```

2. **娉ㄥ唽閫傞厤鍣?*
```python
# 鍦?master_agent.py 鎴栭厤缃腑娉ㄥ唽
benchmark_adapter = MyBenchmarkAdapter()
```

3. **浣跨敤鑷畾涔夊熀鍑?*
```bash
python master/master_agent.py \
  --mode closed-loop \
  --benchmark my-benchmark \
  ...
```

---

## 馃幆 甯歌浠诲姟

### 浠诲姟 1: 鏌ョ湅浼樺寲鍘嗗彶

```bash
# 鏌ョ湅鏌愪釜绠楀瓙瀹舵棌鐨勬墍鏈変紭鍖栫増鏈?cat reference/dsa-sparse-attention/README.md

# 鏌ョ湅鐗瑰畾鐗堟湰鐨勮缁嗕俊鎭?cat reference/dsa-sparse-attention/variants/r3-tiling/kernel.py
```

### 浠诲姟 2: 姣旇緝涓や釜鐗堟湰

```bash
# 浣跨敤 diff 姣旇緝
diff \
  reference/matmul/variants/r1-baseline/kernel.py \
  reference/matmul/variants/r5-optimized/kernel.py
```

### 浠诲姟 3: 閲嶆柊杩愯澶辫触鐨勮疆娆?
```bash
# 鏌ョ湅澶辫触鎬荤粨
cat reference/matmul/_failed/matmul-r4/summary.md

# 鍒嗘瀽澶辫触鍘熷洜
cat reference/matmul/_failed/matmul-r4/phase1-transcript.jsonl

# 浣跨敤鏂扮殑鎻愮ず閲嶆柊杩愯
python master/master_agent.py \
  --mode manual \
  --family matmul \
  --parent r3-working
```

### 浠诲姟 4: 瀵煎嚭鏈€浣崇増鏈?
```bash
# 鎵撳寘瑙ｅ喅鏂规
python scripts/pack_solution.py \
  --kernel reference/matmul/variants/r5-final/kernel.py \
  --config reference/matmul/variants/r5-final/config.toml \
  --output matmul-solution.json

# 澶嶅埗鍒扮敓浜х幆澧?cp matmul-solution.json /path/to/production/
```

---

## 馃敡 閰嶇疆璋冧紭

### 璋冧紭 1: 鍔犲揩 Phase 1 杩唬

```toml
# config/default_config.toml
[phase1]
max_iterations = 5      # 鍑忓皯鍒?5 娆?subset_ratio = 0.05     # 鍙敤 5% 宸ヤ綔璐熻浇
time_budget = 7200      # 2 灏忔椂
```

### 璋冧紭 2: 鎻愰珮 Phase 2 鎺㈢储娣卞害

```toml
[phase2]
max_iterations = 20     # 澧炲姞鍒?20 娆?exploration_paths = 5   # 鎺㈢储 5 鏉¤矾寰?variance_runs = 30      # 鏇村鏂瑰樊鍒嗘瀽
```

### 璋冧紭 3: 鍚敤骞惰鎵ц锛堟湭鏉ュ姛鑳斤級

```toml
[agents.coordinator]
parallel_execution = true
max_parallel_agents = 4
```

---

## 馃悰 鏁呴殰鎺掓煡

### 闂 1: 鎵句笉鍒版暟鎹泦

**閿欒**:
```
FileNotFoundError: Dataset not found at /path/to/flashinfer-trace
```

**瑙ｅ喅**:
```bash
# 璁剧疆姝ｇ‘鐨勭幆澧冨彉閲?export AKO_DATASET_PATH=/correct/path/to/flashinfer-trace

# 鎴栧湪閰嶇疆鏂囦欢涓缃?[benchmark]
dataset_path = "/correct/path/to/flashinfer-trace"
```

### 闂 2: CUDA 鐗堟湰涓嶅吋瀹?
**閿欒**:
```
RuntimeError: CUDA driver version is insufficient for CUDA runtime version
```

**瑙ｅ喅**:
```bash
# 妫€鏌?CUDA 鐗堟湰
nvidia-smi

# 瀹夎鍖归厤鐨?PyTorch
pip install torch==2.2.0+cu121 -f https://download.pytorch.org/whl/torch_stable.html
```

### 闂 3: NCU 鎵句笉鍒?
**閿欒**:
```
FileNotFoundError: ncu command not found
```

**瑙ｅ喅**:
```bash
# 瀹夎 NVIDIA Nsight Compute
# 浠?https://developer.nvidia.com/nsight-compute 涓嬭浇

# 娣诲姞鍒?PATH
export PATH=/path/to/ncu:$PATH
```

### 闂 4: 浼樺寲闄峰叆鍋滄粸

**鐥囩姸**: 杩炵画澶氳疆鏃犳€ц兘鎻愬崌

**璇婃柇**:
```bash
# 鏌ョ湅鏈€杩戠殑杩唬
tail -20 master/harness-ledger.md

# 鏌ョ湅 NCU 鎶ュ憡
ncu --print-summary ncu_report.ncu-rep
```

**瑙ｅ喅**:
- 灏濊瘯涓嶅悓鐨勪紭鍖栫瓥鐣?- 妫€鏌ユ槸鍚﹁揪鍒扮‖浠舵瀬闄?- 鑰冭檻鎹竴涓紭鍖栨柟鍚?
---

## 馃搳 鎬ц兘鍩哄噯

### 棰勬湡鎬ц兘鎻愬崌

| 绠楀瓙绫诲瀷 | Phase 1 | Phase 2 | Phase 3 |
|---------|---------|---------|---------|
| MatMul | 2-3脳 | 5-8脳 | 8-10脳 |
| Reduce | 3-5脳 | 6-10脳 | 10-15脳 |
| Attention | 2-4脳 | 5-8脳 | 8-12脳 |

### 鏃堕棿棰勭畻

- **Phase 1**: 2-4 灏忔椂锛堝揩閫熸帰绱級
- **Phase 2**: 4-8 灏忔椂锛堟繁搴︿紭鍖栵級
- **Phase 3**: 2-4 灏忔椂锛堥獙璇佸綊妗ｏ級
- **鎬昏**: 8-16 灏忔椂/绠楀瓙

---

## 馃帗 杩涢樁涓婚

### 杩涢樁 1: 鑷畾涔夋櫤鑳戒綋

鍒涘缓鑷繁鐨勪笓涓氭櫤鑳戒綋锛堜緥濡傦細鍐呭瓨浼樺寲涓撳锛?
### 杩涢樁 2: 绛栫暐缁勫悎浼樺寲

鑷姩鎼滅储鏈€浣崇瓥鐣ョ粍鍚?
### 杩涢樁 3: 璺ㄧ畻瀛愮煡璇嗚縼绉?
鍒╃敤宸蹭紭鍖栫畻瀛愮殑缁忛獙鍔犻€熸柊绠楀瓙浼樺寲

### 杩涢樁 4: 纭欢鎰熺煡浼樺寲

閽堝鐗瑰畾 GPU 鏋舵瀯鐨勬繁搴︿紭鍖?
---

## 馃挕 鏈€浣冲疄璺?
1. **浠庣畝鍗曞紑濮?*: 鍏堢敤 Mode 1 鎵嬪姩妯″紡鐔熸倝娴佺▼
2. **灏忔蹇窇**: Phase 1 蹇€熻凯浠ｏ紝涓嶈杩囧害浼樺寲
3. **璁板綍涓€鍒?*: ITERATIONS.md 鏄綘鐨勪紭鍖栨棩蹇?4. **楠岃瘉姝ｇ‘鎬?*: 姣忔浼樺寲鍚庤繍琛?sanitizer
5. **鐗堟湰鎺у埗**: 瀹氭湡 git commit
6. **鍙傝€冩枃妗?*: 閬囧埌闂鍏堟煡鐪?docs/

---

## 馃 鑾峰彇甯姪

- **鏂囨。**: `docs/` 鐩綍
- **绀轰緥**: `demo/` 鐩綍
- **鎶€鑳芥枃妗?*: `skills/*/SKILL.md`
- **閰嶇疆**: `config/default_config.toml`

---

**鐜板湪浣犲凡缁忓噯澶囧ソ寮€濮嬩紭鍖栦簡锛?* 馃殌

```bash
# 寮€濮嬩綘鐨勭涓€娆′紭鍖?python demo/end_to_end_demo.py
```
