# 椤圭洰鏀硅繘璇存槑锛氫粠闈欐€佺瓥鐣ュ簱鍒板姩鎬佹帰绱?
## 馃幆 鍏抽敭鏀硅繘

### 闂鍙戠幇

鐢ㄦ埛鎸囧嚭锛歁LSys2026 FlashInfer Contest 椤圭洰浣跨敤**鍔ㄦ€佹帰绱?*鑰岄潪闈欐€佺瓥鐣ュ簱锛岃繖鏄粬浠彇寰?30.71脳 鍔犻€熺殑鍏抽敭鍘熷洜銆?
### 鍘熻璁＄殑灞€闄?
鉂?**闈欐€佺瓥鐣ュ簱** (`strategy-library/SKILL.md`)锛?- 9 涓浐瀹氱瓥鐣?- Agent 鍙樻垚"绛栫暐鎵ц鍣?
- 闄愬埗鎺㈢储绌洪棿
- 鏃犳硶搴斿鏂扮畻瀛愮被鍨?- 缂轰箯鍒涙柊鎬?
### 鏂拌璁＄殑浼樺娍

鉁?**鍔ㄦ€佹帰绱㈠紩鎿?* (`optimization-knowledge/SKILL.md` + `dynamic_exploration.py`)锛?- **鐭ヨ瘑搴撹€岄潪娓呭崟**锛氭彁渚涚煡璇嗗弬鑰冿紝涓嶅己鍒舵墽琛岃矾寰?- **鍔ㄦ€佺敓鎴愭柟鍚?*锛氭牴鎹?profiling 鏁版嵁瀹炴椂鐢熸垚浼樺寲鏂瑰悜
- **鍒涢€犳€ф€濊€?*锛氫笉鍙楅瀹氫箟绛栫暐闄愬埗
- **Humanize Planning**锛氶泦鎴?`/humanize:gen-plan` 宸ヤ綔娴?- **绗竴鎬у師鐞?*锛氫粠鐞嗚鏋侀檺鍒嗘瀽锛岃€岄潪濂楃敤妯℃澘

---

## 馃搳 璁捐瀵规瘮

### 闈欐€佺瓥鐣ュ簱鏂瑰紡

```python
# 鏃ф柟寮忥細浠庡浐瀹氬垪琛ㄩ€夋嫨
strategies = [
    "matmul_tiling",
    "vectorized_memory",
    "tensor_core",
    # ... 鍥哄畾鐨?9 涓?]

# 鏈烘鎵ц
for strategy in strategies:
    apply_strategy(strategy)
    test_performance()
```

**缁撴灉**锛欰gent 鏄墽琛屽櫒锛屼笉鏄帰绱㈣€?
### 鍔ㄦ€佹帰绱㈡柟寮?
```python
# 鏂版柟寮忥細鍔ㄦ€佸垎鏋愬拰鐢熸垚
profiling_data = run_ncu_profiling()
bottleneck = deep_analysis(profiling_data)  # 鏍瑰洜鍒嗘瀽
directions = generate_directions(bottleneck)  # 鍔ㄦ€佺敓鎴?
# 姣忎釜鏂瑰悜閮芥槸閽堝褰撳墠闂瀹氬埗鐨?for direction in directions:
    print(f"{direction.name}: {direction.rationale}")
    # 涓嶆槸濂楃敤妯℃澘锛岃€屾槸鍒涢€犳€ц璁?```

**缁撴灉**锛欰gent 鏄紭鍖栧伐绋嬪笀锛岃兘鍒涙柊

---

## 馃敩 鏍稿績鏀硅繘鐐?
### 1. 浠?鎵ц娓呭崟"鍒?鐭ヨ瘑鍙傝€?

**鏃ф柟寮?*锛?```markdown
## 绛栫暐 1: matmul_tiling
搴旂敤杩欎釜绛栫暐...
```

**鏂版柟寮?*锛?```markdown
## 鐭ヨ瘑棰嗗煙锛氬唴瀛樹紭鍖?
### Tiling 鐨勬湰璐?涓嶆槸"搴旂敤 tiling 绛栫暐"锛岃€屾槸鐞嗚В锛?- 涓轰粈涔堥渶瑕?tiling锛燂紙鏁版嵁閲嶇敤锛?- 鏈€浼?tile size 濡備綍璁＄畻锛?- 鏉冭　鏄粈涔堬紵锛坰hared memory vs 瀵勫瓨鍣級

### 浣犵殑闂
- 鏁版嵁閲嶇敤妯″紡鏄粈涔堬紵
- Tiling 鏄敮涓€瑙ｅ悧锛?- 鑳藉惁鍒涢€犳€т慨鏀癸紵
```

### 2. 娣卞害鐡堕鍒嗘瀽

**鏃ф柟寮?*锛氱畝鍗曞垎绫?```python
if dram_throughput > 70%:
    return "memory_bound"
```

**鏂版柟寮?*锛氭牴鍥犲垎鏋?```python
def analyze_bottleneck(profiling_data):
    # 1. 璁＄畻 roofline 浣嶇疆
    # 2. 鍒嗘瀽鏍瑰洜锛堜负浠€涔堟槸杩欎釜鐡堕锛燂級
    # 3. 璁＄畻鐞嗚鏋侀檺
    # 4. 鍒嗘瀽宸窛
    return {
        "primary_bottleneck": "memory_bound",
        "root_cause": "L2 cache miss rate high - poor data locality",
        "theoretical_limit": "8 TB/s",
        "gap": "褰撳墠鍙敤浜?20%锛屾湁 5脳 鎻愬崌绌洪棿"
    }
```

### 3. 鍔ㄦ€佺敓鎴愪紭鍖栨柟鍚?
**鏃ф柟寮?*锛氫粠鍥哄畾鍒楄〃鍖归厤
```python
if bottleneck == "memory_bound":
    return ["vectorized_memory", "kernel_fusion"]
```

**鏂版柟寮?*锛氶拡瀵规€х敓鎴?```python
def generate_directions(bottleneck_analysis, operator_info):
    directions = []
    
    # 鏍规嵁鍏蜂綋鏍瑰洜瀹氬埗鏂瑰悜
    if "L2 cache miss" in root_cause:
        directions.append(OptimizationDirection(
            name="Aggressive Tiling with Shared Memory",
            rationale=f"Root cause: {root_cause}. "
                     f"閫氳繃鍒嗗潡灏嗘暟鎹斁鍏?shared memory锛?
                     f"鍏呭垎鍒╃敤鏁版嵁閲嶇敤",
            potential_speedup="3-8脳 (鍩轰簬鏁版嵁閲嶇敤鐜囧垎鏋?",
            implementation_plan="1. 鍒嗘瀽閲嶇敤妯″紡\n2. 璁＄畻鏈€浼?tile\n..."
        ))
    
    # 鎬绘槸鎺㈢储鍒涙柊鏂瑰悜
    directions.append(OptimizationDirection(
        name="Algorithm Redesign for Hardware",
        rationale="閲嶆柊璁捐绠楁硶浠ラ€傚簲 GPU锛堝 Flash Attention锛?,
        potential_speedup="5-15脳",
        risk_level="high"
    ))
    
    return directions
```

### 4. Humanize Planning 闆嗘垚

**宸ヤ綔娴?*锛?
```
1. 娣卞害鍒嗘瀽鐡堕
   鈫?2. 鍔ㄦ€佺敓鎴愬涓紭鍖栨柟鍚?   鈫?3. Agent 閫夋嫨鏈€鏈夋綔鍔涚殑鏂瑰悜
   鈫?4. 鐢熸垚 draft.md
   鈫?5. 杩愯 /humanize:gen-plan
   鈫?6. 鑾峰緱璇︾粏瀹炵幇璁″垝
   鈫?7. 瀹炵幇骞惰凯浠?```

**draft.md 绀轰緥**锛?```markdown
# Optimization Implementation Draft

## Selected Direction: Aggressive Tiling with Shared Memory

### Rationale
Root cause: L2 cache miss rate high. 閫氳繃鍒嗗潡灏嗙儹鏁版嵁
鏀惧叆 shared memory锛岄璁″彲鍑忓皯 80% DRAM 璁块棶銆?
### Expected Impact
- Potential Speedup: 3-8脳
- Risk Level: medium

### Implementation Plan Outline
1. 鍒嗘瀽鏁版嵁閲嶇敤妯″紡
2. 璁＄畻鏈€浼?tile size
3. 瀹炵幇鍒嗗潡鍔犺浇閫昏緫
4. 澶勭悊杈圭晫鏉′欢

### References
- Flash Attention paper
- CUTLASS tiling patterns

---
Ready for /humanize:gen-plan
```

---

## 馃挕 瀹為檯鏁堟灉瀵规瘮

### MLSys2026 鎴愭灉锛堜娇鐢ㄥ姩鎬佹帰绱級

| Kernel | 鍔犻€熸瘮 | 鏂规硶 |
|--------|--------|------|
| DSA sparse attention | **30.71脳** | 鍔ㄦ€佹帰绱?+ 鍒涙柊绠楁硶 |
| GDN prefill | **2.30脳** | 纭欢鐗规€ф繁搴﹀埄鐢?|

**鍏抽敭**锛氫粬浠病鏈夊鐢ㄩ瀹氫箟绛栫暐锛岃€屾槸锛?1. 娣卞害鐞嗚В绠楀瓙
2. 鐮旂┒纭欢鐗规€?3. 鍒涢€犳€ц璁℃柟妗?4. 杩唬浼樺寲

### 闈欐€佺瓥鐣ュ簱鐨勫ぉ鑺辨澘

鐞嗚涓婇檺锛?- 9 涓瓥鐣ョ粍鍚?= 鏈€澶?2-10脳 鍔犻€?- 鏃犳硶绐佺牬棰勫畾涔夌┖闂?
**涓轰粈涔堬紵** 鍥犱负鐪熸鐨勭獊鐮存潵鑷細
- 绠楁硶绾ч噸鏋勶紙Flash Attention 鐨?1-pass 璁捐锛?- 纭欢鐗规€ф繁搴﹀埄鐢紙TMA, TMEM锛?- 鍒涙柊鎬х粍鍚?- 閽堝鎬у畾鍒?
杩欎簺閮戒笉鍦ㄥ浐瀹氱瓥鐣ュ簱涓€?
---

## 馃殌 鏂扮殑宸ヤ綔娴?
### Phase 1: 娣卞害鐞嗚В + 鍔ㄦ€佹帰绱?
```python
# 1. 娣卞害鍒嗘瀽
profiling_data = run_ncu_profiling(kernel)
bottleneck = engine.analyze_bottleneck(profiling_data)

print(f"Primary bottleneck: {bottleneck['primary_bottleneck']}")
print(f"Root cause: {bottleneck['root_cause']}")
print(f"Gap to theoretical limit: {bottleneck['gap']}")

# 2. 鍔ㄦ€佺敓鎴愭柟鍚戯紙涓嶉檺浜?9 涓瓥鐣ワ級
directions = engine.generate_optimization_directions(
    bottleneck, 
    operator_info
)

# 鍙兘鐢熸垚 5-10 涓拡瀵规€ф柟鍚戯紝鍖呮嫭鍒涙柊鏂瑰悜
for d in directions:
    print(f"{d.name}: {d.rationale}")
    print(f"  Expected: {d.potential_speedup}")
    print(f"  Risk: {d.risk_level}")

# 3. Agent 閫夋嫨 + Humanize Planning
selected = agent_select_direction(directions)
draft = engine.generate_humanize_plan_prompt(directions, selected, operator_info)

# 4. 鐢熸垚璇︾粏璁″垝
save_draft(draft, "docs/draft.md")
run_humanize_gen_plan("docs/draft.md")
```

### Phase 2: 鍒涙柊鎬у疄鐜?
Agent 涓嶆槸鏈烘鎵ц锛岃€屾槸锛?- 鐮旂┒鐩稿叧璁烘枃鍜屽疄鐜?- 鐞嗚В璁捐鎬濇兂
- 鍒涢€犳€у湴搴旂敤鍜屼慨鏀?- 灏忔楠岃瘉鍋囪

---

## 馃摑 鏂囦欢鏇存柊

### 鏂板鏂囦欢

1. **`skills/optimization-knowledge/SKILL.md`**
   - 鐭ヨ瘑搴撹€岄潪绛栫暐娓呭崟
   - 寮鸿皟绗竴鎬у師鐞嗘€濊€?   - 妗堜緥瀛︿範鑰岄潪妯℃澘
   - 鍔ㄦ€佹帰绱㈡祦绋?
2. **`agents/dynamic_exploration.py`**
   - 鍔ㄦ€佹帰绱㈠紩鎿庡疄鐜?   - 娣卞害鐡堕鍒嗘瀽
   - 浼樺寲鏂瑰悜鐢熸垚
   - Humanize Planning 闆嗘垚

### 淇濈暀浣嗛檷绾?
3. **`skills/strategy-library/SKILL.md`**
   - 闄嶇骇涓哄弬鑰冭祫鏂?   - 涓嶄綔涓轰富瑕佷紭鍖栬矾寰?   - 鍙緵 Agent 鍙傝€冧絾涓嶅己鍒?
---

## 馃幆 鍏抽敭鍚ず

### 1. Agent 鐨勮鑹插畾浣?
**閿欒**锛欰gent = 绛栫暐鎵ц鍣?- 缁欏畠涓€涓瓥鐣ュ垪琛?- 璁╁畠閫愪釜灏濊瘯
- 閫夋嫨鏈€濂界殑

**姝ｇ‘**锛欰gent = 浼樺寲宸ョ▼甯?- 缁欏畠鐭ヨ瘑鍜屽伐鍏?- 璁╁畠鍒嗘瀽鍜屾€濊€?- 鍒涢€犳€у湴瑙ｅ喅闂

### 2. 闈欐€?vs 鍔ㄦ€?
**闈欐€佹柟娉曢€傚悎**锛?- 鏍囧噯闂
- 宸茬煡瑙ｅ喅鏂规
- 蹇€熷師鍨?
**鍔ㄦ€佹柟娉曢€傚悎**锛?- 澶嶆潅闂
- 闇€瑕佸垱鏂?- 杩芥眰鏋佽嚧鎬ц兘

Kernel 浼樺寲鏄剧劧灞炰簬鍚庤€呫€?
### 3. LLM Agent 鐨勪紭鍔?
LLM Agent 鐨勭湡姝ｄ环鍊硷細
- 鉁?鐞嗚В澶嶆潅姒傚康
- 鉁?鍒涢€犳€ф€濊€?- 鉁?璺ㄩ鍩熺煡璇嗙粍鍚?- 鉂?**涓嶆槸**鏈烘鎵ц娓呭崟

鎴戜滑搴旇鍙戞尌浼樺娍鑰岄潪闄愬埗瀹冦€?
---

## 馃檹 鎰熻阿鐢ㄦ埛鍙嶉

杩欎釜鏀硅繘瀹屽叏鏉ヨ嚜鐢ㄦ埛鐨勬礊瀵燂細

> "D:\Agent\mlsys2026-flashinfer-contest-main 杩欎釜椤圭洰閲屼笉鏄彲浠ュ姩鎬佹帰绱㈠悧锛岀敤杩欎釜浼樺娍鏇挎崲9涓潤鎬佺瓥鐣ュ簱鏄笉鏄洿濂斤紵"

杩欎釜闂涓€閽堣琛€鍦版寚鍑轰簡鍘熻璁＄殑鏍规湰闂锛屼績鎴愪簡浠?鎵ц鍣?鍒?鎺㈢储鑰?鐨勮寖寮忚浆鍙樸€?
---

## 馃搳 鎬荤粨

### 鏀硅繘鍓嶏細闈欐€佺瓥鐣ュ簱

```
绛栫暐搴?鈫?鍖归厤鐡堕 鈫?搴旂敤绛栫暐 鈫?娴嬭瘯
```
- 绠€鍗曚絾鏈夐檺
- 2-10脳 鍔犻€熷ぉ鑺辨澘

### 鏀硅繘鍚庯細鍔ㄦ€佹帰绱?
```
娣卞害鍒嗘瀽 鈫?鍔ㄦ€佺敓鎴愭柟鍚?鈫?鍒涢€犳€ц璁?鈫?瀹炵幇楠岃瘉
```
- 澶嶆潅浣嗗己澶?- 10-30脳 鍔犻€熸綔鍔涳紙濡?MLSys2026锛?
---

**杩欐墠鏄湡姝ｇ殑澶氭櫤鑳戒綋浼樺寲绯荤粺锛?* 馃殌
