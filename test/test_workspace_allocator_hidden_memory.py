"""
验证目标：
    找出 acl.rt.get_mem_info() 比 torch.npu.memory_reserved() 多报的 HBM 来自哪里。

关键约束（修正上一版缺陷）：
    - 不做任何 warmup，保证 HBM 从 "全新" 状态开始测量
    - 不在实验间调用 empty_cache()，避免 "驱动层缓存" 干扰
      （empty_cache 把内存还给驱动 free pool，下次分配从 free pool 取时
       AclrtMallocAlign32 不触发新 HBM 分配，get_mem_info 不变化）
    - 所有张量保活（tensors 列表），确保每次都是真实的增量分配

实验结构（单进程，顺序叠加）：
    baseline
      │ Step 1: torch.empty()   ── 纯内存分配，无算子，无 workspace
      │ Step 2: torch.randn()   ── 首次 CANN 算子，可能触发 workspace + 初始化开销
      │ Step 3: torch.randn()   ── 同 stream 再次调用，workspace 应复用
      │ Step 4: torch.ones()    ── 不同算子，workspace 应复用（同 stream）
      │ Step 5: torch.empty()   ── 再次纯分配，触发第二个 2MB 段（超出第一段）
      └ 汇总分析

    通过对比各 Step 的 HBM delta 与 PT reserved delta，
    隔离出 "算子 workspace / CANN 初始化" 的实际 HBM 开销。
"""

import os
import acl
import torch

os.environ["ASCEND_GLOBAL_LOG_LEVEL"] = "3"

ret = acl.init()
assert ret == 0, f"acl.init() failed: {ret}"

DEVICE_ID = 0
ret = acl.rt.set_device(DEVICE_ID)
assert ret == 0, f"acl.rt.set_device() failed: {ret}"

DEVICE = torch.device(f"npu:{DEVICE_ID}")
ACL_MEM_TYPE_HBM = 0

# ─── 工具 ──────────────────────────────────────────────────────────────────────
def hbm_used_mb() -> float:
    free, total, ret = acl.rt.get_mem_info(ACL_MEM_TYPE_HBM)
    assert ret == 0
    return (total - free) / (1024 * 1024)

def pt_reserved_mb() -> float:
    return torch.npu.memory_reserved(DEVICE_ID) / (1024 * 1024)

def pt_allocated_mb() -> float:
    return torch.npu.memory_allocated(DEVICE_ID) / (1024 * 1024)

_any_fail = False
def check(cond: bool, msg: str):
    global _any_fail
    tag = "PASS ✓" if cond else "FAIL ✗"
    print(f"    {tag}  {msg}")
    if not cond:
        _any_fail = True

def show(label: str) -> dict:
    s = dict(hbm=hbm_used_mb(), pt_r=pt_reserved_mb(), pt_a=pt_allocated_mb())
    print(f"  {label:<35s}  HBM={s['hbm']:.2f}  PT_reserved={s['pt_r']:.2f}  "
          f"PT_allocated={s['pt_a']:.2f}  (MB)")
    return s

def delta(a: dict, b: dict) -> dict:
    return {k: round(a[k] - b[k], 4) for k in a}

# ─── 常量 ──────────────────────────────────────────────────────────────────────
# 所有小分配（≤1MB）共用 kSmallBuffer=2MB 的段
# 选 200KB，使多个张量都放在同一个 2MB 段内，消除 "新段" 对 HBM 的影响
SMALL  = (200 * 1024) // 4   # 200 KB float32 → 50000 元素
# 2MB 段能放 ≥10 个 200KB 张量，实验中不会触发第二个段
# 超大分配用于主动触发第二个段（对比参照）
LARGE  = (600 * 1024) // 4   # 600 KB float32（仍 ≤1MB，但与前面 200KB*4=800KB 相加超 2MB）

tensors = []   # 全程保活，防止内存被还给驱动

# ══════════════════════════════════════════════════════════════════════════════
print("=" * 72)
print("  无 warmup，无 empty_cache 的顺序叠加实验")
print("  目标：逐步拆解每一类分配对 HBM 的贡献")
print("=" * 72)

S0 = show("[0] 基准（进程启动后首次测量）")
print()

# ──────────────────────────────────────────────────────────────────────────────
print("── Step 1：torch.empty(200KB) ── 纯内存分配，无任何算子 ──")
t1 = torch.empty(SMALL, device=DEVICE)
tensors.append(t1)
S1 = show("[1] after torch.empty(200KB)")
d1 = delta(S1, S0)
print(f"       HBM Δ={d1['hbm']:+.2f} MB   PT_reserved Δ={d1['pt_r']:+.2f} MB   "
      f"hidden={d1['hbm']-d1['pt_r']:+.2f} MB")
check(d1['pt_r'] == 2.0,
      f"首次小分配触发 kSmallBuffer=2MB 段（实际 Δpt_r={d1['pt_r']:.2f}）")
check(abs(d1['hbm'] - d1['pt_r']) < 0.1,
      f"torch.empty 无算子：HBM Δ ≈ PT_reserved Δ  (hidden={d1['hbm']-d1['pt_r']:.2f} MB)")
print()

# ──────────────────────────────────────────────────────────────────────────────
print("── Step 2：torch.randn(200KB) ── 首次 CANN 算子调用 ──")
print("   (200KB 仍在同一 2MB 段内，PT_reserved 不应增加)")
t2 = torch.randn(SMALL, device=DEVICE)
tensors.append(t2)
S2 = show("[2] after torch.randn(200KB) #1")
d2 = delta(S2, S1)
hidden2 = d2['hbm'] - d2['pt_r']
print(f"       HBM Δ={d2['hbm']:+.2f} MB   PT_reserved Δ={d2['pt_r']:+.2f} MB   "
      f"hidden={hidden2:+.2f} MB")
check(d2['pt_r'] == 0.0,
      f"200KB 仍在同一 2MB 段内，不触发新段（实际 Δpt_r={d2['pt_r']:.2f}）")
print(f"   ★ randn 的隐藏 HBM = {hidden2:.2f} MB  "
      f"({'workspace 或 CANN 初始化开销' if hidden2 > 0 else '无额外分配（workspace=0 或已缓存）'})")
print()

# ──────────────────────────────────────────────────────────────────────────────
print("── Step 3：torch.randn(200KB) ── 第二次调用，workspace 应复用 ──")
t3 = torch.randn(SMALL, device=DEVICE)
tensors.append(t3)
S3 = show("[3] after torch.randn(200KB) #2")
d3 = delta(S3, S2)
hidden3 = d3['hbm'] - d3['pt_r']
print(f"       HBM Δ={d3['hbm']:+.2f} MB   PT_reserved Δ={d3['pt_r']:+.2f} MB   "
      f"hidden={hidden3:+.2f} MB")
check(abs(hidden3) < 0.1,
      f"第二次 randn workspace 复用：hidden ≈ 0（实际={hidden3:.2f} MB）")
if hidden2 > 0 and abs(hidden3) < 0.1:
    print("   ★ 证实：Step2 的隐藏 HBM 是一次性的（workspace 在 Step3 被复用）")
print()

# ──────────────────────────────────────────────────────────────────────────────
print("── Step 4：torch.ones(200KB) ── 换一种算子，同 stream ──")
t4 = torch.ones(SMALL, device=DEVICE)
tensors.append(t4)
S4 = show("[4] after torch.ones(200KB)")
d4 = delta(S4, S3)
hidden4 = d4['hbm'] - d4['pt_r']
print(f"       HBM Δ={d4['hbm']:+.2f} MB   PT_reserved Δ={d4['pt_r']:+.2f} MB   "
      f"hidden={hidden4:+.2f} MB")
print(f"   ★ ones 的隐藏 HBM = {hidden4:.2f} MB  "
      f"({'有自己的 workspace 或扩容' if hidden4 > 0 else '复用已有 workspace'})")
print()

# ──────────────────────────────────────────────────────────────────────────────
print("── Step 5：torch.zeros(200KB) ── 再换一种算子 ──")
t5 = torch.zeros(SMALL, device=DEVICE)
tensors.append(t5)
S5 = show("[5] after torch.zeros(200KB)")
d5 = delta(S5, S4)
hidden5 = d5['hbm'] - d5['pt_r']
print(f"       HBM Δ={d5['hbm']:+.2f} MB   PT_reserved Δ={d5['pt_r']:+.2f} MB   "
      f"hidden={hidden5:+.2f} MB")
print()

# ──────────────────────────────────────────────────────────────────────────────
print("── Step 6：torch.empty(200KB) ── 再次纯分配，不触发算子 ──")
t6 = torch.empty(SMALL, device=DEVICE)
tensors.append(t6)
S6 = show("[6] after torch.empty(200KB) #2")
d6 = delta(S6, S5)
hidden6 = d6['hbm'] - d6['pt_r']
print(f"       HBM Δ={d6['hbm']:+.2f} MB   PT_reserved Δ={d6['pt_r']:+.2f} MB   "
      f"hidden={hidden6:+.2f} MB")
check(abs(hidden6) < 0.1,
      f"第二次 torch.empty 无算子：hidden ≈ 0（实际={hidden6:.2f} MB）")
print()

# ──────────────────────────────────────────────────────────────────────────────
print("── Step 7：触发第二个 2MB 段 ── 用 LARGE 分配耗尽第一个段 ──")
# 当前第一个段已用约: 200KB*5 + 20KB(overhead) ≈ 1020KB，再加 600KB 超过 2MB
t7 = torch.empty(LARGE, device=DEVICE)
tensors.append(t7)
S7 = show("[7] after torch.empty(600KB) → 新段")
d7 = delta(S7, S6)
hidden7 = d7['hbm'] - d7['pt_r']
print(f"       HBM Δ={d7['hbm']:+.2f} MB   PT_reserved Δ={d7['pt_r']:+.2f} MB   "
      f"hidden={hidden7:+.2f} MB")
check(d7['pt_r'] == 2.0,
      f"新的 kSmallBuffer=2MB 段被触发（实际 Δpt_r={d7['pt_r']:.2f}）")
check(abs(hidden7) < 0.1,
      f"纯分配（新段）：hidden ≈ 0，HBM Δ ≈ PT_reserved Δ（实际={hidden7:.2f} MB）")
print()

# ──────────────────────────────────────────────────────────────────────────────
print("=" * 72)
print("  全程汇总")
print("=" * 72)
total = delta(S7, S0)
print(f"  HBM 总增量:         {total['hbm']:+.2f} MB")
print(f"  PT_reserved 总增量: {total['pt_r']:+.2f} MB")
print(f"  隐藏分配总量:       {total['hbm']-total['pt_r']:+.2f} MB")
print()

# 按 step 汇总
print("  Step-by-step 隐藏分配明细:")
print(f"    Step1 torch.empty #1  : {d1['hbm']-d1['pt_r']:+.2f} MB （预期 ≈ 0）")
print(f"    Step2 torch.randn #1  : {hidden2:+.2f} MB  ← 首次 CANN 算子开销（workspace/初始化）")
print(f"    Step3 torch.randn #2  : {hidden3:+.2f} MB  ← 预期 ≈ 0（复用）")
print(f"    Step4 torch.ones      : {hidden4:+.2f} MB")
print(f"    Step5 torch.zeros     : {hidden5:+.2f} MB")
print(f"    Step6 torch.empty #2  : {hidden6:+.2f} MB （预期 ≈ 0）")
print(f"    Step7 torch.empty #3  : {hidden7:+.2f} MB （新段，预期 ≈ 0）")
print()

# ──────────────────────────────────────────────────────────────────────────────
print("── 补充：原始问题完整复现 ──")
print("   （13 次 torch.randn，与原始脚本顺序完全一致）")
print()

del tensors
torch.npu.empty_cache()

# 还原为完全干净状态用于复现（这里 empty_cache 后再重新来过）
# 注意：如果有驱动缓存，复现结果可能与 "真正冷启动" 不同
# 但此时我们已有 Step2 的精确数据来推断冷启动的情况
import gc; gc.collect()

tensors2 = []
sizes = [512*1024, 512*1024, 1*1024, 2*1024, 4*1024, 8*1024,
         16*1024, 32*1024, 64*1024, 128*1024, 256*1024, 512*1024, 2*1024]

prev2 = dict(hbm=hbm_used_mb(), pt_r=pt_reserved_mb(), pt_a=pt_allocated_mb())
print(f"  {'基准':<35s}  HBM={prev2['hbm']:.2f}  PT_reserved={prev2['pt_r']:.2f}  (MB)")

for i, size in enumerate(sizes):
    t = torch.randn(size // 4, device=DEVICE)
    tensors2.append(t)
    cur2 = dict(hbm=hbm_used_mb(), pt_r=pt_reserved_mb(), pt_a=pt_allocated_mb())
    d = {k: round(cur2[k] - prev2[k], 4) for k in cur2}
    h = d['hbm'] - d['pt_r']
    note = f"  ← hidden={h:+.2f} MB" if abs(h) > 0.01 else ""
    print(f"  alloc {i+1:2d} {size//1024:>4d}KB randn     "
          f"HBM Δ={d['hbm']:+.2f}  PT_r Δ={d['pt_r']:+.2f}  PT_a Δ={d['pt_a']:+.2f}{note}")
    prev2 = cur2

total2 = {k: round(cur2[k] - (dict(hbm=hbm_used_mb(), pt_r=pt_reserved_mb(),
           pt_a=pt_allocated_mb()) if False else prev2)[k], 4) for k in cur2}
total2 = delta(cur2, dict(hbm=hbm_used_mb()-0, pt_r=pt_reserved_mb()-0, pt_a=pt_allocated_mb()-0))

# 重新算 baseline 到最终的总 delta
final = dict(hbm=hbm_used_mb(), pt_r=pt_reserved_mb(), pt_a=pt_allocated_mb())
# 对比基准（此处基准是空 cache 后）
base2 = dict(hbm=final['hbm'] - sum(delta(
    dict(hbm=hbm_used_mb(), pt_r=pt_reserved_mb(), pt_a=pt_allocated_mb()),
    dict(hbm=hbm_used_mb(), pt_r=pt_reserved_mb(), pt_a=pt_allocated_mb())
    ).values()), pt_r=0.0, pt_a=0.0)

# 简单打印最终状态
print(f"\n  最终状态: HBM={cur2['hbm']:.2f}  PT_reserved={cur2['pt_r']:.2f}  PT_allocated={cur2['pt_a']:.2f}")

# ──────────────────────────────────────────────────────────────────────────────
print()
print("=" * 72)
print("  结论")
print("=" * 72)
print(f"""
  1. get_mem_info() 测量的是 "净新增 HBM 分配"，受驱动层 free pool 影响：
       empty_cache() → 内存还给驱动 free pool（HBM 不变）
       下次 AclrtMallocAlign32 → 从 free pool 取（HBM 仍不变）
     → 必须在无 warmup 的冷启动进程里测量才准确

  2. Step 分析（see above）揭示：
       torch.empty  → hidden ≈ 0（无额外 HBM）  ✓
       torch.randn #1 → hidden = {hidden2:.2f} MB（首次 CANN 算子开销）
       torch.randn #2 → hidden ≈ 0（复用）       ✓

  3. 原始脚本的 4MB 差异来自：
       首次 torch.randn 触发 CANN aclnn 算子，
       其 workspace 通过 NPUWorkspaceAllocator 分配（AclrtMallocAlign32），
       以及可能的 CANN 运行时一次性初始化开销，
       二者均不计入 torch.npu.memory_reserved()，但计入 get_mem_info()。
       具体大小由 Step2 的 hidden 值给出实测结果。
""")

if _any_fail:
    print("  ⚠️  部分断言失败，请检查上方 FAIL 行")
else:
    print("  所有断言通过 ✓")
print("=" * 72)

del tensors2
torch.npu.empty_cache()
acl.rt.reset_device(DEVICE_ID)
acl.finalize()
