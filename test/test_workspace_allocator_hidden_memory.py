"""
验证目标：
    证明 torch.npu.memory_reserved() 与 acl.rt.get_mem_info() 的差值
    来自 NPUWorkspaceAllocator，而非 NPUCachingAllocator。

实验逻辑：
    对照组 A：torch.empty()          ← 只触发 CachingAllocator，无算子 workspace
    对照组 B：torch.randn()          ← 触发 CachingAllocator + WorkspaceAllocator
    对照组 C：torch.empty() + fill_  ← 触发 CachingAllocator + WorkspaceAllocator（fill_ 也是算子）

    预期：A 中 HBM delta == PyTorch reserved delta
          B/C 中 HBM delta > PyTorch reserved delta（差值 = workspace）
"""

import sys
import os
import acl
import torch

# ─── ACL 初始化 ────────────────────────────────────────────────────────────────
os.environ["ASCEND_GLOBAL_LOG_LEVEL"] = "3"  # 只打 ERROR，避免刷屏

ret = acl.init()
assert ret == 0, f"acl.init() failed: {ret}"

DEVICE_ID = 0
ret = acl.rt.set_device(DEVICE_ID)
assert ret == 0, f"acl.rt.set_device() failed: {ret}"

DEVICE = torch.device(f"npu:{DEVICE_ID}")
ACL_MEM_TYPE_HBM = 0  # ACL_HBM_MEM


# ─── 工具函数 ──────────────────────────────────────────────────────────────────
def hbm_used_mb() -> float:
    free, total, ret = acl.rt.get_mem_info(ACL_MEM_TYPE_HBM)
    assert ret == 0
    return (total - free) / (1024 * 1024)


def pytorch_reserved_mb() -> float:
    return torch.npu.memory_reserved(DEVICE_ID) / (1024 * 1024)


def pytorch_allocated_mb() -> float:
    return torch.npu.memory_allocated(DEVICE_ID) / (1024 * 1024)


def full_reset():
    """释放所有 PyTorch 持有内存，回到干净状态。"""
    torch.npu.empty_cache()


def snapshot(label: str) -> dict:
    s = {
        "hbm_used":        hbm_used_mb(),
        "pt_reserved":     pytorch_reserved_mb(),
        "pt_allocated":    pytorch_allocated_mb(),
    }
    print(f"  [{label}]  HBM used={s['hbm_used']:.2f} MB  "
          f"PT reserved={s['pt_reserved']:.2f} MB  "
          f"PT allocated={s['pt_allocated']:.2f} MB")
    return s


def diff(after: dict, before: dict) -> dict:
    return {k: round(after[k] - before[k], 4) for k in after}


def check(condition: bool, msg: str):
    status = "PASS ✓" if condition else "FAIL ✗"
    print(f"    {status}  {msg}")
    if not condition:
        global _any_fail
        _any_fail = True


_any_fail = False

# ─── 主逻辑 ────────────────────────────────────────────────────────────────────

# 预热：让 PyTorch lazy init 全部完成，使初始状态干净
_warmup = torch.ones(1, device=DEVICE)
del _warmup
full_reset()
import time; time.sleep(0.2)

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("实验 A：torch.empty() — 只分配内存，不执行任何算子")
print("=" * 70)
full_reset()
before_A = snapshot("before")

NUM_ELEMENTS = (512 * 1024) // 4  # 512 KB / float32
t_A = torch.empty(NUM_ELEMENTS, device=DEVICE)

after_A = snapshot("after ")
d_A = diff(after_A, before_A)
print(f"  delta: HBM={d_A['hbm_used']:.2f} MB  PT_reserved={d_A['pt_reserved']:.2f} MB")

check(d_A["pt_reserved"] == 2.0,
      f"小分配(512KB)应触发 kSmallBuffer=2MB 段: pt_reserved delta={d_A['pt_reserved']:.2f} MB")
check(abs(d_A["hbm_used"] - d_A["pt_reserved"]) < 0.1,
      "torch.empty 无算子 workspace：HBM delta 应 ≈ PT reserved delta"
      f"（差={d_A['hbm_used'] - d_A['pt_reserved']:.2f} MB）")

del t_A

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("实验 B：torch.randn() — 分配 + 执行随机算子（有 workspace）")
print("=" * 70)
full_reset()
before_B = snapshot("before")

t_B = torch.randn(NUM_ELEMENTS, device=DEVICE)

after_B = snapshot("after ")
d_B = diff(after_B, before_B)
print(f"  delta: HBM={d_B['hbm_used']:.2f} MB  PT_reserved={d_B['pt_reserved']:.2f} MB")
workspace_B = d_B["hbm_used"] - d_B["pt_reserved"]
print(f"  推断 WorkspaceAllocator 占用: {workspace_B:.2f} MB")

check(d_B["pt_reserved"] == 2.0,
      f"小分配(512KB)应触发 kSmallBuffer=2MB 段: pt_reserved delta={d_B['pt_reserved']:.2f} MB")
check(workspace_B > 0,
      f"torch.randn 应有额外 HBM（WorkspaceAllocator）：差值={workspace_B:.2f} MB")
check(workspace_B % 2.0 == 0,
      f"WorkspaceAllocator 按 kRoundLarge=2MB 对齐：{workspace_B:.2f} MB 应为 2 的倍数")

del t_B

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("实验 C：torch.empty() + fill_() — 分开的内存分配与算子执行")
print("=" * 70)
full_reset()
before_C = snapshot("before")

t_C = torch.empty(NUM_ELEMENTS, device=DEVICE)
mid_C = snapshot("after empty ")
d_C_empty = diff(mid_C, before_C)

t_C.fill_(1.0)  # 触发 fill_ 算子，会调用 WorkspaceAllocator
after_C = snapshot("after fill_")
d_C_fill = diff(after_C, mid_C)
print(f"  empty delta: HBM={d_C_empty['hbm_used']:.2f} MB  PT_reserved={d_C_empty['pt_reserved']:.2f} MB")
print(f"  fill_ delta: HBM={d_C_fill['hbm_used']:.2f} MB  PT_reserved={d_C_fill['pt_reserved']:.2f} MB")
workspace_C = d_C_fill["hbm_used"]
print(f"  推断 fill_ WorkspaceAllocator 占用: {workspace_C:.2f} MB")

check(abs(d_C_empty["hbm_used"] - d_C_empty["pt_reserved"]) < 0.1,
      "torch.empty 阶段：HBM delta ≈ PT reserved delta（无 workspace）"
      f"（差={d_C_empty['hbm_used'] - d_C_empty['pt_reserved']:.2f} MB）")
check(d_C_fill["pt_reserved"] == 0,
      f"fill_() 不触发新的 CachingAllocator 分配：PT reserved delta={d_C_fill['pt_reserved']:.2f} MB")
# fill_ 可能 workspace=0（某些硬件上），也可能 >0
print(f"    INFO  fill_() workspace={'存在' if workspace_C > 0 else '为零（fill_ 无需 workspace）'}: {workspace_C:.2f} MB")

del t_C

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("实验 D：连续两次 torch.randn()，验证 workspace 是否复用")
print("=" * 70)
full_reset()
before_D = snapshot("before")

t_D1 = torch.randn(NUM_ELEMENTS, device=DEVICE)
mid_D = snapshot("after 1st randn")
d_D1 = diff(mid_D, before_D)

t_D2 = torch.randn(NUM_ELEMENTS, device=DEVICE)  # 同 stream，workspace 应复用
after_D = snapshot("after 2nd randn")
d_D2 = diff(after_D, mid_D)

print(f"  1st randn delta: HBM={d_D1['hbm_used']:.2f} MB  PT_reserved={d_D1['pt_reserved']:.2f} MB")
print(f"  2nd randn delta: HBM={d_D2['hbm_used']:.2f} MB  PT_reserved={d_D2['pt_reserved']:.2f} MB")

check(d_D1["hbm_used"] > d_D1["pt_reserved"],
      f"1st randn：HBM delta({d_D1['hbm_used']:.2f}) > PT reserved delta({d_D1['pt_reserved']:.2f})，workspace 已分配")
check(abs(d_D2["hbm_used"] - d_D2["pt_reserved"]) < 0.1,
      f"2nd randn（同 stream）：workspace 复用，HBM delta ≈ PT reserved delta"
      f"（差={d_D2['hbm_used'] - d_D2['pt_reserved']:.2f} MB）")

del t_D1, t_D2

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("实验 E：empty_cache 同时释放 WorkspaceAllocator（验证回收路径）")
print("=" * 70)
full_reset()

t_E = torch.randn(NUM_ELEMENTS, device=DEVICE)
before_E = snapshot("after randn（持有张量）")

del t_E
torch.npu.empty_cache()
after_E = snapshot("after del+empty_cache")
d_E = diff(after_E, before_E)
print(f"  delta（释放后）: HBM={d_E['hbm_used']:.2f} MB  PT_reserved={d_E['pt_reserved']:.2f} MB")

check(d_E["pt_reserved"] <= 0,
      f"empty_cache 应释放所有 CachingAllocator 缓存：PT reserved delta={d_E['pt_reserved']:.2f} MB")
# WorkspaceAllocator 也应随 empty_cache 释放
# 注意：emptyCache 调用链: NpuCachingAllocator::emptyCache → NPUWorkspaceAllocator::emptyCache
workspace_released = -d_E["hbm_used"] - (-d_E["pt_reserved"])
print(f"  WorkspaceAllocator 随 empty_cache 释放: {-d_E['hbm_used']:.2f} MB HBM 回收，"
      f"其中 {-d_E['pt_reserved']:.2f} MB 来自 CachingAllocator，"
      f"{workspace_released:.2f} MB 来自 WorkspaceAllocator")
check(-d_E["hbm_used"] >= -d_E["pt_reserved"],
      f"HBM 回收量 ≥ PT reserved 回收量（WorkspaceAllocator 也被回收）")

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("实验 F：复现原始问题 — 与用户脚本完全相同的分配序列")
print("=" * 70)
full_reset()
sizes = [512*1024, 512*1024, 1*1024, 2*1024, 4*1024, 8*1024,
         16*1024, 32*1024, 64*1024, 128*1024, 256*1024, 512*1024, 2*1024]

baseline = snapshot("baseline")
tensors = []
prev = baseline
for i, size in enumerate(sizes):
    t = torch.randn(size // 4, device=DEVICE)
    tensors.append(t)
    cur = snapshot(f"alloc {i+1:2d}  {size//1024:>4d}KB")
    d = diff(cur, prev)
    hidden = d["hbm_used"] - d["pt_reserved"]
    if abs(hidden) > 0.01:
        print(f"    ↑ WorkspaceAllocator 贡献: {hidden:.2f} MB (仅首次分配可见)")
    prev = cur

total_d = diff(prev, baseline)
print(f"\n  总计：HBM={total_d['hbm_used']:.2f} MB  PT_reserved={total_d['pt_reserved']:.2f} MB  "
      f"WorkspaceAllocator={total_d['hbm_used']-total_d['pt_reserved']:.2f} MB")
check(total_d["pt_reserved"] == 4.0,
      f"2个2MB段：PT reserved total delta=4.00 MB（实际={total_d['pt_reserved']:.2f} MB）")
check(abs((total_d["hbm_used"] - total_d["pt_reserved"]) - 4.0) < 0.5,
      f"WorkspaceAllocator 应贡献约 4 MB（实际={total_d['hbm_used']-total_d['pt_reserved']:.2f} MB）")

del tensors
full_reset()

# ─── 汇总 ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
if _any_fail:
    print("结论：部分断言失败，请检查上方 FAIL 行")
else:
    print("结论：所有断言通过 ✓")
    print()
    print("  消失的 4 MB = NPUWorkspaceAllocator 为 torch.randn() 底层")
    print("  aclnn 算子分配的 workspace，按 kRoundLarge=2MB 对齐，")
    print("  通过 AclrtMallocAlign32 直接占用 HBM，但不计入")
    print("  torch.npu.memory_reserved() / memory_allocated() 统计。")
print("=" * 70)

# ─── 清理 ─────────────────────────────────────────────────────────────────────
ret = acl.rt.reset_device(DEVICE_ID)
ret = acl.finalize()
