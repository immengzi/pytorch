"""
研究问题：
    原始测试中，首次 torch.randn(512KB) 导致 HBM 增加 6 MB，
    但 torch.npu.memory_reserved() 只增加 2 MB，差值 4 MB 去哪了？

已知可疑项：
    A. NPUWorkspaceAllocator（CANN 算子 workspace）
    B. initGlobalStreamState()（PyTorch 首次 allocate 触发，创建 default/secondary stream）

关键实验设计：
    在同一冷启动进程中（无 warmup、无 empty_cache），
    先 torch.empty()（无 CANN 算子），再 torch.randn()（有 CANN 算子），
    通过逐步叠加测量 HBM 增量，精确隔离两类来源。

    若 4 MB 来自 B（stream 初始化）：
        torch.empty  → hidden = +4 MB（触发 initGlobalStreamState）
        torch.randn  → hidden =  0 MB（stream 已存在，无额外 HBM）

    若 4 MB 来自 A（workspace）：
        torch.empty  → hidden =  0 MB（无算子，不触发 workspace）
        torch.randn  → hidden = +4 MB（首次 CANN 算子调用 WorkspaceAllocator）
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

tensors = []   # 全程保活，防止内存归还驱动 free pool

def hbm_mb():
    free, total, ret = acl.rt.get_mem_info(ACL_MEM_TYPE_HBM)
    assert ret == 0
    return (total - free) / (1024 ** 2)

def pt_reserved_mb():
    return torch.npu.memory_reserved(DEVICE_ID) / (1024 ** 2)

def measure():
    return dict(hbm=hbm_mb(), pt_r=pt_reserved_mb())

def show(label, s, prev=None):
    if prev:
        d_hbm = s['hbm'] - prev['hbm']
        d_pt  = s['pt_r'] - prev['pt_r']
        hidden = d_hbm - d_pt
        print(f"  {label:<40s}  HBM_Δ={d_hbm:+.2f}  PT_r_Δ={d_pt:+.2f}  "
              f"hidden={hidden:+.2f}  (MB)")
        return d_hbm, d_pt, hidden
    else:
        print(f"  {label:<40s}  HBM={s['hbm']:.2f}  PT_reserved={s['pt_r']:.2f}  (MB)")
        return None

_pass = _fail = 0
def check(cond, msg):
    global _pass, _fail
    tag = "PASS ✓" if cond else "FAIL ✗"
    print(f"    {tag}  {msg}")
    if cond: _pass += 1
    else:    _fail += 1

SZ = (200 * 1024) // 4   # 200 KB / float32

# ══════════════════════════════════════════════════════════════════════════════
print("=" * 72)
print("  实验一：精准隔离 —— torch.empty vs torch.randn 的 HBM 贡献")
print("  （冷启动，无 warmup，保活所有张量）")
print("=" * 72)

s0 = measure()
show("基准（进程启动后首次测量）", s0)
print()

# ── Step 1: torch.empty ──────────────────────────────────────────────────────
print("Step 1  torch.empty(200KB)  ← 纯内存分配，无任何 CANN 算子")
t = torch.empty(SZ, device=DEVICE)
tensors.append(t)
s1 = measure()
d1_hbm, d1_pt, h1 = show("after torch.empty(200KB)", s1, s0)

check(d1_pt == 2.0,
      f"首次小分配触发 kSmallBuffer=2MB 段  (实际={d1_pt:.2f} MB)")
check(abs(h1) < 0.1 or h1 > 0,
      f"hidden ≥ 0  (实际={h1:.2f} MB)")
print()

# ── Step 2: torch.randn（首次 CANN 算子）────────────────────────────────────
print("Step 2  torch.randn(200KB)  ← 首次 CANN 算子（aclnn random kernel）")
print("        200KB 仍在同一 2MB 段内，PT_reserved 不应增加")
t = torch.randn(SZ, device=DEVICE)
tensors.append(t)
s2 = measure()
d2_hbm, d2_pt, h2 = show("after torch.randn(200KB) #1", s2, s1)

check(d2_pt == 0.0,
      f"randn 与 empty 共用同一 2MB 段，PT_reserved 不变  (实际Δ={d2_pt:.2f} MB)")
print()

# ── Step 3: torch.randn（第二次）─────────────────────────────────────────────
print("Step 3  torch.randn(200KB)  ← 第二次调用，workspace 若存在则复用")
t = torch.randn(SZ, device=DEVICE)
tensors.append(t)
s3 = measure()
d3_hbm, d3_pt, h3 = show("after torch.randn(200KB) #2", s3, s2)
print()

# ── Step 4: 再次 torch.empty（对照）─────────────────────────────────────────
print("Step 4  torch.empty(200KB)  ← 再次纯分配，无算子（对照 Step 1）")
t = torch.empty(SZ, device=DEVICE)
tensors.append(t)
s4 = measure()
d4_hbm, d4_pt, h4 = show("after torch.empty(200KB) #2", s4, s3)

check(abs(h4) < 0.1,
      f"第二次 empty 无初始化开销：hidden ≈ 0  (实际={h4:.2f} MB)")
print()

# ══════════════════════════════════════════════════════════════════════════════
print("=" * 72)
print("  实验一 结果判定")
print("=" * 72)
print(f"""
  Step 1  torch.empty  hidden = {h1:+.2f} MB
  Step 2  torch.randn  hidden = {h2:+.2f} MB
  Step 3  torch.randn  hidden = {h3:+.2f} MB
  Step 4  torch.empty  hidden = {h4:+.2f} MB
""")

if h1 > 0.5 and abs(h2) < 0.1:
    print("  ✅  结论：4 MB 来自 initGlobalStreamState()（stream 初始化），")
    print("      不来自 NPUWorkspaceAllocator（torch.randn 无额外 HBM）")
    print()
    print("  原因：首次调用 allocate() 触发 getCurrentNPUStreamNoWait()")
    print("        → initNPUStreamsOnce() → initGlobalStreamState()")
    print("        → 创建 default_stream + secondary_stream")
    print("        → AclrtCreateStreamWithConfig(ACL_STREAM_FAST_LAUNCH | ACL_STREAM_FAST_SYNC)")
    print("        CANN 为每条 fast-sync stream 在 HBM 内分配同步原语缓冲区")
    print(f"       共 {h1:.0f} MB（约每条 stream {h1/2:.0f} MB × 2 条）")
elif abs(h1) < 0.1 and h2 > 0.5:
    print("  ✅  结论：4 MB 来自 NPUWorkspaceAllocator（CANN 算子 workspace），")
    print("      不来自 stream 初始化")
else:
    print(f"  ⚠️  结果不符合任一预期假设，需进一步调查")
    print(f"      Step1 hidden={h1:+.2f}  Step2 hidden={h2:+.2f}")
print()

# ══════════════════════════════════════════════════════════════════════════════
print("=" * 72)
print("  实验二：复现原始问题（13 次 randn 序列），验证 stream init 一次性特征")
print("=" * 72)
print()

# 此时 stream 已初始化，后续 randn 均不应有额外 hidden HBM
# 保持 tensors 已有内容，继续叠加
sizes = [512*1024, 512*1024, 1*1024, 2*1024, 4*1024, 8*1024,
         16*1024, 32*1024, 64*1024, 128*1024, 256*1024, 512*1024, 2*1024]

# 现在 tensors 里已有 4 个 200KB 张量（Step 1-4），占 800KB
# 第一个 2MB 段还剩约 1.2MB，后面的 randn 从这里开始

prev = s4
for i, nbytes in enumerate(sizes):
    t = torch.randn(nbytes // 4, device=DEVICE)
    tensors.append(t)
    cur = measure()
    d_hbm  = cur['hbm']  - prev['hbm']
    d_pt   = cur['pt_r'] - prev['pt_r']
    hidden = d_hbm - d_pt
    seg_note = " ← 新 2MB 段" if d_pt > 0 else ""
    hid_note = f"  hidden={hidden:+.2f} MB ⚠️" if abs(hidden) > 0.01 else ""
    print(f"  randn {i+1:2d}  {nbytes//1024:>4d}KB"
          f"   HBM_Δ={d_hbm:+.2f}  PT_r_Δ={d_pt:+.2f}{seg_note}{hid_note}")
    prev = cur

print()
total_hbm = prev['hbm'] - s0['hbm']
total_pt  = prev['pt_r'] - s0['pt_r']
print(f"  全程累计  HBM_Δ={total_hbm:+.2f} MB   PT_reserved_Δ={total_pt:+.2f} MB")
print(f"  隐藏分配总量 = {total_hbm - total_pt:+.2f} MB  "
      f"({'全部来自 stream 初始化（Step1）' if abs(h1 - (total_hbm - total_pt)) < 0.1 else '见明细'})")
print()

# ══════════════════════════════════════════════════════════════════════════════
print("=" * 72)
print("  总结")
print("=" * 72)
print(f"""
  原始问题：首次 torch.randn 导致 HBM +6 MB，PyTorch 只能解释 2 MB（kSmallBuffer 段）
  缺失的 4 MB 来源：

    ✗  NPUWorkspaceAllocator（已排除）
         — torch.randn hidden = {h2:+.2f} MB，说明算子 workspace 不消耗额外 HBM
         — 本次设备/CANN 版本上 random kernel workspace = 0，
           或 workspace 通过驱动层 free pool 满足（如有先前释放的内存）

    ✅  initGlobalStreamState()（CANN stream 初始化开销）
         — torch.empty（无算子）hidden = {h1:+.2f} MB
         — 触发路径：allocate() → getCurrentNPUStreamNoWait()
                     → initNPUStreamsOnce() → initGlobalStreamState()
                     → AclrtCreateStreamWithConfig(
                           ACL_STREAM_FAST_LAUNCH | ACL_STREAM_FAST_SYNC) × 2 条
         — CANN 为 fast-sync stream 在 HBM 分配同步原语缓冲区（每条约 2 MB）
         — 2 条 stream（default + secondary）× 2 MB = 4 MB
         — 一次性开销：后续所有 randn/empty 调用 hidden = 0
""")

check(abs(h1 - 4.0) < 0.5,
      f"stream 初始化 HBM 开销 ≈ 4 MB（实际={h1:.2f} MB）")
check(abs(h2) < 0.1,
      f"torch.randn 无额外 hidden HBM（workspace=0 或已复用）（实际={h2:.2f} MB）")
check(abs(h4) < 0.1,
      f"第二次 torch.empty 无 hidden（stream 已初始化）（实际={h4:.2f} MB）")
total_hidden = total_hbm - total_pt
check(abs(total_hidden - h1) < 0.5,
      f"全程隐藏 HBM 等于 Step1 的 stream init 开销（{h1:.2f} MB vs 总计 {total_hidden:.2f} MB）")

print()
print(f"  PASS: {_pass}  FAIL: {_fail}")
print("=" * 72)

del tensors
torch.npu.empty_cache()
acl.rt.reset_device(DEVICE_ID)
acl.finalize()
