# NPU 内存分配详解：块划分、round_size 与隐藏开销

## 一、测试场景

顺序分配 13 个张量，大小如下（`torch.randn`，NPU device）：

```
# 按字节
sizes = [512KB, 512KB, 1KB, 2KB, 4KB, 8KB, 16KB, 32KB, 64KB, 128KB, 256KB, 512KB, 2KB]
```

设备：Atlas A2 训练系列，torch-npu **2.8.0**，`kMinBlockSize = 512 B`。

---

## 二、round_size() 机制：为什么多出 512 字节

### 2.1 v2.8.0 实现

```cpp
static size_t round_size(size_t size)
{
    size = size + 32;                         // 固定 +32 字节 padding
    if (size < kMinBlockSize) {
        return kMinBlockSize;
    } else {
        return kMinBlockSize * ((size + kMinBlockSize - 1) / kMinBlockSize);
    }
}
```

- `kMinBlockSize = 512`（对齐单位）
- 等价公式：`round_size(n) = ⌈(n + 32) / 512⌉ × 512`

### 2.2 为什么恰好多出 512 字节

凡请求大小是 512 的整数倍（如 1 KB、2 KB、4 KB、… 512 KB），加 32 字节后变为 512 的非整数倍，必须向上对齐到下一个 512 边界，固定额外增加 `512 − 32 = 480` 字节的对齐余量，加上 32 字节 padding，总开销 = **512 字节（= 1 × kMinBlockSize）**。

```
例：请求 512 KB = 524288 字节
  +32  →  524320 字节
  ÷512 →  1024.0625（非整数）
  向上取整 → 1025
  ×512 →  524800 字节  （比请求多 512 字节）
```

### 2.3 v2.11.0 实现（新老芯片分支）

```cpp
static size_t round_size(size_t size)
{
    size += AddPadSize();   // 新芯片(≥910_95)：+0；老芯片：+32
    if (size < kMinBlockSize) {
        return kMinBlockSize;
    } else {
        auto divisions = CachingAllocatorConfig::roundup_power2_divisions(size);
        if (divisions > 1 && size > (kMinBlockSize * divisions)) {
            return roundup_power2_next_division(size, divisions);
        } else {
            return kMinBlockSize * ((size + kMinBlockSize - 1) / kMinBlockSize);
        }
    }
}

size_t AddPadSize()
{
    static size_t add_size = -1;
    if (add_size == -1) {
        if (GetSocVersion() >= SocVersion::Ascend910_95) {
            add_size = 0;   // 新芯片：不加 padding
        } else {
            add_size = 32;  // 老芯片：保留 32 字节
        }
    }
    return add_size;
}
```

| 芯片 | padding | 额外开销（512B 对齐请求） |
|---|---|---|
| Ascend 910 B / 老系列 | +32 字节 | +512 字节 |
| Ascend 910_95 及以上 | +0 字节 | 0 字节 |

---

## 三、段（Segment）分配策略

`get_allocation_size()` 决定向驱动申请的段大小：

```cpp
// NPUAllocatorConfig.h
constexpr size_t kSmallSize   = 1048576;   // 1 MB：小/大分配分界线
constexpr size_t kSmallBuffer = 2097152;   // 2 MB：小分配段大小
constexpr size_t kLargeBuffer = 20971520;  // 20 MB：中等分配段大小

static size_t get_allocation_size(size_t size) {
    if (size <= kSmallSize)          // ≤ 1 MB → 小池（Small Pool）
        return kSmallBuffer;         // 向驱动申请 2 MB 段
    else if (size < kMinLargeAlloc)  // 1–10 MB → 大池（Large Pool）
        return kLargeBuffer;         // 向驱动申请 20 MB 段
    else
        return kRoundLarge * ((size + kRoundLarge - 1) / kRoundLarge);
}
```

本次测试的所有 13 个分配请求（原始大小）均 ≤ 1 MB，全部走**小池（Small Pool）**，段大小固定为 **2 MB**。

> **注意**：`get_allocation_size()` 的输入是原始请求大小，`round_size()` 之后的大小用于在段内寻找合适的空闲块并确定块边界，二者作用不同。
>
> 虽然 round_size(1 MB) = 1 MB + 32 B → 向上对齐后 **> 1 MB**，实际上触发了大池分配（20 MB 段）。本例中最大请求为 512 KB，不触发该边界。

---

## 四、逐次分配详解

### 4.1 round_size 计算表

| # | 请求大小 | 加 32 B 后 | round_size 后 | 段内额外开销 |
|--:|--:|--:|--:|--:|
| 1 | 524,288 B (512 KB) | 524,320 B | **524,800 B** | +512 B |
| 2 | 524,288 B (512 KB) | 524,320 B | **524,800 B** | +512 B |
| 3 | 1,024 B (1 KB) | 1,056 B | **1,536 B** | +512 B |
| 4 | 2,048 B (2 KB) | 2,080 B | **2,560 B** | +512 B |
| 5 | 4,096 B (4 KB) | 4,128 B | **4,608 B** | +512 B |
| 6 | 8,192 B (8 KB) | 8,224 B | **8,704 B** | +512 B |
| 7 | 16,384 B (16 KB) | 16,416 B | **16,896 B** | +512 B |
| 8 | 32,768 B (32 KB) | 32,800 B | **33,280 B** | +512 B |
| 9 | 65,536 B (64 KB) | 65,568 B | **66,048 B** | +512 B |
| 10 | 131,072 B (128 KB) | 131,104 B | **131,584 B** | +512 B |
| 11 | 262,144 B (256 KB) | 262,176 B | **262,656 B** | +512 B |
| 12 | 524,288 B (512 KB) | 524,320 B | **524,800 B** | +512 B |
| 13 | 2,048 B (2 KB) | 2,080 B | **2,560 B** | +512 B |

每个分配额外多占 512 字节（= 1 × kMinBlockSize）。

### 4.2 块（Block）布局

段 1（Segment 1）基地址：`0x000012c041200000`

| # | 段内偏移（起→止）| 字节数 | 张量地址（实测） |
|--:|---|--:|---|
| 1 | 0 → 524,800 | 524,800 | `0x000012c041200000` |
| 2 | 524,800 → 1,049,600 | 524,800 | `0x000012c041280200` |
| 3 | 1,049,600 → 1,051,136 | 1,536 | `0x000012c041300400` |
| 4 | 1,051,136 → 1,053,696 | 2,560 | `0x000012c041300a00` |
| 5 | 1,053,696 → 1,058,304 | 4,608 | `0x000012c041301400` |
| 6 | 1,058,304 → 1,067,008 | 8,704 | `0x000012c041302600` |
| 7 | 1,067,008 → 1,083,904 | 16,896 | `0x000012c041304800` |
| 8 | 1,083,904 → 1,117,184 | 33,280 | `0x000012c041308a00` |
| 9 | 1,117,184 → 1,183,232 | 66,048 | `0x000012c041310c00` |
| 10 | 1,183,232 → 1,314,816 | 131,584 | `0x000012c041320e00` |
| 11 | 1,314,816 → 1,577,472 | 262,656 | `0x000012c041341000` |
| — | 1,577,472 → 2,097,152 | **519,680（空闲）** | — |
| 13 | 1,577,472 → 1,580,032 | 2,560 | `0x000012c041381200` |

段 2（Segment 2）基地址：`0x000012c041400000`

| # | 段内偏移（起→止）| 字节数 | 张量地址（实测） |
|--:|---|--:|---|
| 12 | 0 → 524,800 | 524,800 | `0x000012c041400000` |
| — | 524,800 → 2,097,152 | **1,572,352（空闲）** | — |

### 4.3 为什么分配 12 触发新段

分配 11 结束后，段 1 已用 1,577,472 字节：

```
段 1 剩余空间 = 2,097,152 − 1,577,472 = 519,680 字节
分配 12 需要  = round_size(512KB) = 524,800 字节
524,800 > 519,680  →  段 1 放不下  →  向驱动申请新段（段 2，2 MB）
```

### 4.4 为什么分配 13 回到段 1

分配 13 发生时，空闲块池中有两个候选：

```
段 1 空闲块：519,680 字节（起于偏移 1,577,472）
段 2 空闲块：1,572,352 字节（起于偏移 524,800）

分配 13 需要 = round_size(2KB) = 2,560 字节
```

两者都足够容纳 2,560 字节。NPUCachingAllocator 的小池使用**最小适配（best-fit）**策略：优先选取大小最接近需求的空闲块，以减少碎片。段 1 的 519,680 字节空闲块小于段 2 的 1,572,352 字节，因此分配 13 回到段 1，地址 = `0x000012c041200000 + 1,577,472 = 0x000012c041381200` ✓

---

## 五、内存统计逐步演进

| # | HBM 实际占用 | HBM Δ | PT reserved | PT reserved Δ | PT allocated |
|--:|--:|--:|--:|--:|--:|
| 初始 | 396.16 MB | — | 0 MB | — | 0 MB |
| 1 | 402.16 MB | **+6.00 MB** | 2.00 MB | +2.00 MB | 0.50 MB |
| 2 | 402.16 MB | 0 | 2.00 MB | 0 | 1.00 MB |
| 3 | 402.16 MB | 0 | 2.00 MB | 0 | 1.00 MB |
| 4 | 402.16 MB | 0 | 2.00 MB | 0 | 1.00 MB |
| 5 | 402.16 MB | 0 | 2.00 MB | 0 | 1.01 MB |
| 6 | 402.16 MB | 0 | 2.00 MB | 0 | 1.02 MB |
| 7 | 402.16 MB | 0 | 2.00 MB | 0 | 1.03 MB |
| 8 | 402.16 MB | 0 | 2.00 MB | 0 | 1.07 MB |
| 9 | 402.16 MB | 0 | 2.00 MB | 0 | 1.13 MB |
| 10 | 402.16 MB | 0 | 2.00 MB | 0 | 1.25 MB |
| 11 | 402.16 MB | 0 | 2.00 MB | 0 | 1.50 MB |
| 12 | 404.16 MB | **+2.00 MB** | 4.00 MB | +2.00 MB | 2.00 MB |
| 13 | 404.16 MB | 0 | 4.00 MB | 0 | 2.01 MB |

**分配 1 触发 +6 MB**（而非 +2 MB）的分解：

```
+4 MB  ← initGlobalStreamState() 创建 2 条 ACL_STREAM_FAST_LAUNCH stream，
          CANN 在 stream 创建时预申请 HBM 内部资源（≈2 MB × 2），
          完全不经过 torch-npu 分配器，对 PT 统计不可见
+2 MB  ← alloc_block() → AclrtMallocAlign32(kSmallBuffer = 2MB)，
          段 1 创建，PT reserved +2 MB
```

### 5.1 PT allocated 统计口径

`torch.npu.memory_allocated()` 记录的是 `round_size()` 之后的块大小之和，**不是**原始张量字节数。

验证：
```
round_size 累计 = 524800×2 + 1536 + 2560 + 4608 + 8704 + 16896 +
                  33280 + 66048 + 131584 + 262656 + 524800 + 2560
               = 2,104,832 字节 = 2.0078 MB → 显示为 2.01 MB ✓
```

---

## 六、完整内存汇总

| 来源 | 字节 | 对 PT 可见 | 对 get_mem_info 可见 |
|---|---|---|---|
| CANN stream 内部资源（ACL_STREAM_FAST_LAUNCH × 2） | ≈4 MB | ✗ | ✓ |
| 段 1（NPUCachingAllocator kSmallBuffer） | 2 MB | ✓（reserved） | ✓ |
| 段 2（NPUCachingAllocator kSmallBuffer） | 2 MB | ✓（reserved） | ✓ |
| **合计 HBM** | **≈8 MB** | — | — |
| 其中 PT reserved | 4 MB | ✓ | ✓ |
| 持久差值（HBM − reserved） | **4 MB** | ✗ | ✓ |

---

## 七、测试代码补充说明

原始脚本功能正确，以下一点值得注意：

**清理顺序问题**：当前代码在 `input()` 之后先调用 `acl.rt.reset_device()`，再执行 `del tensors` 和 `torch.npu.empty_cache()`。`reset_device()` 应在所有 NPU 资源释放之后调用，建议改为：

```python
# 建议的清理顺序
del tensors
import gc; gc.collect()
torch.npu.empty_cache()          # 归还 cached memory 到驱动
acl.rt.reset_device(device_id)  # 最后重置设备
acl.finalize()
```

此调整不影响内存测量阶段的正确性，仅对进程退出的安全性有意义。
