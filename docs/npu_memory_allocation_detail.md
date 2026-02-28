# NPU 内存分配完整分析：分段策略、块布局与隐藏开销

## 一、测试场景

顺序分配 13 个张量（`torch.randn`，NPU device），大小如下：

```
sizes = [512KB, 512KB, 1KB, 2KB, 4KB, 8KB, 16KB, 32KB, 64KB, 128KB, 256KB, 512KB, 2KB]
```

设备：Atlas A2 训练系列，torch-npu **2.8.0**，`kMinBlockSize = 512 B`。

实测输出：

```
初始 HBM used：396.16 MB   PT allocated：0.00 MB   PT reserved：0.00 MB

分配  1（512KB）：HBM used=402.16 MB (+6.00)   PT allocated=0.50 MB   PT reserved=2.00 MB
分配  2（512KB）：HBM used=402.16 MB (+0.00)   PT allocated=1.00 MB   PT reserved=2.00 MB
分配  3（  1KB）：HBM used=402.16 MB (+0.00)   PT allocated=1.00 MB   PT reserved=2.00 MB
...（中间分配省略，HBM/PT reserved 均无变化）
分配 12（512KB）：HBM used=404.16 MB (+2.00)   PT allocated=2.00 MB   PT reserved=4.00 MB
分配 13（  2KB）：HBM used=404.16 MB (+0.00)   PT allocated=2.01 MB   PT reserved=4.00 MB

最终 HBM used：404.16 MB，PT reserved：4.00 MB，差值：4.00 MB
```

---

## 二、段（Segment）分配策略

`NPUCachingAllocator` 不对每次分配单独向驱动申请内存，而是以**段（Segment）**为单位批量申请，段大小由 `get_allocation_size()` 决定：

```cpp
// NPUCachingAllocator.cpp:2426-2435
static size_t get_allocation_size(size_t size) {
    if (size <= kSmallSize)          // ≤ 1 MB → 小池（Small Pool）
        return kSmallBuffer;         // 向驱动申请 2 MB 段
    else if (size < kMinLargeAlloc)  // 1–10 MB → 大池（Large Pool）
        return kLargeBuffer;         // 向驱动申请 20 MB 段
    else
        return kRoundLarge * ((size + kRoundLarge - 1) / kRoundLarge);
}
```

相关常量（`NPUAllocatorConfig.h`）：

```cpp
constexpr size_t kSmallSize   = 1048576;   // 1 MB：小/大分配分界线
constexpr size_t kSmallBuffer = 2097152;   // 2 MB：小分配段大小
constexpr size_t kLargeBuffer = 20971520;  // 20 MB：中等分配段大小
```

本次测试的所有 13 个请求均 ≤ 512 KB（< 1 MB），全部走**小池**，每次向驱动申请的段大小固定为 **2 MB**。

> `get_allocation_size()` 的输入是原始请求大小（用于判断走哪个池、申请多大的段）。`round_size()` 的输出（见第四节）用于在段内划分块边界，二者作用不同。

---

## 三、逐次分配详解

### 3.1 块（Block）布局

段 1（Segment 1）基地址：`0x000012c041200000`，由分配 1 触发创建。

| # | 请求大小 | round_size 后 | 段内偏移（起 → 止）| 张量地址（实测）|
|--:|--:|--:|---|---|
| 1 | 512 KB | 524,800 B | 0 → 524,800 | `0x000012c041200000` |
| 2 | 512 KB | 524,800 B | 524,800 → 1,049,600 | `0x000012c041280200` |
| 3 | 1 KB | 1,536 B | 1,049,600 → 1,051,136 | `0x000012c041300400` |
| 4 | 2 KB | 2,560 B | 1,051,136 → 1,053,696 | `0x000012c041300a00` |
| 5 | 4 KB | 4,608 B | 1,053,696 → 1,058,304 | `0x000012c041301400` |
| 6 | 8 KB | 8,704 B | 1,058,304 → 1,067,008 | `0x000012c041302600` |
| 7 | 16 KB | 16,896 B | 1,067,008 → 1,083,904 | `0x000012c041304800` |
| 8 | 32 KB | 33,280 B | 1,083,904 → 1,117,184 | `0x000012c041308a00` |
| 9 | 64 KB | 66,048 B | 1,117,184 → 1,183,232 | `0x000012c041310c00` |
| 10 | 128 KB | 131,584 B | 1,183,232 → 1,314,816 | `0x000012c041320e00` |
| 11 | 256 KB | 262,656 B | 1,314,816 → 1,577,472 | `0x000012c041341000` |
| — | — | — | 1,577,472 → 2,097,152（**519,680 B 空闲**）| — |
| 13 | 2 KB | 2,560 B | 1,577,472 → 1,580,032 | `0x000012c041381200` |

段 2（Segment 2）基地址：`0x000012c041400000`，由分配 12 触发创建。

| # | 请求大小 | round_size 后 | 段内偏移（起 → 止）| 张量地址（实测）|
|--:|--:|--:|---|---|
| 12 | 512 KB | 524,800 B | 0 → 524,800 | `0x000012c041400000` |
| — | — | — | 524,800 → 2,097,152（**1,572,352 B 空闲**）| — |

### 3.2 为什么分配 12 触发新段

分配 11 结束后，段 1 已累计使用 1,577,472 字节：

```
段 1 剩余  = 2,097,152 − 1,577,472 = 519,680 字节
分配 12 需 = round_size(512KB) = 524,800 字节

524,800 > 519,680  →  段 1 放不下  →  向驱动 AclrtMallocAlign32(2MB)，创建段 2
```

### 3.3 为什么分配 13 回到段 1

分配 13 发生时，空闲块池中有两个候选：

```
段 1 空闲块：519,680 字节（起于偏移 1,577,472）
段 2 空闲块：1,572,352 字节（起于偏移 524,800）

分配 13 需 = round_size(2KB) = 2,560 字节
```

两者均可容纳 2,560 字节。NPUCachingAllocator 小池采用**最小适配（best-fit）**策略，优先选取大小最接近需求的空闲块以减少碎片。段 1 空闲块（519,680 B）小于段 2 空闲块（1,572,352 B），因此分配 13 回到段 1：

```
地址 = 0x000012c041200000 + 1,577,472
     = 0x000012c041200000 + 0x181200
     = 0x000012c041381200  ✓
```

---

## 四、多出的 512 字节：round_size() 机制

### 4.1 v2.8.0 实现

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

等价公式：`round_size(n) = ⌈(n + 32) / 512⌉ × 512`

### 4.2 为什么恰好多出 512 字节

测试中所有请求大小均为 512 的整数倍（1 KB、2 KB、…、512 KB）。加 32 字节后变为非 512 整数倍，向上对齐到下一个 512 边界，固定额外增加 512 字节（= 1 × kMinBlockSize）：

```
例：请求 512 KB = 524,288 字节
  + 32  →  524,320 字节
  ÷ 512 →  1024.0625（非整数）
  向上取整 → 1025
  × 512 →  524,800 字节  （比请求多 512 字节）
```

### 4.3 各分配 round_size 计算

| # | 请求大小 | +32 后 | round_size 后 | 额外开销 |
|--:|--:|--:|--:|--:|
| 1 | 524,288 B (512 KB) | 524,320 B | 524,800 B | +512 B |
| 2 | 524,288 B (512 KB) | 524,320 B | 524,800 B | +512 B |
| 3 | 1,024 B (1 KB) | 1,056 B | 1,536 B | +512 B |
| 4 | 2,048 B (2 KB) | 2,080 B | 2,560 B | +512 B |
| 5 | 4,096 B (4 KB) | 4,128 B | 4,608 B | +512 B |
| 6 | 8,192 B (8 KB) | 8,224 B | 8,704 B | +512 B |
| 7 | 16,384 B (16 KB) | 16,416 B | 16,896 B | +512 B |
| 8 | 32,768 B (32 KB) | 32,800 B | 33,280 B | +512 B |
| 9 | 65,536 B (64 KB) | 65,568 B | 66,048 B | +512 B |
| 10 | 131,072 B (128 KB) | 131,104 B | 131,584 B | +512 B |
| 11 | 262,144 B (256 KB) | 262,176 B | 262,656 B | +512 B |
| 12 | 524,288 B (512 KB) | 524,320 B | 524,800 B | +512 B |
| 13 | 2,048 B (2 KB) | 2,080 B | 2,560 B | +512 B |

每个分配在段内额外占用 512 字节。

### 4.4 v2.11.0 实现（新老芯片分支）

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

| 芯片 | padding | 额外开销（512 B 对齐请求）|
|---|---|---|
| Ascend 910 B / 老系列 | +32 字节 | +512 字节 |
| Ascend 910_95 及以上 | +0 字节 | 0 字节 |

---

## 五、多出的 4 MB：首次分配触发 stream 初始化

### 5.1 触发路径

`NpuCachingAllocator::allocate()` 在每次分配时调用 `getCurrentNPUStreamNoWait()`（`NPUCachingAllocator.cpp:3447-3472`）：

```cpp
c10::DataPtr allocate(size_t size) override {
    int device = 0;
    NPU_CHECK_ERROR(c10_npu::GetDevice(&device));
    LazySetDevice(device);
    ...
    this->malloc(&devPtr, device, size,
        c10_npu::getCurrentNPUStreamNoWait(device));  // ← 触发 stream 初始化
}
```

`getCurrentNPUStreamNoWait()` → `initNPUStreamsOnce()` → `initGlobalStreamState()`（`NPUStream.cpp:203-240`）：

```cpp
static void initGlobalStreamState() {
    // 创建 default stream
    NPU_CHECK_ERROR(acl::AclrtCreateStreamWithConfig(
        &default_streams[device_id].stream,
        0,
        ACL_STREAM_FAST_LAUNCH | ACL_STREAM_FAST_SYNC));  // ← FAST_LAUNCH 触发预申请

    // 创建 secondary stream
    NPU_CHECK_ERROR(acl::AclrtCreateStreamWithConfig(
        &secondary_streams[device_id].stream,
        0,
        ACL_STREAM_FAST_LAUNCH | ACL_STREAM_FAST_SYNC));  // ← FAST_LAUNCH 触发预申请
}
```

`initNPUStreamsOnce()` 受 `initialize_flag[device_index]` 保护（`NPUStream.cpp:262-268`），对每个设备**只执行一次**。

### 5.2 CANN 内部 HBM 分配

两个 flag 的代价各不相同（引自 CANN 官方文档）：

| flag | 代价 |
|---|---|
| `ACL_STREAM_FAST_LAUNCH` | **增加内存消耗**：创建 stream 时预申请系统内部 HBM 资源，换取后续任务下发时延缩短 |
| `ACL_STREAM_FAST_SYNC` | **增加 CPU 消耗**：`aclrtSynchronizeStream` 时主动轮询而非被动等待，与内存无关 |

4 MB HBM 的来源是 **`ACL_STREAM_FAST_LAUNCH`**：CANN 在 `AclrtCreateStreamWithConfig` 内部、stream 创建时即完成 HBM 预申请，**完全不经过 torch-npu 的任何分配器**，因此：

- 不被 `NPUCachingAllocator` 的 `stats.reserved_bytes` 记录
- 对 `torch.npu.memory_reserved()` / `torch.npu.memory_allocated()` 不可见
- 对 `aclrtGetMemInfo(ACL_HBM_MEM)` **可见**

由实验数据反推：`initGlobalStreamState()` 创建 2 条带 `ACL_STREAM_FAST_LAUNCH` 的 stream，合计消耗约 **4 MB HBM**（每条约 2 MB），进程内一次性开销。

---

## 六、完整内存汇总

### 6.1 两种内存统计体系

**`torch.npu.memory_reserved()`** 只反映 `NPUCachingAllocator` 自身的统计，在 `alloc_block()` 成功后写入：

```cpp
// NPUCachingAllocator.cpp:2627-2630
for_each_selected_stat_type(p.stat_types, [&](size_t stat_type) {
    update_stat(stats.segment[stat_type], 1);
    update_stat(stats.reserved_bytes[stat_type], size);  // 仅此处更新
});
```

范围：仅限 `NPUCachingAllocator` 通过 `AclrtMallocAlign32` 申请的 segment。

**`acl.rt.get_mem_info(ACL_MEM_TYPE_HBM)`** 调用底层 `aclrtGetMemInfo(ACL_HBM_MEM)`，返回设备 HBM **全局物理占用量**，包括：
- `NPUCachingAllocator` 分配的 segment
- `NPUWorkspaceAllocator` 分配的算子 workspace
- CANN runtime 内部分配的任何缓冲区（对 PyTorch 完全不透明）

两者之差即为"PyTorch 不知道的 HBM 占用"。

### 6.2 逐步演进

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

**分配 1 触发 +6 MB** 的分解：

```
+4 MB  ← initGlobalStreamState() 创建 2 条 ACL_STREAM_FAST_LAUNCH stream
          CANN 预申请 HBM 内部资源（≈2 MB × 2），对 PyTorch 不可见
+2 MB  ← alloc_block() → AclrtMallocAlign32(kSmallBuffer = 2MB)
          段 1 创建，PT reserved +2 MB
```

### 6.3 PT allocated 统计口径

`torch.npu.memory_allocated()` 记录的是 `round_size()` 之后的块大小之和，**不是**原始张量字节数。

验证（所有 13 次分配的 round_size 累计）：

```
524800×2 + 1536 + 2560 + 4608 + 8704 + 16896 + 33280 + 66048 + 131584 + 262656 + 524800 + 2560
= 2,104,832 字节 = 2.0078 MB → 显示为 2.01 MB  ✓
```

### 6.4 最终汇总

| 来源 | 大小 | PT reserved 可见 | HBM get_mem_info 可见 |
|---|---|---|---|
| CANN stream 内部资源（ACL_STREAM_FAST_LAUNCH × 2 条） | ≈4 MB | ✗ | ✓ |
| 段 1（NPUCachingAllocator，kSmallBuffer） | 2 MB | ✓ | ✓ |
| 段 2（NPUCachingAllocator，kSmallBuffer） | 2 MB | ✓ | ✓ |
| **HBM 合计** | **≈8 MB** | — | — |
| PT reserved 合计 | 4 MB | ✓ | ✓ |
| **持久差值（HBM − PT reserved）** | **4 MB** | ✗ | ✓ |

> **测量注意**：`empty_cache()` 调用 `aclrtFree()` 将段归还驱动的 free pool，而非立即释放物理 HBM；下次 `AclrtMallocAlign32` 从 free pool 取内存时 `aclrtGetMemInfo()` 数值不变。因此只有在进程冷启动（无 warmup、不调用 `empty_cache()`、保活所有张量）的条件下，`get_mem_info()` 的差值才准确反映累计 HBM 分配量。

---

## 附：完整测试代码

```python
import torch
import os
import sys
import acl

os.environ['ASCEND_GLOBAL_LOG_LEVEL'] = '3'  # ERROR 级别，避免调试日志干扰输出

# 初始化 ACL
ret = acl.init()
if ret != 0:
    print(f"Failed to initialize ACL: {ret}")
    sys.exit(1)

device_id = 2
ret = acl.rt.set_device(device_id)
if ret != 0:
    print(f"Failed to set device: {ret}")
    acl.finalize()
    sys.exit(1)

print(f"Using device: npu:{device_id}")

def get_npu_mem_info(attr):
    free, total, ret = acl.rt.get_mem_info(attr)
    if ret != 0:
        print(f"Failed to get mem info: {ret}")
        return 0, 0
    return free, total

ACL_MEM_TYPE_HBM = 0
device = torch.device(f'npu:{device_id}')

# 获取初始内存信息
print("\n=== Initial Memory Status ===")
free_init, total_init = get_npu_mem_info(ACL_MEM_TYPE_HBM)
print(f"HBM Memory:")
print(f"  Total: {total_init / (1024*1024):.2f} MB")
print(f"  Free:  {free_init / (1024*1024):.2f} MB")
print(f"  Used:  {(total_init - free_init) / (1024*1024):.2f} MB")
print(f"  Allocated: {torch.npu.memory_allocated() / (1024*1024):.2f} MB")
print(f"  Reserved:  {torch.npu.memory_reserved() / (1024*1024):.2f} MB")

sizes = [512*1024, 512*1024, 1*1024, 2*1024, 4*1024, 8*1024,
         16*1024, 32*1024, 64*1024, 128*1024, 256*1024, 512*1024, 2*1024]
tensors = []

print("\n=== Allocating Memory ===")
for i, size in enumerate(sizes):
    num_elements = size // 4
    tensor = torch.randn(num_elements, device=device)
    tensors.append(tensor)
    addr = tensor.data_ptr()
    print(f"\n  Allocation {i+1}: {size/1024} KB, address: {addr:016x}")

    if i > 0:
        base_address = tensors[0].data_ptr()
        offset = addr - base_address
        if offset < 2 * 1024 * 1024:
            print(f"    In the same 2MB block as allocation 1 (offset: {offset} bytes)")
        else:
            print(f"    In a new block (offset: {offset} bytes >= 2MB)")

    free_hbm, total_hbm = get_npu_mem_info(ACL_MEM_TYPE_HBM)
    print(f"    HBM used:           {(total_hbm - free_hbm) / (1024*1024):.2f} MB")
    print(f"    PyTorch allocated:  {torch.npu.memory_allocated() / (1024*1024):.2f} MB")
    print(f"    PyTorch reserved:   {torch.npu.memory_reserved() / (1024*1024):.2f} MB")

print("\n=== Final Memory Status ===")
free_final, total_final = get_npu_mem_info(ACL_MEM_TYPE_HBM)
print(f"HBM Memory:")
print(f"  Total: {total_final / (1024*1024):.2f} MB")
print(f"  Free:  {free_final / (1024*1024):.2f} MB")
print(f"  Used:  {(total_final - free_final) / (1024*1024):.2f} MB")
print(f"  Allocated: {torch.npu.memory_allocated() / (1024*1024):.2f} MB")
print(f"  Reserved:  {torch.npu.memory_reserved() / (1024*1024):.2f} MB")

print("\nMemory allocation completed.")
input("Press Enter to exit...")

# 清理资源（顺序：先释放张量，再重置设备）
del tensors
import gc
gc.collect()
torch.npu.empty_cache()
print(f"After empty_cache reserved: {torch.npu.memory_reserved()/(1024*1024):.2f} MB")

ret = acl.rt.reset_device(device_id)
if ret != 0:
    print(f"Failed to reset device: {ret}")

ret = acl.finalize()
if ret != 0:
    print(f"Failed to finalize ACL: {ret}")

print("Resources released successfully.")
```
