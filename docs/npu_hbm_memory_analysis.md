# NPU HBM 统计差值分析：消失的 4 MB

## 一、现象

在以下分配序列中：

```python
sizes = [512KB, 512KB, 1KB, 2KB, 4KB, 8KB, 16KB, 32KB, 64KB, 128KB, 256KB, 512KB, 2KB]
tensors = [torch.randn(size // 4, device=device) for size in sizes]
```

两种内存查询接口的结果出现持续差值：

| 指标 | 值 |
|---|---|
| HBM 实际消耗（`acl.rt.get_mem_info` 差值） | **8 MB** |
| PyTorch 报告的 reserved（`torch.npu.memory_reserved()`） | **4 MB** |
| **无法解释的差值** | **4 MB** |

逐步观察揭示差值来源：

```
分配 1（512KB randn）：  HBM Δ = +6 MB，PT reserved Δ = +2 MB  →  一次性差值 +4 MB
分配 2–11（同段内）：    HBM Δ =  0 MB，PT reserved Δ =  0 MB
分配 12（512KB randn）： HBM Δ = +2 MB，PT reserved Δ = +2 MB  →  差值为 0
分配 13（2KB randn）：   HBM Δ =  0 MB，PT reserved Δ =  0 MB
```

4 MB 差值在第一次分配时**一次性**出现，此后完全不再增长。

---

## 二、两种内存统计体系

### 2.1 `torch.npu.memory_reserved()`

只反映 `NPUCachingAllocator` 自身的统计，在 `alloc_block()` 成功后写入：

```cpp
// NPUCachingAllocator.cpp:2627-2630
for_each_selected_stat_type(p.stat_types, [&](size_t stat_type) {
    update_stat(stats.segment[stat_type], 1);
    update_stat(stats.reserved_bytes[stat_type], size);  // 仅此处更新
});
```

**范围**：仅限 `NPUCachingAllocator` 通过 `AclrtMallocAlign32` 申请的 segment。

### 2.2 `acl.rt.get_mem_info(ACL_MEM_TYPE_HBM)`

调用底层 `aclrtGetMemInfo(ACL_HBM_MEM)`，返回**设备 HBM 全局物理占用量**，包括：
- `NPUCachingAllocator` 分配的 segment
- `NPUWorkspaceAllocator` 分配的算子 workspace
- CANN runtime 内部分配的任何缓冲区（对 PyTorch 完全不透明）

两者之差即为"PyTorch 不知道的 HBM 占用"。

---

## 三、可见的 4 MB：NPUCachingAllocator 的分配行为

### 3.1 分配大小策略

`NPUCachingAllocator.cpp:2426-2435`（`get_allocation_size` 函数）：

```cpp
static size_t get_allocation_size(size_t size) {
    if (size <= kSmallSize)          // ≤ 1 MB
        return kSmallBuffer;         // 向驱动申请 2 MB
    else if (size < kMinLargeAlloc)  // 1–10 MB
        return kLargeBuffer;         // 向驱动申请 20 MB
    else
        return kRoundLarge * ((size + kRoundLarge - 1) / kRoundLarge);
}
```

相关常量（`NPUAllocatorConfig.h:44-46`）：

```cpp
constexpr size_t kSmallSize   = 1048576;   // 1 MB：小/大分配分界线
constexpr size_t kSmallBuffer = 2097152;   // 2 MB：小分配段大小
constexpr size_t kLargeBuffer = 20971520;  // 20 MB：中等分配段大小
```

### 3.2 本例分配行为

所有 13 次分配均 ≤ 1 MB（小分配），均使用 `kSmallBuffer = 2 MB` 段：

- **分配 1**（512 KB）：池内无空闲段 → `AclrtMallocAlign32(2 MB)` → 创建第一个 2 MB 段
- **分配 2–11、13**：大小之和 ≤ 2 MB，全部填入第一个段，不触发新申请
- **分配 12**（512 KB）：第一个段剩余空间不足 → `AclrtMallocAlign32(2 MB)` → 创建第二个 2 MB 段

结果：`memory_reserved() = 2 MB × 2 = 4 MB` ✓

---

## 四、隐藏的 4 MB：首次 `allocate()` 的 stream 初始化

### 4.1 实验定位

在保证"无 warmup、不调用 `empty_cache()`、所有张量保活"的条件下，逐步叠加测量：

```
torch.empty(200KB)  → HBM Δ = +6.00 MB，PT_reserved Δ = +2.00 MB，hidden = +4.00 MB
torch.randn(200KB)  → HBM Δ =  0.00 MB，PT_reserved Δ =  0.00 MB，hidden =  0.00 MB
torch.randn(200KB)  → HBM Δ =  0.00 MB，PT_reserved Δ =  0.00 MB，hidden =  0.00 MB
torch.empty(200KB)  → HBM Δ =  0.00 MB，PT_reserved Δ =  0.00 MB，hidden =  0.00 MB
```

**关键结论**：
- `torch.empty()`（不执行任何 CANN 算子）产生 hidden = +4 MB
- `torch.randn()`（执行 CANN 算子）hidden = 0
- 第二次 `torch.empty()` hidden = 0

差值与 CANN 算子执行无关，是纯内存分配路径的一次性副作用。

### 4.2 触发路径

`NpuCachingAllocator::allocate()` 的完整调用链（`NPUCachingAllocator.cpp:3447-3472`）：

```cpp
c10::DataPtr allocate(size_t size) override {
    int device = 0;
    NPU_CHECK_ERROR(c10_npu::GetDevice(&device));
    LazySetDevice(device);                                   // (1) 设备初始化
    ...
    this->malloc(&devPtr, device, size,
        c10_npu::getCurrentNPUStreamNoWait(device));         // (2) 触发 stream 初始化
}
```

`getCurrentNPUStreamNoWait()` → `initNPUStreamsOnce()` → `initGlobalStreamState()`（`NPUStream.cpp:203-240`）：

```cpp
static void initGlobalStreamState() {
    // 创建 default stream
    NPU_CHECK_ERROR(acl::AclrtCreateStreamWithConfig(
        &default_streams[device_id].stream,
        0,
        ACL_STREAM_FAST_LAUNCH | ACL_STREAM_FAST_SYNC));   // ← FAST_LAUNCH 触发预申请

    // 创建 secondary stream
    NPU_CHECK_ERROR(acl::AclrtCreateStreamWithConfig(
        &secondary_streams[device_id].stream,
        0,
        ACL_STREAM_FAST_LAUNCH | ACL_STREAM_FAST_SYNC));   // ← FAST_LAUNCH 触发预申请
}
```

`initNPUStreamsOnce()` 受 `initialize_flag[device_index]` 保护（`NPUStream.cpp:262-268`），对每个设备只执行一次。

### 4.3 CANN 内部 HBM 分配

两个 flag 的代价各不相同（引自 CANN 官方文档）：

| flag | 代价 |
|---|---|
| `ACL_STREAM_FAST_LAUNCH` | **增加内存消耗**：创建 stream 时预申请系统内部资源，换取后续下发任务时延缩短 |
| `ACL_STREAM_FAST_SYNC` | **增加 CPU 消耗**：`aclrtSynchronizeStream` 时主动轮询而非被动等待，与内存无关 |

4 MB HBM 差值的真正来源是 **`ACL_STREAM_FAST_LAUNCH`**：CANN 在 `AclrtCreateStreamWithConfig` 内部、stream 创建时即完成 HBM 的预申请，**完全不经过 torch-npu 的任何分配器**，因此：
- 不被 `NPUCachingAllocator` 的 `stats.reserved_bytes` 记录
- 不被 `NPUWorkspaceAllocator` 的统计记录
- 对 `torch.npu.memory_reserved()` / `torch.npu.memory_allocated()` 不可见
- 对 `aclrtGetMemInfo(ACL_HBM_MEM)` **可见**

由实验数据反推：`initGlobalStreamState()` 创建 2 条带 `ACL_STREAM_FAST_LAUNCH` 的 stream，合计消耗 **4 MB HBM**（每条约 2 MB）。

---

## 五、完整因果链

```
原始脚本：首次 torch.randn(512KB) 对应 allocate(512KB)

allocate(512KB)
  ├─ [首次执行，之后不再] getCurrentNPUStreamNoWait()
  │    └─ initNPUStreamsOnce()
  │         └─ initGlobalStreamState()
  │              ├─ AclrtCreateStreamWithConfig(default_stream,
  │              │       ACL_STREAM_FAST_LAUNCH | ACL_STREAM_FAST_SYNC)
  │              │    └─ CANN 内部：ACL_STREAM_FAST_LAUNCH 预申请 HBM 内部资源 ≈ +2 MB
  │              │       （torch-npu 不可见）
  │              └─ AclrtCreateStreamWithConfig(secondary_stream,
  │                      ACL_STREAM_FAST_LAUNCH | ACL_STREAM_FAST_SYNC)
  │                   └─ CANN 内部：ACL_STREAM_FAST_LAUNCH 预申请 HBM 内部资源 ≈ +2 MB
  │                      （torch-npu 不可见）
  │
  │       stream 初始化 HBM 小计：+4 MB（不计入 PyTorch 任何统计）
  │
  └─ alloc_block() → AclrtMallocAlign32(kSmallBuffer = 2 MB)
                      └─ HBM +2 MB，memory_reserved() +2 MB ✓

首次分配：HBM Δ = 4(stream) + 2(segment) = +6 MB，PyTorch visible = +2 MB

第 12 次分配（512KB randn，需要第二个段）：
  └─ alloc_block() → AclrtMallocAlign32(kSmallBuffer = 2 MB)
                      └─ HBM +2 MB，memory_reserved() +2 MB ✓
     （stream 已初始化，hidden = 0）

全程汇总：
  HBM 消耗 = 4 MB(stream init) + 2 MB(seg-1) + 2 MB(seg-2) = 8 MB
  PyTorch 可见 = 2 MB(seg-1) + 2 MB(seg-2) = 4 MB
  持久差值 = 4 MB  ←── 全部来自 CANN stream 初始化，进程内一次性开销
```

---

## 六、附：`get_mem_info()` 的驱动层缓存效应

直接测量时需注意：`empty_cache()` 调用 `aclrtFree()` 将内存归还驱动，但驱动将其放入 **free pool** 而非立即释放物理 HBM。下次 `AclrtMallocAlign32` 从 free pool 取内存时，`aclrtGetMemInfo()` 的数值不变。

```
empty_cache()          → aclrtFree(2MB) → 进入驱动 free pool（HBM used 不变）
下次 allocate()         → AclrtMallocAlign32 从 free pool 取   （HBM used 仍不变）
```

因此，**只有在进程冷启动（无 warmup、不调用 `empty_cache()`、保活所有张量）的条件下**，`get_mem_info()` 的差值才准确反映真实的累计 HBM 分配量。
