import argparse
from typing import Iterable

import torch
import torch_npu


def _format_bytes(num_bytes: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(num_bytes)
    unit = units[0]
    for unit in units:
        if abs(value) < 1024.0 or unit == units[-1]:
            break
        value /= 1024.0
    return f"{num_bytes} ({value:.2f} {unit})"


def _print_stat(stats: dict, key: str) -> None:
    print(f"{key:<30}: {_format_bytes(int(stats.get(key, 0)))}")


def _print_granularity(stats: dict, prefix: str) -> None:
    keys = sorted(k for k in stats if k.startswith(prefix))
    print(f"{prefix}:")
    if not keys:
        print("  <all zero or key not present>")
        return
    for key in keys:
        value = int(stats[key])
        if value == 0:
            continue
        print(f"  {key}: {_format_bytes(value)}")


def dump_allocator_stats(tag: str, extra_keys: Iterable[str] = ()) -> None:
    stats = torch.npu.memory_stats()
    free_bytes, total_bytes = torch.npu.mem_get_info()
    print(f"\n========== {tag} ==========")
    print(f"{'mem_get_info.free':<30}: {_format_bytes(free_bytes)}")
    print(f"{'mem_get_info.total':<30}: {_format_bytes(total_bytes)}")

    keys = [
        "requested_bytes.all.current",
        "rounding_bytes.all.current",
        "allocated_bytes.all.current",
        "active_bytes.all.current",
        "reserved_bytes.all.current",
        "segment_free_bytes.all.current",
        "inactive_split_bytes.all.current",
    ]
    keys.extend(extra_keys)

    for key in keys:
        _print_stat(stats, key)

    reserved = int(stats.get("reserved_bytes.all.current", 0))
    active = int(stats.get("active_bytes.all.current", 0))
    print(f"{'reserved_minus_active':<30}: {_format_bytes(reserved - active)}")

    _print_granularity(stats, "rounding_bytes_by_granularity")
    _print_granularity(stats, "segment_free_bytes_by_granularity")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print allocator-focused NPU memory stats for LLM profiling."
    )
    parser.add_argument("--tag", default="allocator_stats")
    args = parser.parse_args()

    if not torch.npu.is_available():
        raise RuntimeError("NPU is not available.")

    dump_allocator_stats(args.tag)


if __name__ == "__main__":
    main()
