import os
import gc
import unittest
import numpy as np

import torch
import torch_npu
from torch_npu.testing.common_utils import SupportedDevices
from torch_npu.testing.testcase import TestCase, run_tests

os.environ["PYTORCH_NPU_ALLOC_CONF"] = "roundup_power2_divisions:[8:1,64:4,128:8,256:2,>:16]"
# roundup_power2_divisions: 0:1, 1:1, 2:1, 3:1, 4:0, 5:0, 6:4, 7:8, 8:2, 9:16, 10:16, 11:16, 12:16, 13:16, 14:16, 15:16


class Test_roundup_power2_divisions(TestCase):
    def test_divisions(self):
        def power2_div(size, div_factor):
            pow2 = 1
            while pow2 < size:
                pow2 = pow2 * 2
            if pow2 == size:
                return pow2
            step = pow2 / 2 / div_factor
            ret = pow2 / 2
            while ret < size:
                ret = ret + step
            return ret

        def align_to_512_or_2m(size):
            new_size = ((size + 511) // 512) * 512
            if new_size < 10485760:
                return new_size
            else:
                return ((size + 2097152 - 1) // 2097152) * 2097152

        def do_test(nelems, divisions=0):
            nbytes = nelems
            # do not use torch.rand
            x = torch.from_numpy(np.random.randn(nelems).astype(np.uint8)).to("npu")
            requested_bytes = torch.npu.memory_stats()["requested_bytes.all.current"]
            do_assert(requested_bytes, nbytes)

            max_memory_allocated = torch.npu.max_memory_allocated()
            allocated_bytes = torch.npu.memory_stats()["allocated_bytes.all.current"]
            active_bytes = torch.npu.memory_stats()["active_bytes.all.current"]
            do_assert(max_memory_allocated, allocated_bytes)
            do_assert(max_memory_allocated, active_bytes)

            if divisions > 1:
                do_assert(max_memory_allocated, power2_div(nbytes, divisions))
            else:
                do_assert(max_memory_allocated, align_to_512_or_2m(nbytes))

        def do_assert(x, y):
            self.assertEqual(x, y)

        torch.npu.memory.empty_cache()
        do_test(513)
        do_test(1025)
        do_test(33 * 1024 * 1024) # index:5, division:0
        do_test(65 * 1024 * 1024, 4) # index:6, division:4
        do_test(200 * 1024 * 1024, 8) # index:7, division:8
        do_test(510 * 1024 * 1024, 2) # index:8, division:2
        do_test(513 * 1024 * 1024, 16) # index:9, division:16

    @SupportedDevices(['Ascend910A'])
    def test_allocator_rounding_with_pad32(self):
        torch.npu.memory.empty_cache()

        x = torch.from_numpy(np.random.randn(512).astype(np.uint8)).to("npu")
        allocated_bytes = torch.npu.memory_stats()["allocated_bytes.all.current"]
        active_bytes = torch.npu.memory_stats()["active_bytes.all.current"]
        self.assertEqual(allocated_bytes, active_bytes)
        self.assertEqual(allocated_bytes, 1024)

    @SupportedDevices(['Ascend910A'])
    def test_rounding_bytes_stats(self):
        torch.npu.memory.empty_cache()
        torch.npu.reset_peak_memory_stats()
        torch.npu.reset_accumulated_memory_stats()

        x = torch.from_numpy(np.zeros(513, dtype=np.uint8)).to("npu")
        y = torch.from_numpy(np.zeros(65 * 1024 * 1024, dtype=np.uint8)).to("npu")

        stats = torch.npu.memory_stats()
        self.assertEqual(stats["requested_bytes.all.current"], 65 * 1024 * 1024 + 513)
        self.assertEqual(stats["rounding_bytes.all.current"], 15 * 1024 * 1024 + 511)
        self.assertEqual(stats["rounding_bytes.all.peak"], 15 * 1024 * 1024 + 511)
        self.assertEqual(stats["rounding_bytes_by_granularity.512.all.current"], 511)
        self.assertEqual(stats["rounding_bytes_by_granularity.16777216.all.current"], 15 * 1024 * 1024)

        del x
        del y
        gc.collect()

        stats = torch.npu.memory_stats()
        self.assertEqual(stats["rounding_bytes.all.current"], 0)
        self.assertEqual(stats["rounding_bytes.all.allocated"], 15 * 1024 * 1024 + 511)
        self.assertEqual(stats["rounding_bytes.all.freed"], 15 * 1024 * 1024 + 511)

    @SupportedDevices(['Ascend910A'])
    def test_segment_free_bytes_stats(self):
        torch.npu.memory.empty_cache()
        torch.npu.reset_peak_memory_stats()
        torch.npu.reset_accumulated_memory_stats()

        x = torch.from_numpy(np.zeros(513, dtype=np.uint8)).to("npu")

        stats = torch.npu.memory_stats()
        self.assertEqual(stats["allocated_bytes.all.current"], 1024)
        self.assertEqual(stats["segment_free_bytes.all.current"], 2 * 1024 * 1024 - 1024)
        self.assertEqual(
            stats["segment_free_bytes_by_granularity.2097152.all.current"],
            2 * 1024 * 1024 - 1024,
        )

        del x
        gc.collect()
        torch.npu.memory.empty_cache()

        stats = torch.npu.memory_stats()
        self.assertEqual(stats["segment_free_bytes.all.current"], 0)
        self.assertEqual(stats["segment_free_bytes.all.peak"], 2 * 1024 * 1024)


if __name__ == '__main__':
    run_tests()
