"""Scatter/gather chain processing: happy path, errors, halt semantics."""

import unittest

from dmaring import (
    CoherentMemory,
    DeviceHaltedError,
    DmaRing,
    SyntheticDevice,
    reference_gather_copy,
)


def make_source(size=256):
    """Deterministic byte source generated in-repo (no external data)."""
    return bytes(((i * 31 + 7) ^ (i >> 1)) & 0xFF for i in range(size))


def build(mem_size=256):
    mem = CoherentMemory(mem_size)
    src = make_source(mem_size)
    mem.cpu_write(0, src)
    mem.flush()
    rings = {0: DmaRing(4, ring_id=0), 1: DmaRing(2, ring_id=1)}
    dev = SyntheticDevice(mem)
    return mem, src, rings, dev


class TestHappyPathCrossRing(unittest.TestCase):
    def test_cross_ring_gather_copy(self):
        mem, src, rings, dev = build()
        # Chain spans both rings: (0,0) -> (1,1) -> (0,2) -> END.
        rings[0].cpu_write_many([
            {"slot": 0, "offset": 0, "length": 16, "next": (1, 1)},
            {"slot": 2, "offset": 100, "length": 12, "next": None},
        ])
        rings[1].cpu_write(1, offset=32, length=8, next=(0, 2))
        for slot in (0, 2):
            rings[0].cpu_release(slot)
        rings[1].cpu_release(1)

        result = dev.process_chain(rings, (0, 0))
        self.assertEqual(result.status, "ok")
        self.assertIsNone(result.error)
        self.assertEqual(result.bytes_copied, 36)
        self.assertEqual(len(result.items), 3)
        self.assertEqual([ (c.ring_id, c.slot) for c in result.items ],
                         [(0, 0), (1, 1), (0, 2)])

        expected = reference_gather_copy(
            src, [[(0, 16), (32, 8), (100, 12)]])
        self.assertEqual(bytes(dev.output), expected)

        # Per-item completion counts.
        self.assertEqual(dev.completion_count(0, 0), 1)
        self.assertEqual(dev.completion_count(1, 1), 1)
        self.assertEqual(dev.completion_count(0, 2), 1)
        self.assertEqual(dev.completion_count(0, 1), 0)

        # CPU can acquire every completion after the legal handoff.
        self.assertEqual(rings[0].cpu_acquire(0).status, "ok")
        self.assertEqual(rings[1].cpu_acquire(1).bytes_copied, 8)
        self.assertEqual(rings[0].cpu_acquire(2).bytes_copied, 12)

    def test_empty_chain_completes_ok(self):
        _, _, rings, dev = build()
        result = dev.process_chain(rings, None)
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.items, ())
        self.assertEqual(result.bytes_copied, 0)

    def test_reused_slots_accumulate_completion_counts(self):
        mem, src, rings, dev = build()
        for _ in range(3):
            rings[0].cpu_write(1, offset=0, length=4, next=None)
            rings[0].cpu_release(1)
            result = dev.process_chain(rings, (0, 1))
            self.assertEqual(result.status, "ok")
            rings[0].cpu_acquire(1)
        self.assertEqual(dev.completion_count(0, 1), 3)
        expected = reference_gather_copy(src, [[(0, 4)]] * 3)
        self.assertEqual(bytes(dev.output), expected)


class TestChainErrors(unittest.TestCase):
    def test_out_of_bounds_buffer_error_completes_chain(self):
        mem, src, rings, dev = build()
        rings[0].cpu_write(0, offset=250, length=16, next=None)  # 266 > 256
        rings[0].cpu_release(0)
        result = dev.process_chain(rings, (0, 0))
        self.assertEqual(result.status, "error")
        self.assertIn("exceeds memory size", result.error)
        # The bad descriptor itself got an error completion...
        self.assertEqual(len(result.items), 1)
        self.assertEqual(result.items[0].status, "error")
        self.assertEqual(result.items[0].bytes_copied, 0)
        self.assertEqual(dev.completion_count(0, 0), 1)
        # ...and the CPU can acquire it after the handback.
        self.assertEqual(rings[0].cpu_acquire(0).status, "error")

    def test_half_chain_error_skips_rest_and_halts(self):
        mem, src, rings, dev = build()
        # Chain of three; the middle descriptor is out of bounds.
        rings[0].cpu_write_many([
            {"slot": 0, "offset": 0, "length": 8, "next": (1, 0)},
            {"slot": 3, "offset": 64, "length": 4, "next": None},
        ])
        rings[1].cpu_write(0, offset=300, length=4, next=(0, 3))  # bad
        rings[0].cpu_release(0)
        rings[0].cpu_release(3)
        rings[1].cpu_release(0)

        result = dev.process_chain(rings, (0, 0))
        self.assertEqual(result.status, "error")
        # First item ok, middle item error, tail not consumed.
        self.assertEqual([c.status for c in result.items], ["ok", "error"])
        self.assertEqual(result.bytes_copied, 8)
        self.assertEqual(bytes(dev.output), src[0:8])
        self.assertEqual(dev.completion_count(0, 0), 1)
        self.assertEqual(dev.completion_count(1, 0), 1)
        self.assertEqual(dev.completion_count(0, 3), 0)  # never consumed
        # Tail descriptor still belongs to the device, unreleased.
        self.assertEqual(rings[0].owner_of(3), "device")

        # Subsequent chains are not consumed while halted.
        with self.assertRaises(DeviceHaltedError):
            dev.process_chain(rings, (0, 3))
        self.assertEqual(dev.completion_count(0, 3), 0)

        # After reset the remaining chain can be consumed.
        dev.reset()
        result2 = dev.process_chain(rings, (0, 3))
        self.assertEqual(result2.status, "ok")
        self.assertEqual(dev.completion_count(0, 3), 1)

    def test_chain_loop_detected(self):
        mem, src, rings, dev = build()
        # (0,0) -> (1,1) -> (0,0): a cycle across rings.
        rings[0].cpu_write(0, offset=0, length=4, next=(1, 1))
        rings[1].cpu_write(1, offset=8, length=4, next=(0, 0))
        rings[0].cpu_release(0)
        rings[1].cpu_release(1)
        result = dev.process_chain(rings, (0, 0))
        self.assertEqual(result.status, "error")
        self.assertIn("loop", result.error)
        # Each descriptor on the cycle was consumed exactly once.
        self.assertEqual(dev.completion_count(0, 0), 1)
        self.assertEqual(dev.completion_count(1, 1), 1)
        self.assertEqual(bytes(dev.output), src[0:4] + src[8:12])
        with self.assertRaises(DeviceHaltedError):
            dev.process_chain(rings, None)

    def test_self_loop_detected(self):
        mem, src, rings, dev = build()
        rings[0].cpu_write(0, offset=0, length=4, next=(0, 0))
        rings[0].cpu_release(0)
        result = dev.process_chain(rings, (0, 0))
        self.assertEqual(result.status, "error")
        self.assertIn("loop", result.error)
        self.assertEqual(dev.completion_count(0, 0), 1)

    def test_unknown_ring_pointer(self):
        mem, src, rings, dev = build()
        rings[0].cpu_write(0, offset=0, length=4, next=(9, 0))
        rings[0].cpu_release(0)
        result = dev.process_chain(rings, (0, 0))
        self.assertEqual(result.status, "error")
        self.assertIn("unknown ring_id 9", result.error)

    def test_missing_release_surfaces_as_chain_error(self):
        mem, src, rings, dev = build()
        rings[0].cpu_write(0, offset=0, length=4, next=None)
        # No cpu_release: device cannot acquire the head descriptor.
        result = dev.process_chain(rings, (0, 0))
        self.assertEqual(result.status, "error")
        self.assertIn("acquire failed", result.error)
        self.assertEqual(dev.completion_count(0, 0), 0)
        self.assertEqual(bytes(dev.output), b"")

    def test_slot_out_of_range_in_chain(self):
        mem, src, rings, dev = build()
        rings[0].cpu_write(0, offset=0, length=4, next=(1, 7))  # ring 1 cap 2
        rings[0].cpu_release(0)
        result = dev.process_chain(rings, (0, 0))
        self.assertEqual(result.status, "error")
        self.assertIn("acquire failed", result.error)


if __name__ == "__main__":
    unittest.main()
