"""Tests for dma_ring.  Expected values are derived independently of the
core (hand-computed slices of the deterministic source data), never by
calling the implementation under test."""

import math
import unittest

from dma_ring import (
    ChainError,
    CopyEngine,
    DmaRing,
    GenerationError,
    OwnershipError,
    VisibilityError,
)

# Deterministic source data generated in-repo (no external inputs).
SOURCE = bytes((i * 7 + 3) % 256 for i in range(256))


def make_ring(capacity=8, pool_size=256):
    ring = DmaRing(capacity, pool_size)
    ring.pool[:] = SOURCE[:pool_size]
    return ring


def write_and_release(ring, index, *, offset, length, next=None):
    ring.cpu_write(index, offset=offset, length=length, next=next)
    ring.cpu_release(index)


class TestInputValidation(unittest.TestCase):
    def test_capacity_must_be_power_of_two_within_limit(self):
        for good in (1, 2, 4, 8, 16, 32, 64):
            DmaRing(good, 0)
        for bad in (0, 3, 5, 6, 7, 63, 65, 128, -2):
            with self.assertRaises(ValueError, msg=f"capacity={bad}"):
                DmaRing(bad, 0)

    def test_capacity_rejects_bool_and_non_finite(self):
        with self.assertRaises(TypeError):
            DmaRing(True, 0)
        with self.assertRaises(ValueError):
            DmaRing(float("nan"), 0)
        with self.assertRaises(ValueError):
            DmaRing(float("inf"), 0)
        with self.assertRaises(TypeError):
            DmaRing(2.5, 0)
        DmaRing(4.0, 0)  # integral float accepted

    def test_integer_fields_reject_bool_nan_inf(self):
        ring = make_ring()
        with self.assertRaises(TypeError):
            ring.cpu_write(0, offset=True, length=1)
        with self.assertRaises(ValueError):
            ring.cpu_write(0, offset=float("nan"), length=1)
        with self.assertRaises(ValueError):
            ring.cpu_write(0, offset=0, length=math.inf)
        with self.assertRaises(TypeError):
            ring.cpu_write(0, offset=0, length=1, next=False)
        with self.assertRaises(TypeError):
            ring.cpu_write(0, offset=0, length=1, next=1.5)

    def test_negative_offset_length_rejected(self):
        ring = make_ring()
        with self.assertRaises(ValueError):
            ring.cpu_write(0, offset=-1, length=1)
        with self.assertRaises(ValueError):
            ring.cpu_write(0, offset=0, length=-4)

    def test_index_out_of_range_located(self):
        ring = make_ring()
        with self.assertRaises(IndexError):
            ring.cpu_write(8, offset=0, length=1)
        with self.assertRaises(IndexError):
            ring.device_acquire(-1)

    def test_failed_write_leaves_slot_unchanged(self):
        ring = make_ring()
        ring.cpu_write(2, offset=10, length=4, next=None)
        before = ring.inspect(2)
        with self.assertRaises(ValueError):
            ring.cpu_write(2, offset=99, length=-1, next=3)
        self.assertEqual(ring.inspect(2), before)


class TestOwnershipAndVisibility(unittest.TestCase):
    def test_device_acquire_without_release_raises(self):
        ring = make_ring()
        ring.cpu_write(0, offset=0, length=4)
        with self.assertRaises(OwnershipError):
            ring.device_acquire(0)

    def test_device_read_without_acquire_raises(self):
        ring = make_ring()
        write_and_release(ring, 0, offset=0, length=4)
        with self.assertRaises(VisibilityError):
            ring.device_read(0)

    def test_cpu_acquire_before_device_release_raises(self):
        ring = make_ring()
        write_and_release(ring, 0, offset=0, length=4)
        with self.assertRaises(OwnershipError):
            ring.cpu_acquire(0)

    def test_cpu_read_completion_without_acquire_raises(self):
        ring = make_ring()
        write_and_release(ring, 0, offset=0, length=4)
        ring.device_acquire(0)
        ring.device_release(0, status="ok", bytes_done=4)
        with self.assertRaises(VisibilityError):
            ring.cpu_read_completion(0)

    def test_cpu_write_after_release_raises(self):
        ring = make_ring()
        write_and_release(ring, 0, offset=0, length=4)
        with self.assertRaises(OwnershipError):
            ring.cpu_write(0, offset=0, length=4)

    def test_double_acquire_raises(self):
        ring = make_ring()
        write_and_release(ring, 0, offset=0, length=4)
        ring.device_acquire(0)
        with self.assertRaises(OwnershipError):
            ring.device_acquire(0)

    def test_full_handoff_sequence(self):
        ring = make_ring()
        write_and_release(ring, 0, offset=8, length=4)
        snap = ring.device_acquire(0)
        self.assertEqual((snap.offset, snap.length, snap.next), (8, 4, None))
        self.assertEqual(snap.generation, 1)
        ring.device_release(0, status="ok", bytes_done=4)
        comp = ring.cpu_acquire(0)
        self.assertEqual((comp.status, comp.bytes_done, comp.error),
                         ("ok", 4, None))
        self.assertEqual(ring.cpu_read_completion(0).status, "ok")

    def test_pending_writes_invisible_until_release(self):
        ring = make_ring()
        ring.cpu_write(0, offset=5, length=3)
        # device cannot see anything: no release happened
        with self.assertRaises(OwnershipError):
            ring.device_acquire(0)
        ring.cpu_release(0)
        self.assertEqual(ring.device_acquire(0).offset, 5)


class TestGenerationABA(unittest.TestCase):
    def test_stale_generation_detected_after_slot_reuse(self):
        ring = make_ring()
        write_and_release(ring, 0, offset=0, length=4)
        snap = ring.device_acquire(0, expected_generation=1)
        ring.device_release(0, status="ok", bytes_done=4)
        ring.cpu_acquire(0)
        # CPU recycles the same slot: generation advances.
        write_and_release(ring, 0, offset=16, length=4)
        self.assertEqual(ring.inspect(0)["generation"], 2)
        with self.assertRaises(GenerationError):
            ring.device_acquire(0, expected_generation=snap.generation)
        # Current generation still acquires fine.
        ring.device_acquire(0, expected_generation=2)

    def test_ring_wrap_reuse_increments_generation_each_release(self):
        ring = DmaRing(2, 16)
        for expected_gen in (1, 2, 3):
            write_and_release(ring, 1, offset=0, length=1)
            snap = ring.device_acquire(1)
            self.assertEqual(snap.generation, expected_gen)
            ring.device_release(1, status="ok", bytes_done=1)
            ring.cpu_acquire(1)


class TestChainsAndCopying(unittest.TestCase):
    def test_scatter_gather_chain_across_ring_wrap(self):
        ring = make_ring()
        # Chain wraps the ring: 6 -> 7 -> 0, out-of-order pool ranges.
        write_and_release(ring, 6, offset=64, length=8, next=7)
        write_and_release(ring, 7, offset=0, length=8, next=0)
        write_and_release(ring, 0, offset=128, length=8, next=None)
        engine = CopyEngine(ring)
        engine.submit_chain(6)
        engine.run()
        # Independent reference: hand-concatenated slices of SOURCE.
        expected = SOURCE[64:72] + SOURCE[0:8] + SOURCE[128:136]
        self.assertEqual(engine.output, expected)
        self.assertFalse(engine.halted)
        (result,) = engine.results
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.descriptors, (6, 7, 0))
        self.assertEqual(result.bytes_copied, 24)
        for idx in (6, 7, 0):
            self.assertEqual(ring.completion_count(idx), 1)
            self.assertEqual(ring.cpu_acquire(idx).status, "ok")

    def test_walk_chain_validates_whole_chain(self):
        ring = make_ring()
        write_and_release(ring, 3, offset=0, length=4, next=1)
        write_and_release(ring, 1, offset=4, length=4, next=None)
        self.assertEqual(ring.walk_chain(3), [3, 1])

    def test_missing_release_mid_chain_halts_and_skips_later_chains(self):
        ring = make_ring()
        write_and_release(ring, 0, offset=0, length=4, next=1)
        ring.cpu_write(1, offset=4, length=4)  # never released
        write_and_release(ring, 2, offset=8, length=4, next=None)
        engine = CopyEngine(ring)
        engine.submit_chain(0)
        engine.submit_chain(2)
        engine.run()
        self.assertTrue(engine.halted)
        (result,) = engine.results
        self.assertEqual(result.status, "error")
        self.assertEqual(result.error_index, 1)
        self.assertEqual(result.descriptors, (0,))  # prefix completed ok
        self.assertEqual(engine.output, SOURCE[0:4])
        # Subsequent chain not consumed while halted.
        self.assertEqual(engine.queued, 1)
        self.assertEqual(ring.completion_count(2), 0)
        # After the CPU releases the missing descriptor it can be submitted
        # as its own chain; processing resumes once the halt is cleared.
        ring.cpu_release(1)
        engine.submit_chain(1)
        engine.reset_halt()
        engine.run()
        # Queued order: chain head 2 (offset 8) ran before chain head 1.
        self.assertEqual(engine.output, SOURCE[0:4] + SOURCE[8:12] + SOURCE[4:8])
        self.assertEqual(len(engine.results), 3)

    def test_cycle_in_chain_error_completes(self):
        ring = make_ring()
        write_and_release(ring, 0, offset=0, length=4, next=1)
        write_and_release(ring, 1, offset=4, length=4, next=0)  # 1 -> 0 cycle
        engine = CopyEngine(ring)
        engine.submit_chain(0)
        engine.run()
        self.assertTrue(engine.halted)
        (result,) = engine.results
        self.assertEqual(result.status, "error")
        self.assertEqual(result.error_index, 1)
        self.assertEqual(result.descriptors, (0, 1))
        self.assertEqual(ring.cpu_acquire(0).status, "ok")
        bad = ring.cpu_acquire(1)
        self.assertEqual(bad.status, "error")
        self.assertEqual(bad.bytes_done, 0)
        with self.assertRaises(ChainError) as ctx:
            ring.walk_chain(0)
        self.assertEqual(ctx.exception.reason, "cycle")

    def test_self_loop_detected(self):
        ring = make_ring()
        write_and_release(ring, 5, offset=0, length=4, next=5)
        engine = CopyEngine(ring)
        engine.submit_chain(5)
        engine.run()
        (result,) = engine.results
        self.assertEqual(result.status, "error")
        self.assertEqual(result.error_index, 5)
        self.assertEqual(engine.output, b"")

    def test_buffer_out_of_range_error_completes_chain(self):
        ring = DmaRing(4, 16)
        ring.pool[:] = SOURCE[:16]
        write_and_release(ring, 0, offset=0, length=8, next=1)
        write_and_release(ring, 1, offset=8, length=16, next=None)  # 8+16 > 16
        engine = CopyEngine(ring)
        engine.submit_chain(0)
        engine.run()
        (result,) = engine.results
        self.assertEqual(result.status, "error")
        self.assertEqual(result.error_index, 1)
        self.assertEqual(engine.output, SOURCE[0:8])  # half-chain completed
        self.assertEqual(ring.cpu_acquire(1).status, "error")
        with self.assertRaises(ChainError) as ctx:
            ring.walk_chain(0)
        self.assertEqual(ctx.exception.reason, "buffer-out-of-range")
        self.assertEqual(ctx.exception.index, 1)

    def test_next_out_of_range(self):
        ring = make_ring()
        write_and_release(ring, 0, offset=0, length=4, next=99)
        engine = CopyEngine(ring)
        engine.submit_chain(0)
        engine.run()
        (result,) = engine.results
        self.assertEqual(result.status, "error")
        self.assertEqual(result.error_index, 0)
        self.assertEqual(engine.output, b"")
        with self.assertRaises(ChainError):
            ring.walk_chain(0)

    def test_completion_counts_accumulate_over_reuse(self):
        ring = make_ring()
        engine = CopyEngine(ring)
        for _ in range(3):
            write_and_release(ring, 0, offset=0, length=4)
            engine.submit_chain(0)
            engine.run()
            ring.cpu_acquire(0)
        self.assertEqual(ring.completion_count(0), 3)
        self.assertEqual(engine.output, SOURCE[0:4] * 3)

    def test_submit_chain_requires_device_ownership(self):
        ring = make_ring()
        ring.cpu_write(0, offset=0, length=4)
        engine = CopyEngine(ring)
        with self.assertRaises(OwnershipError):
            engine.submit_chain(0)

    def test_zero_length_descriptor_copies_nothing(self):
        ring = make_ring()
        write_and_release(ring, 0, offset=10, length=0, next=None)
        engine = CopyEngine(ring)
        engine.submit_chain(0)
        engine.run()
        self.assertEqual(engine.output, b"")
        self.assertEqual(ring.cpu_acquire(0).status, "ok")

    def test_device_release_validation(self):
        ring = make_ring()
        write_and_release(ring, 0, offset=0, length=4)
        ring.device_acquire(0)
        with self.assertRaises(ValueError):
            ring.device_release(0, status="bogus", bytes_done=4)
        with self.assertRaises(ValueError):
            ring.device_release(0, status="error")  # missing error message
        with self.assertRaises(ValueError):
            ring.device_release(0, status="ok", bytes_done=5)  # > length
        with self.assertRaises(TypeError):
            ring.device_release(0, status="ok", bytes_done=True)
        # Failed releases leave the slot device-held.
        self.assertTrue(ring.inspect(0)["device_held"])
        ring.device_release(0, status="ok", bytes_done=4)
        self.assertEqual(ring.cpu_acquire(0).status, "ok")


if __name__ == "__main__":
    unittest.main()
