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
    RequestEngine,
    RequestError,
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


# ---------------------------------------------------------------------------
# Multi-descriptor requests: out-of-order completion, generation checks,
# error-release semantics.  Expected byte strings are hand-computed slices
# of SOURCE, never produced by the implementation under test.
# ---------------------------------------------------------------------------

def release_descriptors(ring, specs):
    """specs: list of (index, offset, length)."""
    for index, offset, length in specs:
        write_and_release(ring, index, offset=offset, length=length,
                          next=None)


def recycle(ring, indices):
    """CPU acquires finished slots so they can be written again."""
    for idx in indices:
        ring.cpu_acquire(idx)


class TestRequestNormal(unittest.TestCase):
    def test_out_of_order_completion_publishes_only_at_end(self):
        ring = make_ring()
        release_descriptors(ring, [(2, 0, 4), (5, 8, 4), (7, 16, 4)])
        engine = RequestEngine(ring)
        snaps = engine.submit_request([2, 5, 7])
        self.assertEqual([s.index for s in snaps], [2, 5, 7])
        self.assertEqual(engine.pending, 1)
        self.assertEqual(engine.results, [])

        # Complete in scrambled order; no result until the last one.
        self.assertIsNone(
            engine.complete_descriptor(0, 5, generation=1, bytes_done=4))
        self.assertIsNone(
            engine.complete_descriptor(0, 7, generation=1, bytes_done=4))
        self.assertEqual(engine.results, [])
        self.assertEqual(engine.pending, 1)
        result = engine.complete_descriptor(0, 2, generation=1,
                                            bytes_done=4)

        # output follows actual (out-of-order) completion order: 5 then 7
        # then 2.
        self.assertEqual(engine.output,
                         SOURCE[8:12] + SOURCE[16:20] + SOURCE[0:4])
        # Result descriptors follow submission order.
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.request_id, 0)
        self.assertEqual([d.index for d in result.descriptors], [2, 5, 7])
        self.assertTrue(all(d.status == "ok" for d in result.descriptors))
        self.assertEqual(result.bytes_copied, 12)
        self.assertEqual(engine.pending, 0)
        self.assertEqual(engine.results, [result])
        for idx in (2, 5, 7):
            self.assertEqual(ring.completion_count(idx), 1)
            self.assertEqual(ring.cpu_acquire(idx).status, "ok")

    def test_single_descriptor_request_is_boundary_case(self):
        ring = make_ring()
        release_descriptors(ring, [(3, 40, 5)])
        engine = RequestEngine(ring)
        engine.submit_request([3])
        result = engine.complete_descriptor(0, 3, generation=1, bytes_done=5)
        self.assertEqual(result.status, "ok")
        self.assertEqual([d.index for d in result.descriptors], [3])
        self.assertEqual(engine.output, SOURCE[40:45])

    def test_zero_length_descriptor_counts_as_completed(self):
        ring = make_ring()
        release_descriptors(ring, [(0, 10, 0), (1, 10, 3)])
        engine = RequestEngine(ring)
        engine.submit_request([0, 1])
        engine.complete_descriptor(0, 0, generation=1, bytes_done=0)
        result = engine.complete_descriptor(0, 1, generation=1,
                                            bytes_done=3)
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.bytes_copied, 3)
        self.assertEqual(engine.output, SOURCE[10:13])

    def test_request_fills_entire_ring(self):
        ring = make_ring(capacity=4, pool_size=256)
        specs = [(i, i * 4, 4) for i in range(4)]
        release_descriptors(ring, specs)
        engine = RequestEngine(ring)
        snaps = engine.submit_request([0, 1, 2, 3])
        self.assertEqual(len(snaps), 4)
        result = None
        for idx in (3, 1, 0, 2):  # arbitrary order
            result = engine.complete_descriptor(0, idx, generation=1,
                                                bytes_done=4)
        self.assertEqual(result.status, "ok")
        expected = b"".join(SOURCE[i * 4:i * 4 + 4] for i in (3, 1, 0, 2))
        self.assertEqual(engine.output, expected)
        self.assertEqual(result.bytes_copied, 16)

    def test_two_concurrent_requests_finish_independently(self):
        ring = make_ring()
        release_descriptors(ring, [(0, 0, 4), (1, 4, 4),
                                   (2, 8, 4), (3, 12, 4)])
        engine = RequestEngine(ring)
        engine.submit_request([0, 1], request_id=10)
        engine.submit_request([2, 3], request_id=11)
        self.assertEqual(engine.pending, 2)
        # Request 11 finishes first even though submitted second.
        r11 = engine.complete_descriptor(11, 2, generation=1, bytes_done=4)
        self.assertIsNone(r11)
        r11 = engine.complete_descriptor(11, 3, generation=1, bytes_done=4)
        self.assertEqual(r11.request_id, 11)
        self.assertEqual(engine.pending, 1)
        r10 = engine.complete_descriptor(10, 1, generation=1, bytes_done=4)
        r10 = engine.complete_descriptor(10, 0, generation=1, bytes_done=4)
        self.assertEqual([r.request_id for r in engine.results], [11, 10])
        self.assertEqual(engine.output,
                         SOURCE[8:12] + SOURCE[12:16]
                         + SOURCE[4:8] + SOURCE[0:4])


class TestRequestGenerations(unittest.TestCase):
    def test_wrong_generation_rejected_as_stale(self):
        ring = make_ring()
        release_descriptors(ring, [(0, 0, 4), (1, 4, 4)])
        engine = RequestEngine(ring)
        engine.submit_request([0, 1])
        for wrong_gen in (0, 2):
            with self.assertRaises(RequestError) as ctx:
                engine.complete_descriptor(0, 0, generation=wrong_gen,
                                           bytes_done=4)
            self.assertEqual(ctx.exception.reason, "stale-generation")
            self.assertEqual(ctx.exception.index, 0)
        # Rejected calls change nothing: slot still held, request pending.
        self.assertTrue(ring.inspect(0)["device_held"])
        self.assertEqual(engine.request_snapshot(0)["remaining"], (0, 1))
        self.assertEqual(engine.output, b"")
        # Correct generation still works.
        engine.complete_descriptor(0, 0, generation=1, bytes_done=4)
        engine.complete_descriptor(0, 1, generation=1, bytes_done=4)
        self.assertEqual(engine.results[-1].status, "ok")

    def test_stale_completion_after_ring_wrap_reuse(self):
        ring = DmaRing(2, 16)
        ring.pool[:] = SOURCE[:16]
        release_descriptors(ring, [(0, 0, 2), (1, 2, 2)])
        engine = RequestEngine(ring)
        first = engine.submit_request([0, 1])
        self.assertEqual([s.generation for s in first], [1, 1])
        engine.complete_descriptor(0, 0, generation=1, bytes_done=2)
        engine.complete_descriptor(0, 1, generation=1, bytes_done=2)

        # CPU recycles both slots around the ring: generation 2.
        recycle(ring, [0, 1])
        release_descriptors(ring, [(0, 4, 2), (1, 6, 2)])
        second = engine.submit_request([0, 1])
        self.assertEqual([s.generation for s in second], [2, 2])
        with self.assertRaises(RequestError) as ctx:
            engine.complete_descriptor(1, 0, generation=1, bytes_done=2)
        self.assertEqual(ctx.exception.reason, "stale-generation")
        # New generation accepted.
        engine.complete_descriptor(1, 0, generation=2, bytes_done=2)
        result = engine.complete_descriptor(1, 1, generation=2,
                                            bytes_done=2)
        self.assertEqual(result.status, "ok")
        self.assertEqual(ring.completion_count(0), 2)

    def test_duplicate_completion_rejected(self):
        ring = make_ring()
        release_descriptors(ring, [(0, 0, 4), (1, 4, 4)])
        engine = RequestEngine(ring)
        engine.submit_request([0, 1])
        engine.complete_descriptor(0, 0, generation=1, bytes_done=4)
        with self.assertRaises(RequestError) as ctx:
            engine.complete_descriptor(0, 0, generation=1, bytes_done=4)
        self.assertEqual(ctx.exception.reason, "duplicate-completion")
        # No double bytes, no premature publication.
        self.assertEqual(engine.output, SOURCE[0:4])
        self.assertEqual(engine.pending, 1)
        self.assertEqual(ring.completion_count(0), 1)
        engine.complete_descriptor(0, 1, generation=1, bytes_done=4)
        self.assertEqual(engine.results[-1].status, "ok")

    def test_completion_after_request_finished_is_unknown_request(self):
        ring = make_ring()
        release_descriptors(ring, [(0, 0, 4)])
        engine = RequestEngine(ring)
        engine.submit_request([0])
        engine.complete_descriptor(0, 0, generation=1, bytes_done=4)
        with self.assertRaises(RequestError) as ctx:
            engine.complete_descriptor(0, 0, generation=1, bytes_done=4)
        self.assertEqual(ctx.exception.reason, "unknown-request")

    def test_index_not_in_request_rejected(self):
        ring = make_ring()
        release_descriptors(ring, [(0, 0, 4), (1, 4, 4)])
        engine = RequestEngine(ring)
        engine.submit_request([0], request_id=5)
        engine.submit_request([1], request_id=6)
        with self.assertRaises(RequestError) as ctx:
            engine.complete_descriptor(5, 1, generation=1, bytes_done=4)
        self.assertEqual(ctx.exception.reason, "not-in-request")
        # Request 6's descriptor is untouched.
        self.assertEqual(engine.request_snapshot(6)["remaining"], (1,))

    def test_generation_bool_rejected(self):
        ring = make_ring()
        release_descriptors(ring, [(0, 0, 4)])
        engine = RequestEngine(ring)
        engine.submit_request([0])
        with self.assertRaises(TypeError):
            engine.complete_descriptor(0, 0, generation=True, bytes_done=4)


class TestRequestErrorsAndRelease(unittest.TestCase):
    def test_error_descriptor_skips_remaining_and_releases_slots(self):
        ring = make_ring()
        release_descriptors(ring, [(0, 0, 4), (1, 4, 4),
                                   (2, 8, 4), (3, 12, 4)])
        engine = RequestEngine(ring)
        engine.submit_request([0, 1, 2, 3])
        # Two descriptors finish ok (half-completed request), then an error.
        engine.complete_descriptor(0, 1, generation=1, bytes_done=4)
        engine.complete_descriptor(0, 3, generation=1, bytes_done=4)
        result = engine.complete_descriptor(
            0, 0, generation=1, status="error", error="device fault")
        self.assertEqual(result.status, "error")
        self.assertEqual(result.error_index, 0)
        self.assertEqual(result.error, "device fault")
        statuses = {d.index: d.status for d in result.descriptors}
        self.assertEqual(statuses, {0: "error", 1: "ok",
                                    2: "skipped", 3: "ok"})
        # Only the two ok descriptors copied bytes.
        self.assertEqual(result.bytes_copied, 8)
        self.assertEqual(engine.output, SOURCE[4:8] + SOURCE[12:16])
        # Request is gone and every occupied slot is back with the CPU.
        self.assertEqual(engine.pending, 0)
        for idx in (0, 1, 2, 3):
            info = ring.inspect(idx)
            self.assertEqual(info["owner"], "cpu")
            self.assertFalse(info["device_held"])
        self.assertEqual(ring.cpu_acquire(2).status, "skipped")
        self.assertEqual(ring.completion_count(2), 1)
        self.assertEqual(ring.cpu_acquire(0).status, "error")

    def test_error_on_first_descriptor_skips_all_others(self):
        ring = make_ring()
        release_descriptors(ring, [(0, 0, 4), (1, 4, 4), (2, 8, 4)])
        engine = RequestEngine(ring)
        engine.submit_request([0, 1, 2])
        result = engine.complete_descriptor(
            0, 1, generation=1, status="error", error="bad")
        self.assertEqual([d.status for d in result.descriptors],
                         ["skipped", "error", "skipped"])
        self.assertEqual(result.bytes_copied, 0)
        self.assertEqual(engine.output, b"")
        for idx in (0, 1, 2):
            self.assertEqual(ring.inspect(idx)["owner"], "cpu")

    def test_other_concurrent_request_unaffected_by_abort(self):
        ring = make_ring()
        release_descriptors(ring, [(0, 0, 4), (1, 4, 4), (2, 8, 4)])
        engine = RequestEngine(ring)
        engine.submit_request([0, 2], request_id=7)
        engine.submit_request([1], request_id=8)
        bad = engine.complete_descriptor(
            7, 0, generation=1, status="error", error="fault")
        self.assertEqual(bad.status, "error")
        # Request 8 still pending; finish it normally.
        ok = engine.complete_descriptor(8, 1, generation=1, bytes_done=4)
        self.assertEqual(ok.status, "ok")
        self.assertEqual(engine.output, SOURCE[4:8])

    def test_invalid_error_payload_rejected_without_state_change(self):
        ring = make_ring()
        release_descriptors(ring, [(0, 0, 4), (1, 4, 4)])
        engine = RequestEngine(ring)
        engine.submit_request([0, 1])
        # error status needs a message: ring validation propagates.
        with self.assertRaises(ValueError):
            engine.complete_descriptor(0, 0, generation=1, status="error")
        # bytes_done beyond descriptor length likewise.
        with self.assertRaises(ValueError):
            engine.complete_descriptor(0, 0, generation=1, bytes_done=5)
        # Nothing happened: descriptor still pending and device-held.
        self.assertEqual(engine.request_snapshot(0)["remaining"], (0, 1))
        self.assertTrue(ring.inspect(0)["device_held"])
        self.assertEqual(engine.output, b"")
        # Same descriptor can then complete ok.
        result = engine.complete_descriptor(0, 0, generation=1, bytes_done=4)
        self.assertIsNone(result)
        engine.complete_descriptor(0, 1, generation=1, bytes_done=4)
        self.assertEqual(engine.results[-1].status, "ok")

    def test_unknown_request_completion_rejected(self):
        ring = make_ring()
        release_descriptors(ring, [(0, 0, 4)])
        engine = RequestEngine(ring)
        engine.submit_request([0])
        with self.assertRaises(RequestError) as ctx:
            engine.complete_descriptor(99, 0, generation=1, bytes_done=4)
        self.assertEqual(ctx.exception.reason, "unknown-request")
        # The pending request is untouched.
        self.assertEqual(engine.request_snapshot(0)["remaining"], (0,))


class TestRequestSubmissionValidation(unittest.TestCase):
    def test_empty_request_rejected(self):
        ring = make_ring()
        engine = RequestEngine(ring)
        with self.assertRaises(RequestError) as ctx:
            engine.submit_request([])
        self.assertEqual(ctx.exception.reason, "empty-request")
        self.assertEqual(engine.pending, 0)

    def test_duplicate_descriptor_in_one_submission_rejected(self):
        ring = make_ring()
        release_descriptors(ring, [(0, 0, 4), (1, 4, 4)])
        engine = RequestEngine(ring)
        with self.assertRaises(RequestError) as ctx:
            engine.submit_request([0, 1, 0])
        self.assertEqual(ctx.exception.reason, "duplicate-descriptor")
        self.assertFalse(ring.inspect(0)["device_held"])
        self.assertFalse(ring.inspect(1)["device_held"])

    def test_descriptor_not_released_rejects_whole_request(self):
        ring = make_ring()
        write_and_release(ring, 0, offset=0, length=4)
        ring.cpu_write(1, offset=4, length=4)  # staged, never released
        engine = RequestEngine(ring)
        with self.assertRaises(RequestError) as ctx:
            engine.submit_request([0, 1])
        self.assertEqual(ctx.exception.reason, "descriptor-unavailable")
        # Atomicity: the otherwise-valid descriptor 0 was not acquired.
        self.assertFalse(ring.inspect(0)["device_held"])
        self.assertEqual(engine.pending, 0)
        # It can still be submitted once the bad one is fixed.
        ring.cpu_release(1)
        engine.submit_request([0, 1])
        self.assertEqual(engine.pending, 1)

    def test_buffer_out_of_range_rejected_at_submission(self):
        ring = DmaRing(4, 16)
        ring.pool[:] = SOURCE[:16]
        write_and_release(ring, 0, offset=0, length=8)
        write_and_release(ring, 1, offset=8, length=16)  # 8+16 > 16
        engine = RequestEngine(ring)
        with self.assertRaises(RequestError) as ctx:
            engine.submit_request([0, 1])
        self.assertEqual(ctx.exception.reason, "buffer-out-of-range")
        self.assertEqual(ctx.exception.index, 1)
        self.assertFalse(ring.inspect(0)["device_held"])

    def test_descriptor_already_in_another_request_rejected(self):
        ring = make_ring()
        release_descriptors(ring, [(0, 0, 4), (1, 4, 4)])
        engine = RequestEngine(ring)
        engine.submit_request([0])
        with self.assertRaises(RequestError) as ctx:
            engine.submit_request([0, 1])
        self.assertEqual(ctx.exception.reason, "descriptor-busy")
        # Descriptor 1 was not grabbed by the failed submission.
        self.assertFalse(ring.inspect(1)["device_held"])
        self.assertEqual(engine.request_snapshot(0)["remaining"], (0,))

    def test_failed_submission_does_not_consume_request_id(self):
        ring = make_ring()
        engine = RequestEngine(ring)
        with self.assertRaises(RequestError):
            engine.submit_request([])  # would have consumed id 0
        write_and_release(ring, 0, offset=0, length=4)
        engine.submit_request([0])  # must still be id 0
        result = engine.complete_descriptor(0, 0, generation=1, bytes_done=4)
        self.assertEqual(result.request_id, 0)

    def test_explicit_request_id_rules(self):
        ring = make_ring()
        release_descriptors(ring, [(0, 0, 4), (1, 4, 4)])
        engine = RequestEngine(ring)
        engine.submit_request([0], request_id=3)
        with self.assertRaises(RequestError):
            engine.submit_request([1], request_id=3)
        with self.assertRaises(ValueError):
            engine.submit_request([1], request_id=-1)
        with self.assertRaises(TypeError):
            engine.submit_request([1], request_id=True)
        # Auto ids continue above the highest explicit id used.
        engine.submit_request([1])
        self.assertEqual(engine.request_snapshot(4)["order"], (1,))

    def test_bad_index_in_submission_locates_error(self):
        ring = make_ring()
        engine = RequestEngine(ring)
        with self.assertRaises(IndexError):
            engine.submit_request([8])
        with self.assertRaises(TypeError):
            engine.submit_request([True])


if __name__ == "__main__":
    unittest.main()
