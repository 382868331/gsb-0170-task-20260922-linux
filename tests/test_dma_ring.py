"""Tests for dma_ring.  Expected values are derived independently of the
core (hand-computed slices of the deterministic source data), never by
calling the implementation under test."""

import math
import unittest

from dma_ring import (
    ChainError,
    CopyEngine,
    DmaRing,
    DuplicateCompletionError,
    GenerationError,
    Owner,
    OwnershipError,
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


class TestRequests(unittest.TestCase):
    """Multi-descriptor requests: out-of-order completion, generations,
    duplicate rejection, error cleanup and ring wrap.

    Expected bytes are hand-specified slices of SOURCE; the aggregate
    results are recomputed here from those slices, never taken from the
    implementation under test.
    """

    # member index -> (offset, length, hand-computed expected bytes)
    MEMBERS = {
        4: (0, 4, SOURCE[0:4]),
        6: (32, 8, SOURCE[32:40]),
        2: (128, 2, SOURCE[128:130]),
    }
    ORDER = [4, 6, 2]

    def _setup_request(self, ring=None, indices=None):
        ring = ring or make_ring()
        indices = list(self.ORDER if indices is None else indices)
        for idx in indices:
            off, ln, _ = self.MEMBERS[idx]
            write_and_release(ring, idx, offset=off, length=ln)
        rid, gens = ring.register_request(indices)
        return ring, rid, gens, indices

    def _complete_ok(self, ring, idx, rid):
        snap = ring.device_acquire(idx)
        ring.device_release(idx, status="ok", bytes_done=snap.length,
                            request_id=rid)

    # -- normal behaviour ---------------------------------------------------

    def test_result_published_only_after_all_members_complete(self):
        ring, rid, gens, indices = self._setup_request()
        self.assertEqual(gens, (1, 1, 1))
        self.assertFalse(ring.request_is_done(rid))
        with self.assertRaises(RequestError):
            ring.request_acquire(rid)
        self._complete_ok(ring, 4, rid)
        self._complete_ok(ring, 6, rid)
        # Two of three done: still no aggregate result, slots already back.
        self.assertFalse(ring.request_is_done(rid))
        self.assertEqual(ring.inspect(4)["owner"], "cpu")
        with self.assertRaises(RequestError):
            ring.request_acquire(rid)
        self._complete_ok(ring, 2, rid)
        self.assertTrue(ring.request_is_done(rid))
        result = ring.request_acquire(rid)
        # Independent expectations, recomputed from the hand-picked slices.
        expected = [self.MEMBERS[i][2] for i in self.ORDER]
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.error_index, None)
        self.assertEqual(result.bytes_copied, sum(len(b) for b in expected))
        self.assertEqual([c.index for c in result.completions], self.ORDER)
        self.assertEqual(
            [(c.status, c.bytes_done) for c in result.completions],
            [("ok", 4), ("ok", 8), ("ok", 2)])
        for i, chunk in zip(self.ORDER, expected):
            comp = ring.cpu_acquire(i)
            self.assertEqual(comp.status, "ok")
            self.assertEqual(comp.bytes_done, len(chunk))
            self.assertEqual(ring.completion_count(i), 1)

    def test_out_of_order_result_still_in_membership_order(self):
        ring, rid, _, _ = self._setup_request()
        for idx in (2, 4, 6):           # reversed completion order
            self._complete_ok(ring, idx, rid)
        result = ring.request_acquire(rid)
        self.assertEqual([c.index for c in result.completions], [4, 6, 2])

    def test_engine_out_of_order_request_output(self):
        ring = make_ring()
        for idx in self.ORDER:
            off, ln, _ = self.MEMBERS[idx]
            write_and_release(ring, idx, offset=off, length=ln)
        engine = CopyEngine(ring)
        rid = engine.submit_request([4, 6, 2], order=[2, 4, 6])
        engine.run()
        # Output follows device completion order, computed independently:
        # chunk(2) then chunk(4) then chunk(6).
        chunks = {i: self.MEMBERS[i][2] for i in self.ORDER}
        self.assertEqual(engine.output, chunks[2] + chunks[4] + chunks[6])
        (result,) = engine.request_results
        self.assertEqual(result.request_id, rid)
        self.assertEqual(result.status, "ok")
        # Aggregate bytes are order-independent.
        self.assertEqual(result.bytes_copied, 4 + 8 + 2)
        self.assertEqual([c.index for c in result.completions], [4, 6, 2])

    def test_all_permutations_of_completion_order(self):
        """Small exhaustive check over all 3! completion orders: every order
        yields the same ok aggregate and the independently expected output
        for that order."""
        import itertools
        chunks = {i: self.MEMBERS[i][2] for i in self.ORDER}
        for perm in itertools.permutations(self.ORDER):
            ring = make_ring()
            for idx in self.ORDER:
                off, ln, _ = self.MEMBERS[idx]
                write_and_release(ring, idx, offset=off, length=ln)
            engine = CopyEngine(ring)
            engine.submit_request(list(self.ORDER), order=list(perm))
            engine.run()
            self.assertEqual(
                engine.output, b"".join(chunks[i] for i in perm),
                msg=f"order={perm}")
            (result,) = engine.request_results
            self.assertEqual(result.status, "ok")
            self.assertEqual(result.bytes_copied, 14)
            self.assertEqual(
                [c.index for c in result.completions], [4, 6, 2])

    def test_single_descriptor_request(self):
        ring = make_ring()
        write_and_release(ring, 3, offset=16, length=5)
        rid, gens = ring.register_request([3])
        self.assertEqual(gens, (1,))
        self._complete_ok(ring, 3, rid)
        result = ring.request_acquire(rid)
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.bytes_copied, 5)
        self.assertEqual(
            [(c.index, c.status) for c in result.completions], [(3, "ok")])

    # -- ring wrap / reuse --------------------------------------------------

    def test_request_spans_full_ring_then_slots_wrap(self):
        ring = DmaRing(4, 64)
        ring.pool[:] = SOURCE[:64]
        for idx in range(4):
            write_and_release(ring, idx, offset=4 * idx, length=4)
        rid, gens = ring.register_request([0, 1, 2, 3])
        self.assertEqual(gens, (1, 1, 1, 1))
        for idx in (3, 1, 0, 2):
            self._complete_ok(ring, idx, rid)
        result = ring.request_acquire(rid)
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.bytes_copied, 16)
        # Every slot is back with the CPU and unowned: the ring can wrap.
        for idx in range(4):
            info = ring.inspect(idx)
            self.assertEqual(info["owner"], "cpu")
            self.assertIsNone(info["request_id"])
            ring.cpu_acquire(idx)
        # Recycle all four slots into a second request: generations are 2.
        for idx in range(4):
            write_and_release(ring, idx, offset=4 * idx, length=2)
        rid2, gens2 = ring.register_request([3, 2, 1, 0])
        self.assertEqual(gens2, (2, 2, 2, 2))
        for idx in (0, 1, 2, 3):
            self._complete_ok(ring, idx, rid2)
        result2 = ring.request_acquire(rid2)
        self.assertEqual(result2.bytes_copied, 8)
        for idx in range(4):
            self.assertEqual(ring.completion_count(idx), 2)

    # -- generation / stale completion rejection ---------------------------

    def test_stale_completion_after_slot_recycled_rejected(self):
        ring, rid, _, _ = self._setup_request()
        self._complete_ok(ring, 4, rid)        # member 4 done at generation 1
        # CPU illegally/abusively recycles slot 4 before the rest finish.
        ring.cpu_acquire(4)
        ring.cpu_write(4, offset=60, length=1)
        ring.cpu_release(4)                    # slot 4 now generation 2
        with self.assertRaises(GenerationError):
            ring.device_release(
                4, status="ok", bytes_done=1, request_id=rid)
        # The stale replay changed nothing: the recycled slot is untouched,
        # the request is still waiting on its other two members.
        self.assertEqual(ring.inspect(4)["generation"], 2)
        self.assertFalse(ring.request_is_done(rid))
        # Remaining members complete normally; member 4 keeps its gen-1 ok.
        self._complete_ok(ring, 6, rid)
        self._complete_ok(ring, 2, rid)
        result = ring.request_acquire(rid)
        self.assertEqual(result.status, "ok")
        by_index = {c.index: c for c in result.completions}
        self.assertEqual(by_index[4].generation, 1)
        self.assertEqual(by_index[4].status, "ok")

    def test_stale_replay_after_finish_and_recycle_rejected(self):
        ring, rid, _, _ = self._setup_request()
        for idx in self.ORDER:
            self._complete_ok(ring, idx, rid)
        # Slot 4 wraps to generation 2 and is handed out again.
        ring.cpu_acquire(4)
        ring.cpu_write(4, offset=0, length=1)
        ring.cpu_release(4)
        with self.assertRaises(GenerationError):
            ring.device_release(
                4, status="ok", bytes_done=1, request_id=rid)
        # The recycled descriptor is untouched by the rejected replay.
        self.assertEqual(ring.inspect(4)["owner"], "device")
        self.assertEqual(ring.inspect(4)["generation"], 2)

    # -- duplicate completion rejection ------------------------------------

    def test_duplicate_completion_rejected(self):
        ring, rid, _, _ = self._setup_request()
        self._complete_ok(ring, 4, rid)
        # Replayed completion for the same (request, slot, generation): the
        # slot is back with the CPU, but the request-scoped duplicate check
        # fires before the ownership check.
        with self.assertRaises(DuplicateCompletionError) as ctx:
            ring.device_release(4, status="ok", bytes_done=4,
                                request_id=rid)
        self.assertEqual(ctx.exception.request_id, rid)
        self.assertEqual(ctx.exception.index, 4)
        self.assertEqual(ctx.exception.generation, 1)
        # Exactly one completion recorded; the request is still in flight.
        self.assertEqual(ring.completion_count(4), 1)
        self.assertFalse(ring.request_is_done(rid))
        # The other members still finish the request normally.
        self._complete_ok(ring, 6, rid)
        self._complete_ok(ring, 2, rid)
        self.assertTrue(ring.request_is_done(rid))

    def test_replay_after_error_finalisation_is_duplicate(self):
        ring, rid, _, _ = self._setup_request()
        ring.device_acquire(4)
        ring.device_release(4, status="error", error="boom",
                            bytes_done=0, request_id=rid)
        # Request already finalised (others skipped): replaying the error
        # member at the same generation is a duplicate, not a new error.
        with self.assertRaises(DuplicateCompletionError):
            ring.device_release(4, status="ok", bytes_done=4,
                                request_id=rid)

    # -- error requests release resources / half-completed -----------------

    def test_error_completion_skips_rest_and_frees_every_slot(self):
        ring = DmaRing(4, 64)
        ring.pool[:] = SOURCE[:64]
        specs = {0: (0, 4), 1: (8, 4), 2: (32, 64), 3: (16, 4)}  # 2 is bad
        for idx, (off, ln) in specs.items():
            write_and_release(ring, idx, offset=off, length=ln)
        rid, _ = ring.register_request([0, 1, 2, 3])
        # Members 0 and 1 complete ok first (half-completed request)...
        self._complete_ok(ring, 0, rid)
        self._complete_ok(ring, 1, rid)
        # ...then member 2 errors.
        ring.device_acquire(2)
        ring.device_release(2, status="error", error="bad range",
                            bytes_done=0, request_id=rid)
        # The request is finalised immediately; member 3 is force-skipped.
        result = ring.request_acquire(rid)
        self.assertEqual(result.status, "error")
        self.assertEqual(result.error_index, 2)
        self.assertEqual(result.error, "bad range")
        by_index = {c.index: c for c in result.completions}
        self.assertEqual(
            [(i, by_index[i].status, by_index[i].bytes_done)
             for i in range(4)],
            [(0, "ok", 4), (1, "ok", 4), (2, "error", 0),
             (3, "skipped", 0)])
        self.assertEqual(result.bytes_copied, 8)
        # All resources released: every slot back to the CPU, ring can wrap.
        expected_status = {0: "ok", 1: "ok", 2: "error", 3: "skipped"}
        for idx in range(4):
            info = ring.inspect(idx)
            self.assertEqual(info["owner"], "cpu", msg=f"slot {idx}")
            self.assertIsNone(info["request_id"])
            self.assertEqual(info["device_held"], False)
            comp = ring.cpu_acquire(idx)     # hand back to the CPU
            self.assertEqual(comp.status, expected_status[idx])
            ring.cpu_write(idx, offset=0, length=1)
            ring.cpu_release(idx)            # immediate reuse must work
        self.assertEqual(
            [ring.inspect(i)["generation"] for i in range(4)], [2] * 4)

    def test_engine_error_request_halts_and_skips_remaining(self):
        ring = DmaRing(8, 64)
        ring.pool[:] = SOURCE[:64]
        # completion order [1, 2, 0, 3]: bad member 2 is reached second;
        # member 1 copies before it, members 0 and 3 must be skipped.
        write_and_release(ring, 0, offset=0, length=4)
        write_and_release(ring, 1, offset=8, length=4)
        write_and_release(ring, 2, offset=32, length=64)   # out of pool
        write_and_release(ring, 3, offset=16, length=4)
        # A second healthy request queues behind the failing one; it must not
        # be consumed while the engine is halted.
        other = [5, 6]
        for idx in other:
            write_and_release(ring, idx, offset=0, length=1)
        engine = CopyEngine(ring)
        engine.submit_request([0, 1, 2, 3], order=[1, 2, 0, 3])
        engine.submit_request(other)
        engine.run()
        self.assertTrue(engine.halted)
        self.assertEqual(engine.queued, 1)
        (result,) = engine.request_results
        self.assertEqual(result.status, "error")
        self.assertEqual(result.error_index, 2)
        by_index = {c.index: c for c in result.completions}
        self.assertEqual(by_index[1].status, "ok")
        self.assertEqual(by_index[0].status, "skipped")
        self.assertEqual(by_index[3].status, "skipped")
        # Only member 1's 4 bytes were copied.
        self.assertEqual(engine.output, SOURCE[8:12])
        self.assertEqual(result.bytes_copied, 4)
        # The queued healthy request was not touched.
        for idx in other:
            info = ring.inspect(idx)
            self.assertEqual(info["completions"], 0)
        # Skipped slots are reusable: ring wrap needs no manual cleanup.
        for idx in (0, 3):
            self.assertEqual(ring.inspect(idx)["owner"], "cpu")

    def test_skipped_slots_wrap_into_next_request(self):
        ring = DmaRing(2, 32)
        ring.pool[:] = SOURCE[:32]
        write_and_release(ring, 0, offset=0, length=4)
        write_and_release(ring, 1, offset=16, length=64)   # bad
        engine = CopyEngine(ring)
        engine.submit_request([0, 1], order=[1, 0])       # error first
        engine.run()
        (bad,) = engine.request_results
        self.assertEqual([c.status for c in bad.completions],
                         ["skipped", "error"])
        # Both slots wrap immediately into a healthy request.
        for idx in range(2):
            ring.cpu_acquire(idx)
            write_and_release(ring, idx, offset=idx, length=1)
        engine.reset_halt()
        engine.submit_request([0, 1])
        engine.run()
        good = engine.request_results[-1]
        self.assertEqual(good.status, "ok")
        self.assertEqual(good.bytes_copied, 2)

    # -- registration / lifecycle validation -------------------------------

    def test_register_rejects_invalid_inputs_without_state_change(self):
        ring = make_ring()
        write_and_release(ring, 0, offset=0, length=4)
        with self.assertRaises(TypeError):
            ring.register_request({0})
        with self.assertRaises(ValueError):
            ring.register_request([])
        with self.assertRaises(IndexError):
            ring.register_request([0, 99])
        with self.assertRaises(RequestError) as ctx:
            ring.register_request([0, 0])
        self.assertEqual(ctx.exception.reason, "duplicate-member")
        # Slot 1 was only staged, never released: not device-owned.
        ring.cpu_write(1, offset=0, length=4)
        with self.assertRaises(RequestError) as ctx:
            ring.register_request([0, 1])
        self.assertEqual(ctx.exception.reason, "not-device-owned")
        self.assertEqual(ctx.exception.index, 1)
        # Nothing was registered: ids do not advance and slots stay unowned.
        self.assertIsNone(ring.inspect(0)["request_id"])
        write_and_release(ring, 1, offset=0, length=4)
        rid, _ = ring.register_request([0, 1])
        self.assertEqual(rid, 1)
        with self.assertRaises(RequestError) as ctx:
            ring.register_request([1])
        self.assertEqual(ctx.exception.reason, "already-in-request")

    def test_request_acquire_unknown_id(self):
        ring = make_ring()
        with self.assertRaises(RequestError):
            ring.request_acquire(123)
        with self.assertRaises(RequestError):
            ring.request_is_done(123)

    def test_release_must_name_owning_request(self):
        ring, rid, _, indices = self._setup_request()
        # A request-owned slot cannot be released as a plain descriptor.
        with self.assertRaises(RequestError) as ctx:
            ring.device_release(4, status="ok", bytes_done=4)
        self.assertEqual(ctx.exception.reason, "request-owned-slot")
        ring.device_acquire(4)
        # A slot that is not a member of this request cannot post to it
        # (slot 5 was never released: unknown ownership, but membership wins).
        write_and_release(ring, 5, offset=0, length=1)
        with self.assertRaises(RequestError) as ctx:
            ring.device_release(5, status="ok", bytes_done=1,
                                request_id=rid)
        self.assertEqual(ctx.exception.reason, "not-a-member")
        # Nonexistent request id.
        with self.assertRaises(RequestError) as ctx:
            ring.device_release(5, status="ok", bytes_done=1,
                                request_id=999)
        self.assertEqual(ctx.exception.reason, "unknown-request")

    def test_bad_order_rejected_atomically(self):
        ring = make_ring()
        for idx in (0, 1, 2):
            write_and_release(ring, idx, offset=0, length=1)
        engine = CopyEngine(ring)
        with self.assertRaises(RequestError):
            engine.submit_request([0, 1, 2], order=[0, 1])       # missing 2
        with self.assertRaises(RequestError):
            engine.submit_request([0, 1, 2], order=[0, 1, 1])    # dup/extra
        with self.assertRaises(TypeError):
            engine.submit_request([0, 1, 2], order="012")
        # Nothing registered or queued.
        self.assertEqual(engine.queued, 0)
        for idx in (0, 1, 2):
            self.assertIsNone(ring.inspect(idx)["request_id"])
        # The same members are then accepted in a valid order.
        engine.submit_request([0, 1, 2], order=[2, 0, 1])
        self.assertEqual(engine.queued, 1)

    def test_chain_cannot_touch_request_members(self):
        ring = make_ring()
        write_and_release(ring, 0, offset=0, length=4, next=1)
        write_and_release(ring, 1, offset=4, length=4)
        write_and_release(ring, 2, offset=8, length=4)
        rid, _ = ring.register_request([1, 2])
        engine = CopyEngine(ring)
        with self.assertRaises(RequestError):
            engine.submit_chain(1)                    # head is request member
        engine.submit_chain(0)                        # 0 -> 1 traverses in
        engine.run()
        self.assertTrue(engine.halted)
        (result,) = engine.results
        self.assertEqual(result.status, "error")
        self.assertEqual(result.error_index, 1)
        self.assertEqual(result.descriptors, (0,))
        # The request member was not acquired or completed by the chain.
        info = ring.inspect(1)
        self.assertEqual(info["owner"], "device")
        self.assertEqual(info["request_id"], rid)
        self.assertEqual(info["completions"], 0)

    def test_member_cannot_self_report_skipped(self):
        ring, rid, _, _ = self._setup_request()
        ring.device_acquire(4)
        with self.assertRaises(ValueError):
            ring.device_release(4, status="skipped", request_id=rid)
        # The rejected release left the slot device-held and the request
        # still in flight; it can still complete normally.
        self.assertTrue(ring.inspect(4)["device_held"])
        self.assertFalse(ring.request_is_done(rid))
        ring.device_release(4, status="ok", bytes_done=4, request_id=rid)
        self.assertEqual(ring.completion_count(4), 1)

    def test_request_completion_still_validates_bytes_done(self):
        ring, rid, _, _ = self._setup_request()
        ring.device_acquire(4)
        with self.assertRaises(ValueError):
            ring.device_release(4, status="ok", bytes_done=99,
                                request_id=rid)
        self.assertTrue(ring.inspect(4)["device_held"])
        self.assertEqual(ring.completion_count(4), 0)

    def test_zero_length_members(self):
        ring = make_ring()
        for idx in (0, 1):
            write_and_release(ring, idx, offset=10, length=0)
        engine = CopyEngine(ring)
        engine.submit_request([0, 1])
        engine.run()
        (result,) = engine.request_results
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.bytes_copied, 0)
        self.assertEqual(engine.output, b"")

    def test_error_cleanup_does_not_clobber_out_of_band_recycled_slot(self):
        # White-box: a pending member whose slot was moved on out-of-band is
        # still accounted as skipped but its recycled state is untouched.
        ring = make_ring()
        for idx in (0, 1, 2):
            write_and_release(ring, idx, offset=0, length=4)
        rid, _ = ring.register_request([0, 1, 2])
        # Simulate an out-of-band recycle of pending member 2.
        s2 = ring._slots[2]
        s2.request_id = None
        s2.generation = 5
        s2.owner = Owner.CPU
        s2.cpu_held = True
        # Member 0 ok, member 1 errors: cleanup must not touch slot 2.
        self._complete_ok(ring, 0, rid)
        ring.device_acquire(1)
        ring.device_release(1, status="error", error="boom",
                            bytes_done=0, request_id=rid)
        result = ring.request_acquire(rid)
        by_index = {c.index: c for c in result.completions}
        self.assertEqual(by_index[2].status, "skipped")
        self.assertEqual(by_index[2].generation, 1)   # recorded generation
        self.assertEqual(ring.inspect(2)["generation"], 5)  # slot untouched
        self.assertEqual(ring.inspect(2)["owner"], "cpu")

    def test_capacity_one_request(self):
        ring = DmaRing(1, 8)
        ring.pool[:] = SOURCE[:8]
        write_and_release(ring, 0, offset=2, length=3)
        engine = CopyEngine(ring)
        engine.submit_request([0])
        engine.run()
        (result,) = engine.request_results
        self.assertEqual(result.status, "ok")
        self.assertEqual(engine.output, SOURCE[2:5])


if __name__ == "__main__":
    unittest.main()
