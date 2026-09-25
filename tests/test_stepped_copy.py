"""Tests for SteppedCopyEngine: the step-driven, cancellable copy entry.

Expected values are derived independently of the core (hand-computed
slices of the deterministic source data, and a small hand-written
reference model for the event-sequence enumeration), never by calling the
implementation under test.
"""

import itertools
import unittest

from dma_ring import (
    ChainError,
    CopyEngine,
    DmaRing,
    OwnershipError,
    SlotBusyError,
    SteppedCopyEngine,
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


def submit_chain(engine, ring, specs):
    """Write+release a chain from [(index, offset, length), ...] and submit."""
    for pos, (index, offset, length) in enumerate(specs):
        nxt = specs[pos + 1][0] if pos + 1 < len(specs) else None
        write_and_release(ring, index, offset=offset, length=length, next=nxt)
    return engine.submit(specs[0][0])


class TestNormalCompletion(unittest.TestCase):
    def test_step_processes_one_descriptor_at_a_time(self):
        ring = make_ring()
        engine = SteppedCopyEngine(ring)
        req = submit_chain(engine, ring, [(6, 64, 8), (7, 0, 8), (0, 128, 8)])
        self.assertFalse(req.done)
        self.assertIsNone(req.result())
        expected = [SOURCE[64:72], SOURCE[0:8], SOURCE[128:136]]
        for i in range(3):
            self.assertTrue(engine.step())
            self.assertEqual(engine.output, b"".join(expected[:i + 1]))
            self.assertEqual(req.done, i == 2)
        self.assertFalse(engine.step())  # nothing left in flight

    def test_full_completion_result(self):
        ring = make_ring()
        engine = SteppedCopyEngine(ring)
        req = submit_chain(engine, ring, [(6, 64, 8), (7, 0, 8), (0, 128, 8)])
        engine.drain()
        result = req.result()
        self.assertEqual(result.status, "ok")
        self.assertIsNone(result.error)
        self.assertIsNone(result.error_index)
        self.assertEqual(result.bytes_copied, 24)
        self.assertEqual(result.descriptors, (6, 7, 0))
        self.assertEqual(result.skipped, ())
        self.assertEqual(engine.output, SOURCE[64:72] + SOURCE[0:8] + SOURCE[128:136])
        for idx in (6, 7, 0):
            self.assertEqual(ring.completion_count(idx), 1)
            self.assertEqual(ring.cpu_acquire(idx).status, "ok")
        self.assertEqual(engine.active, 0)
        self.assertEqual(len(engine.results), 1)


class TestCancellation(unittest.TestCase):
    def test_cancel_before_any_step(self):
        ring = make_ring()
        engine = SteppedCopyEngine(ring)
        req = submit_chain(engine, ring, [(0, 0, 4), (1, 4, 4)])
        self.assertTrue(req.cancel())
        self.assertTrue(engine.step())  # confirms the cancel
        result = req.result()
        self.assertEqual(result.status, "cancelled")
        self.assertEqual(result.bytes_copied, 0)
        self.assertEqual(result.descriptors, ())
        self.assertEqual(result.skipped, (0, 1))
        self.assertEqual(engine.output, b"")
        for idx in (0, 1):
            self.assertEqual(ring.cpu_acquire(idx).status, "skipped")

    def test_cancel_midway_keeps_prefix(self):
        ring = make_ring()
        engine = SteppedCopyEngine(ring)
        req = submit_chain(engine, ring, [(0, 0, 4), (1, 4, 4), (2, 8, 4)])
        engine.step()  # descriptor 0 copied
        self.assertEqual(engine.output, SOURCE[0:4])
        req.cancel()
        engine.step()  # confirms the cancel; descriptor 1 is NOT copied
        result = req.result()
        self.assertEqual(result.status, "cancelled")
        self.assertEqual(result.bytes_copied, 4)
        self.assertEqual(result.descriptors, (0,))
        self.assertEqual(result.skipped, (1, 2))
        self.assertEqual(engine.output, SOURCE[0:4])  # prefix kept
        self.assertEqual(ring.cpu_acquire(0).status, "ok")
        self.assertEqual(ring.cpu_acquire(1).status, "skipped")
        self.assertEqual(ring.cpu_acquire(2).status, "skipped")

    def test_cancel_registered_first_wins_over_next_copy(self):
        # Cancel is registered while one descriptor remains; the next step
        # must confirm the cancel instead of copying the last descriptor.
        ring = make_ring()
        engine = SteppedCopyEngine(ring)
        req = submit_chain(engine, ring, [(0, 0, 4), (1, 4, 4)])
        engine.step()  # descriptor 0 copied
        req.cancel()
        engine.step()
        self.assertEqual(req.result().status, "cancelled")
        self.assertEqual(req.result().bytes_copied, 4)
        self.assertEqual(ring.cpu_acquire(1).status, "skipped")

    def test_completion_first_wins_over_later_cancel(self):
        # The last descriptor completes before cancel is registered; the
        # success is final and the cancel cannot rewrite it.
        ring = make_ring()
        engine = SteppedCopyEngine(ring)
        req = submit_chain(engine, ring, [(0, 0, 4), (1, 4, 4)])
        engine.step()
        engine.step()  # last descriptor completes -> ok
        self.assertEqual(req.result().status, "ok")
        self.assertFalse(req.cancel())  # too late: no-op
        engine.step()  # nothing in flight
        self.assertEqual(req.result().status, "ok")
        self.assertEqual(req.result().bytes_copied, 8)
        self.assertEqual(len(engine.results), 1)
        self.assertEqual(ring.completion_count(0), 1)
        self.assertEqual(ring.completion_count(1), 1)

    def test_repeated_cancel_and_result_reads_are_idempotent(self):
        ring = make_ring()
        engine = SteppedCopyEngine(ring)
        req = submit_chain(engine, ring, [(0, 0, 4), (1, 4, 4)])
        self.assertTrue(req.cancel())
        self.assertFalse(req.cancel())  # already registered
        self.assertFalse(req.cancel())
        engine.step()
        first = req.result()
        self.assertEqual(first.status, "cancelled")
        self.assertFalse(req.cancel())  # finished: no-op
        for _ in range(3):
            self.assertIs(req.result(), first)
        self.assertEqual(len(engine.results), 1)
        self.assertEqual(ring.completion_count(0), 1)
        self.assertEqual(ring.completion_count(1), 1)

    def test_cpu_cannot_rewrite_slot_until_cancel_confirmed(self):
        ring = make_ring()
        engine = SteppedCopyEngine(ring)
        req = submit_chain(engine, ring, [(0, 0, 4), (1, 4, 4)])
        req.cancel()
        # Cancel is only registered: slots are still device-owned, so the
        # CPU may not take them back early.
        for idx in (0, 1):
            with self.assertRaises(OwnershipError):
                ring.cpu_write(idx, offset=0, length=1)
            with self.assertRaises(OwnershipError):
                ring.cpu_acquire(idx)
        engine.step()  # device confirms the cancel
        for idx in (0, 1):
            self.assertEqual(ring.cpu_acquire(idx).status, "skipped")
        # Now the CPU owns the slots again and may rewrite them.
        ring.cpu_write(0, offset=8, length=2)
        ring.cpu_write(1, offset=10, length=2)


class TestSlotFreezingAndConflicts(unittest.TestCase):
    def test_inflight_requests_may_not_share_slots(self):
        ring = make_ring()
        engine = SteppedCopyEngine(ring)
        submit_chain(engine, ring, [(0, 0, 4), (1, 4, 4)])
        # Overlapping head and overlapping tail are both rejected.
        write_and_release(ring, 2, offset=8, length=4, next=1)
        with self.assertRaises(SlotBusyError) as ctx:
            engine.submit(2)
        self.assertEqual(ctx.exception.index, 1)
        with self.assertRaises(SlotBusyError):
            engine.submit(0)
        # A fully disjoint chain is fine.
        write_and_release(ring, 3, offset=12, length=4, next=None)
        req_b = engine.submit(3)
        engine.drain()
        self.assertEqual(req_b.result().status, "ok")

    def test_failed_submit_leaves_no_partial_occupancy(self):
        ring = make_ring()
        engine = SteppedCopyEngine(ring)
        submit_chain(engine, ring, [(0, 0, 4)])
        # Chain 2 -> 0: slot 2 is free but slot 0 is occupied, so the
        # submit must fail without occupying slot 2.
        write_and_release(ring, 2, offset=8, length=4, next=0)  # collides
        with self.assertRaises(SlotBusyError):
            engine.submit(2)
        self.assertEqual(engine.active, 1)
        # Slot 2 was not left occupied by the rejected submit: re-arm
        # slot 0 for a new generation (slot 2's committed next still
        # points at it) and the same chain submits fine.
        engine.drain()
        ring.cpu_acquire(0)
        write_and_release(ring, 0, offset=12, length=4)
        req = engine.submit(2)
        engine.drain()
        self.assertEqual(req.result().status, "ok")
        self.assertEqual(req.result().bytes_copied, 8)
        self.assertEqual(engine.output, SOURCE[0:4] + SOURCE[8:12] + SOURCE[12:16])

    def test_submit_requires_release_and_valid_chain(self):
        ring = make_ring()
        engine = SteppedCopyEngine(ring)
        ring.cpu_write(0, offset=0, length=4)  # never released
        with self.assertRaises(OwnershipError):
            engine.submit(0)
        write_and_release(ring, 1, offset=0, length=4, next=1)  # self-cycle
        with self.assertRaises(ChainError):
            engine.submit(1)
        write_and_release(ring, 2, offset=0, length=4, next=99)  # bad next
        with self.assertRaises(ChainError):
            engine.submit(2)

    def test_wraparound_reuse_and_stale_handle(self):
        ring = DmaRing(2, 16)
        ring.pool[:] = SOURCE[:16]
        engine = SteppedCopyEngine(ring)
        # Generation 1: request completes ok.
        req1 = submit_chain(engine, ring, [(0, 0, 4), (1, 4, 4)])
        engine.drain()
        result1 = req1.result()
        self.assertEqual(result1.status, "ok")
        ring.cpu_acquire(0)
        ring.cpu_acquire(1)
        # Slots wrap and are recycled: generation 2, new request.
        req2 = submit_chain(engine, ring, [(0, 8, 4), (1, 12, 4)])
        self.assertNotEqual(req1.request_id, req2.request_id)
        self.assertEqual(ring.inspect(0)["generation"], 2)
        self.assertEqual(ring.inspect(1)["generation"], 2)
        # The stale handle still reports its own terminal state...
        self.assertIs(req1.result(), result1)
        self.assertEqual(req1.result().status, "ok")
        # ...but it must not cancel the new-generation request.
        self.assertFalse(req1.cancel())
        engine.drain()
        self.assertEqual(req2.result().status, "ok")
        self.assertEqual(req2.result().bytes_copied, 8)
        self.assertEqual(engine.output, SOURCE[0:8] + SOURCE[8:16])
        # Each generation completed exactly once per slot.
        self.assertEqual(ring.completion_count(0), 2)
        self.assertEqual(ring.completion_count(1), 2)
        self.assertEqual(len(engine.results), 2)


class TestDescriptorErrors(unittest.TestCase):
    def test_error_ends_request_and_releases_remaining_slots(self):
        ring = DmaRing(4, 16)
        ring.pool[:] = SOURCE[:16]
        engine = SteppedCopyEngine(ring)
        # Chain 0 -> 1 -> 2; descriptor 1's range [8, 24) exceeds the pool.
        req = submit_chain(engine, ring, [(0, 0, 8), (1, 8, 16), (2, 0, 4)])
        engine.step()  # descriptor 0 ok
        engine.step()  # descriptor 1 fails device-side validation
        result = req.result()
        self.assertEqual(result.status, "error")
        self.assertEqual(result.error_index, 1)
        self.assertIn("buffer range", result.error)
        self.assertEqual(result.bytes_copied, 8)
        self.assertEqual(result.descriptors, (0, 1))
        self.assertEqual(result.skipped, (2,))
        self.assertEqual(engine.output, SOURCE[0:8])
        self.assertEqual(ring.cpu_acquire(0).status, "ok")
        self.assertEqual(ring.cpu_acquire(1).status, "error")
        self.assertEqual(ring.cpu_acquire(2).status, "skipped")

    def test_error_does_not_affect_disjoint_request(self):
        ring = DmaRing(4, 16)
        ring.pool[:] = SOURCE[:16]
        engine = SteppedCopyEngine(ring)
        req_a = submit_chain(engine, ring, [(0, 0, 8), (1, 8, 16)])
        req_b = submit_chain(engine, ring, [(2, 0, 4), (3, 4, 4)])
        engine.drain()
        self.assertEqual(req_a.result().status, "error")
        self.assertEqual(req_b.result().status, "ok")
        self.assertEqual(req_b.result().bytes_copied, 8)
        self.assertEqual(engine.output, SOURCE[0:8] + SOURCE[0:8])
        self.assertEqual(len(engine.results), 2)

    def test_old_engine_halt_semantics_unchanged(self):
        # The new entry point must not alter CopyEngine: a bad chain still
        # halts the old engine and leaves later queued chains unconsumed.
        ring = DmaRing(4, 16)
        ring.pool[:] = SOURCE[:16]
        write_and_release(ring, 0, offset=8, length=16, next=None)  # bad
        write_and_release(ring, 1, offset=0, length=4, next=None)
        engine = CopyEngine(ring)
        engine.submit_chain(0)
        engine.submit_chain(1)
        engine.run()
        self.assertTrue(engine.halted)
        (result,) = engine.results
        self.assertEqual(result.status, "error")
        self.assertEqual(engine.queued, 1)
        self.assertEqual(ring.completion_count(1), 0)


class TestEventSequenceEnumeration(unittest.TestCase):
    """Exhaustively enumerate short step/cancel sequences on a tiny ring
    and check the protocol invariants after every single event."""

    @staticmethod
    def reference_model(events):
        """Independent hand-written model: returns (status, descriptors
        copied) for the 2-descriptor chain under the given event sequence."""
        copied = 0
        cancel_pending = False
        status = None
        for event in events:
            if status is not None:
                continue  # finished: later events are no-ops
            if event == "cancel":
                cancel_pending = True
            else:  # "step"
                if cancel_pending:
                    status = "cancelled"
                else:
                    copied += 1
                    if copied == 2:
                        status = "ok"
        return status, copied

    def test_all_short_step_cancel_sequences(self):
        for length in range(0, 7):
            for events in itertools.product(("step", "cancel"), repeat=length):
                with self.subTest(events=events):
                    self.check_sequence(events)

    def check_sequence(self, events):
        ring = DmaRing(2, 16)
        ring.pool[:] = SOURCE[:16]
        write_and_release(ring, 0, offset=0, length=4, next=1)
        write_and_release(ring, 1, offset=4, length=4, next=None)
        engine = SteppedCopyEngine(ring)
        req = engine.submit(0)
        expected_status, expected_copied = self.reference_model(events)
        for event in events:
            if event == "step":
                engine.step()
            else:
                req.cancel()
            # Invariant: each generation publishes at most one completion.
            for idx in (0, 1):
                info = ring.inspect(idx)
                self.assertLessEqual(info["completions"], info["generation"])
            # Invariant: at most one terminal result, published once.
            self.assertLessEqual(len(engine.results), 1)
            if req.done:
                self.assertIs(req.result(), engine.results[0])
            else:
                self.assertIsNone(req.result())
                # Unfinished request: slots not yet handed back stay
                # device-owned, so the CPU cannot rewrite them.
                for idx in (0, 1):
                    if ring.inspect(idx)["owner"] == "device":
                        with self.assertRaises(OwnershipError):
                            ring.cpu_write(idx, offset=0, length=1)
        # Drain to a deterministic terminal state and compare with the
        # independent reference model.
        engine.drain()
        self.assertTrue(req.done)
        result = req.result()
        if expected_status is None:
            # Events ran out before anything finished; the drain completes
            # or confirms-cancel according to the final registered intent.
            expected_status = "cancelled" if req.cancel_requested else "ok"
        self.assertEqual(result.status, expected_status)
        if expected_status == "ok":
            self.assertEqual(result.bytes_copied, 8)
            self.assertEqual(engine.output, SOURCE[0:8])
        else:
            self.assertEqual(result.bytes_copied, expected_copied * 4)
            self.assertEqual(engine.output, SOURCE[0:expected_copied * 4])
        # Terminal invariants: exactly one result; each slot completed
        # exactly once for this generation; all slots back with the CPU.
        self.assertEqual(len(engine.results), 1)
        for idx in (0, 1):
            self.assertEqual(ring.completion_count(idx), 1)
            self.assertEqual(ring.cpu_acquire(idx).generation, 1)


if __name__ == "__main__":
    unittest.main()
