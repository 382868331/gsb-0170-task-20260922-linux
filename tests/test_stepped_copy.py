"""Tests for SteppedCopyEngine: incremental, cancellable copies.

All scenarios are deterministic: events are applied in explicit call order
on a single-threaded model.  Expected bytes are hand-computed slices of the
deterministic SOURCE data, never derived from the implementation under test.
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


def make_chain(ring, layout):
    """Write+release a chain from [(slot, offset, length), ...] in order."""
    for i, (slot, offset, length) in enumerate(layout):
        nxt = layout[i + 1][0] if i + 1 < len(layout) else None
        write_and_release(ring, slot, offset=offset, length=length, next=nxt)


def drain(engine):
    """Step until no work remains; return the events."""
    events = []
    while True:
        event = engine.step()
        if event is None:
            return events
        events.append(event)


class TestNormalCompletion(unittest.TestCase):
    def test_step_by_step_completion(self):
        ring = make_ring()
        make_chain(ring, [(6, 64, 8), (7, 0, 8), (0, 128, 8)])
        engine = SteppedCopyEngine(ring)
        req = engine.submit(6)
        self.assertEqual(req.slots, (6, 7, 0))
        self.assertEqual(req.generations, (1, 1, 1))
        self.assertIsNone(engine.result(req))

        events = drain(engine)
        self.assertEqual([e.kind for e in events], ["copy", "copy", "copy"])
        self.assertEqual([e.index for e in events], [6, 7, 0])
        self.assertIsNone(engine.step())  # nothing left to do

        result = engine.result(req)
        self.assertEqual(result.status, "ok")
        self.assertIsNone(result.error)
        self.assertEqual(result.bytes_copied, 24)
        self.assertEqual(result.descriptors, (6, 7, 0))
        self.assertEqual(result.skipped, ())
        # Independent reference: hand-concatenated slices of SOURCE.
        self.assertEqual(engine.output, SOURCE[64:72] + SOURCE[0:8]
                         + SOURCE[128:136])
        for idx in (6, 7, 0):
            self.assertEqual(ring.completion_count(idx), 1)
            self.assertEqual(ring.cpu_acquire(idx).status, "ok")

    def test_result_reads_are_idempotent(self):
        ring = make_ring()
        make_chain(ring, [(0, 0, 4)])
        engine = SteppedCopyEngine(ring)
        req = engine.submit(0)
        drain(engine)
        first = engine.result(req)
        for _ in range(3):
            self.assertIs(engine.result(req), first)
        self.assertEqual(ring.completion_count(0), 1)

    def test_submit_validates_before_mutating(self):
        ring = make_ring()
        make_chain(ring, [(0, 0, 4), (1, 4, 4)])
        engine = SteppedCopyEngine(ring)
        req = engine.submit(0)
        # A second chain whose tail points at the busy slot 1 is rejected.
        write_and_release(ring, 3, offset=12, length=4, next=1)
        with self.assertRaises(SlotBusyError):
            engine.submit(3)
        # Failed submit left nothing behind: slot 3 is not occupied and the
        # original request is unaffected and completes normally.
        self.assertEqual(engine.in_flight, 1)
        drain(engine)
        self.assertEqual(engine.result(req).status, "ok")
        # Slot 3 was never occupied by the failed submit; once slot 1 is
        # released to the device again, the 3 -> 1 chain can run.
        ring.cpu_acquire(1)
        write_and_release(ring, 1, offset=4, length=4, next=None)
        req3 = engine.submit(3)
        drain(engine)
        self.assertEqual(engine.result(req3).status, "ok")


class TestCancellation(unittest.TestCase):
    def setUp(self):
        self.ring = make_ring()
        make_chain(self.ring, [(0, 0, 8), (1, 8, 8), (2, 16, 8)])
        self.engine = SteppedCopyEngine(self.ring)

    def test_cancel_midway(self):
        engine, ring = self.engine, self.ring
        req = engine.submit(0)
        event = engine.step()  # copies descriptor 0 only
        self.assertEqual((event.kind, event.index), ("copy", 0))
        self.assertEqual(engine.output, SOURCE[0:8])

        self.assertTrue(engine.cancel(req))
        self.assertFalse(engine.cancel(req))  # repeat: no-op
        # Cancel is only registered: slots 1 and 2 are still device-owned,
        # so the CPU cannot rewrite them before confirmation.
        for idx in (1, 2):
            with self.assertRaises(OwnershipError, msg=f"slot {idx}"):
                ring.cpu_write(idx, offset=0, length=1)
        self.assertIsNone(engine.result(req))  # not finished yet

        event = engine.step()  # confirms the cancellation
        self.assertEqual(event.kind, "cancelled")
        self.assertEqual(event.skipped, (1, 2))
        self.assertEqual(event.bytes_copied, 8)

        result = engine.result(req)
        self.assertEqual(result.status, "cancelled")
        self.assertEqual(result.bytes_copied, 8)
        self.assertEqual(result.descriptors, (0,))
        self.assertEqual(result.skipped, (1, 2))
        # Unprocessed part produced no output.
        self.assertEqual(engine.output, SOURCE[0:8])
        # Skipped slots were handed back through the normal release path.
        comp1 = ring.cpu_acquire(1)
        self.assertEqual(comp1.status, "skipped")
        self.assertEqual(comp1.bytes_done, 0)
        self.assertEqual(ring.cpu_acquire(2).status, "skipped")
        self.assertEqual(ring.cpu_acquire(0).status, "ok")
        for idx in (0, 1, 2):
            self.assertEqual(ring.completion_count(idx), 1)
        # Repeating cancel after termination is a no-op.
        self.assertFalse(engine.cancel(req))
        for idx in (0, 1, 2):
            self.assertEqual(ring.completion_count(idx), 1)

    def test_cancel_before_start(self):
        engine, ring = self.engine, self.ring
        req = engine.submit(0)
        self.assertTrue(engine.cancel(req))
        event = engine.step()
        self.assertEqual(event.kind, "cancelled")
        self.assertEqual(event.skipped, (0, 1, 2))
        result = engine.result(req)
        self.assertEqual(result.status, "cancelled")
        self.assertEqual(result.bytes_copied, 0)
        self.assertEqual(result.descriptors, ())
        self.assertEqual(engine.output, b"")
        for idx in (0, 1, 2):
            self.assertEqual(ring.cpu_acquire(idx).status, "skipped")
            self.assertEqual(ring.completion_count(idx), 1)

    def test_completion_then_cancel_success_wins(self):
        engine, ring = self.engine, self.ring
        req = engine.submit(0)
        events = drain(engine)  # run to completion first
        self.assertEqual([e.kind for e in events], ["copy"] * 3)
        # Cancel after completion cannot rewrite the result.
        self.assertFalse(engine.cancel(req))
        result = engine.result(req)
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.bytes_copied, 24)
        self.assertEqual(engine.output, SOURCE[0:24])
        for idx in (0, 1, 2):
            self.assertEqual(ring.completion_count(idx), 1)

    def test_cancel_then_step_does_not_copy_next(self):
        engine, ring = self.engine, self.ring
        req = engine.submit(0)
        engine.step()
        engine.step()  # two descriptors copied, one remains
        self.assertEqual(engine.output, SOURCE[0:16])
        self.assertTrue(engine.cancel(req))
        event = engine.step()  # confirms cancel instead of copying slot 2
        self.assertEqual(event.kind, "cancelled")
        self.assertEqual(event.skipped, (2,))
        result = engine.result(req)
        self.assertEqual(result.status, "cancelled")
        self.assertEqual(result.bytes_copied, 16)
        self.assertEqual(engine.output, SOURCE[0:16])
        self.assertEqual(ring.cpu_acquire(2).status, "skipped")


class TestWrapReuseAndStaleHandles(unittest.TestCase):
    def test_slot_reuse_after_cancel_and_stale_handle(self):
        ring = DmaRing(2, 16)
        ring.pool[:] = SOURCE[:16]
        engine = SteppedCopyEngine(ring)

        # Generation 1: chain 0 -> 1, cancelled before anything is copied.
        make_chain(ring, [(0, 0, 4), (1, 4, 4)])
        old_req = engine.submit(0)
        self.assertEqual(old_req.generations, (1, 1))
        self.assertTrue(engine.cancel(old_req))
        event = engine.step()
        self.assertEqual(event.kind, "cancelled")
        old_result = engine.result(old_req)
        self.assertEqual(old_result.status, "cancelled")
        self.assertEqual(old_result.bytes_copied, 0)

        # CPU reclaims the slots and recycles them for a new generation.
        for idx in (0, 1):
            self.assertEqual(ring.cpu_acquire(idx).status, "skipped")
        make_chain(ring, [(0, 8, 4), (1, 12, 4)])
        self.assertEqual(ring.inspect(0)["generation"], 2)

        new_req = engine.submit(0)
        self.assertEqual(new_req.generations, (2, 2))
        self.assertNotEqual(old_req.request_id, new_req.request_id)

        # The stale handle can still query its own terminal state...
        self.assertIs(engine.result(old_req), old_result)
        # ...but it must not cancel the new-generation request.
        self.assertFalse(engine.cancel(old_req))
        self.assertIsNone(engine.result(new_req))

        events = drain(engine)
        self.assertEqual([e.kind for e in events], ["copy", "copy"])
        new_result = engine.result(new_req)
        self.assertEqual(new_result.status, "ok")
        self.assertEqual(new_result.bytes_copied, 8)
        self.assertEqual(engine.output, SOURCE[8:16])
        # Each generation published exactly one completion per slot.
        self.assertEqual(ring.completion_count(0), 2)
        self.assertEqual(ring.completion_count(1), 2)
        # Old result is still the old one.
        self.assertEqual(engine.result(old_req).status, "cancelled")

    def test_old_engine_unaffected_by_new_entry_point(self):
        # CopyEngine halt semantics are unchanged: a bad chain halts the
        # engine and leaves later chains queued.
        ring = make_ring()
        write_and_release(ring, 0, offset=0, length=4, next=1)
        write_and_release(ring, 1, offset=8, length=999, next=None)  # bad
        write_and_release(ring, 2, offset=16, length=4, next=None)
        engine = CopyEngine(ring)
        engine.submit_chain(0)
        engine.submit_chain(2)
        engine.run()
        self.assertTrue(engine.halted)
        self.assertEqual(engine.queued, 1)
        self.assertEqual(ring.completion_count(2), 0)


class TestSlotConflictsAndErrors(unittest.TestCase):
    def test_in_flight_requests_may_not_share_slots(self):
        ring = make_ring()
        make_chain(ring, [(0, 0, 4), (1, 4, 4)])
        make_chain(ring, [(2, 8, 4), (3, 12, 4)])
        engine = SteppedCopyEngine(ring)
        engine.submit(0)
        engine.submit(2)  # disjoint: fine
        self.assertEqual(engine.in_flight, 2)
        # Chain 4 -> 1 overlaps the first request's slot 1.
        write_and_release(ring, 4, offset=16, length=4, next=1)
        with self.assertRaises(SlotBusyError):
            engine.submit(4)
        self.assertEqual(engine.in_flight, 2)

    def test_submit_requires_device_ownership(self):
        ring = make_ring()
        ring.cpu_write(0, offset=0, length=4)  # never released
        engine = SteppedCopyEngine(ring)
        with self.assertRaises(OwnershipError):
            engine.submit(0)

    def test_submit_rejects_cycle_and_bad_next(self):
        ring = make_ring()
        write_and_release(ring, 0, offset=0, length=4, next=0)  # self-loop
        engine = SteppedCopyEngine(ring)
        with self.assertRaises(ChainError) as ctx:
            engine.submit(0)
        self.assertEqual(ctx.exception.reason, "cycle")
        ring2 = make_ring()
        write_and_release(ring2, 0, offset=0, length=4, next=99)
        engine2 = SteppedCopyEngine(ring2)
        with self.assertRaises(ChainError) as ctx2:
            engine2.submit(0)
        self.assertEqual(ctx2.exception.reason, "next-out-of-range")

    def test_descriptor_error_ends_only_its_own_request(self):
        ring = make_ring()
        # Request A: slot 0 ok, slot 1 has a buffer range past the pool.
        make_chain(ring, [(0, 0, 8), (1, 250, 16)])
        # Request B: disjoint chain, all valid.
        make_chain(ring, [(2, 32, 8), (3, 40, 8)])
        engine = SteppedCopyEngine(ring)
        req_a = engine.submit(0)
        req_b = engine.submit(2)

        events = drain(engine)  # oldest request (A) is stepped first
        self.assertEqual([e.kind for e in events],
                         ["copy", "error", "copy", "copy"])

        result_a = engine.result(req_a)
        self.assertEqual(result_a.status, "error")
        self.assertEqual(result_a.error_index, 1)
        self.assertEqual(result_a.bytes_copied, 8)
        self.assertEqual(result_a.descriptors, (0, 1))
        self.assertEqual(result_a.skipped, ())  # slot 1 was the last one
        self.assertEqual(ring.cpu_acquire(1).status, "error")

        result_b = engine.result(req_b)
        self.assertEqual(result_b.status, "ok")
        self.assertEqual(result_b.bytes_copied, 16)
        # Output holds A's copied prefix followed by B's bytes.
        self.assertEqual(engine.output, SOURCE[0:8] + SOURCE[32:48])

    def test_descriptor_error_releases_remaining_slots(self):
        ring = make_ring()
        # Bad descriptor in the middle; slot 2 never gets processed.
        make_chain(ring, [(0, 0, 8), (1, 250, 16), (2, 48, 8)])
        engine = SteppedCopyEngine(ring)
        req = engine.submit(0)
        events = drain(engine)
        self.assertEqual([e.kind for e in events], ["copy", "error"])
        result = engine.result(req)
        self.assertEqual(result.status, "error")
        self.assertEqual(result.skipped, (2,))
        self.assertEqual(result.bytes_copied, 8)
        self.assertEqual(ring.cpu_acquire(2).status, "skipped")
        self.assertEqual(engine.output, SOURCE[0:8])
        # Slot 2 is CPU-owned again and can be reused immediately.
        write_and_release(ring, 2, offset=48, length=8)
        req2 = engine.submit(2)
        drain(engine)
        self.assertEqual(engine.result(req2).status, "ok")


class TestEventSequenceEnumeration(unittest.TestCase):
    """Exhaustively enumerate short event sequences on a tiny ring and check
    the protocol invariants after every single event."""

    RING_CAPACITY = 2
    LAYOUT = [(0, 0, 4), (1, 4, 4)]  # chain 0 -> 1, 8 bytes total

    def run_sequence(self, events):
        ring = DmaRing(self.RING_CAPACITY, 16)
        ring.pool[:] = SOURCE[:16]
        make_chain(ring, self.LAYOUT)
        engine = SteppedCopyEngine(ring)
        req = engine.submit(0)

        observed_result = None
        cancel_pending = False
        for event in events:
            if event == "step":
                engine.step()
            elif event == "cancel":
                if engine.cancel(req):
                    cancel_pending = True
            elif event == "read":
                engine.result(req)
            else:  # pragma: no cover - sequence generator bug guard
                raise AssertionError(f"unknown event {event!r}")

            # Invariant 1: each slot published at most one completion for
            # this generation, no matter what sequence brought us here.
            for idx in (0, 1):
                self.assertLessEqual(ring.completion_count(idx), 1)

            # Invariant 2: a terminal result, once published, never changes.
            current = engine.result(req)
            if observed_result is not None:
                self.assertIs(current, observed_result)
            observed_result = current

            # Invariant 3: output is exactly the copied prefix of the chain.
            done = engine.result(req)
            if done is None:
                # count ok-completed slots via completion statuses
                n_ok = 0
                for idx in (0, 1):
                    info = ring.inspect(idx)
                    if info["completion"] is not None \
                            and info["completion"][0] == "ok":
                        n_ok += 1
                expected_len = 4 * n_ok
            else:
                expected_len = done.bytes_copied
            self.assertEqual(len(engine.output), expected_len)
            self.assertEqual(engine.output, SOURCE[0:expected_len])

            # Invariant 4: while a cancel is registered but unconfirmed,
            # the CPU still cannot rewrite the unprocessed slots.
            if observed_result is None and cancel_pending:
                for idx in (0, 1):
                    info = ring.inspect(idx)
                    if info["completion"] is None:
                        with self.assertRaises(OwnershipError):
                            ring.cpu_write(idx, offset=0, length=1)
            if observed_result is not None:
                cancel_pending = False

        return engine, req, ring

    def test_all_short_sequences(self):
        alphabet = ("step", "cancel", "read")
        checked = 0
        for length in range(0, 7):
            for events in itertools.product(alphabet, repeat=length):
                self.run_sequence(events)
                checked += 1
        self.assertEqual(checked, sum(3 ** n for n in range(7)))

    def test_terminal_states_reached_by_enumeration(self):
        # Spot-check that the enumeration above actually reaches every
        # terminal status and that completion counts stay at one per slot.
        terminal = {}
        alphabet = ("step", "cancel")
        for length in range(1, 6):
            for events in itertools.product(alphabet, repeat=length):
                engine, req, ring = self.run_sequence(events)
                result = engine.result(req)
                if result is not None:
                    terminal.setdefault(result.status, result)
        self.assertEqual(set(terminal), {"ok", "cancelled"})
        self.assertEqual(terminal["ok"].bytes_copied, 8)
        self.assertEqual(terminal["cancelled"].skipped, (0, 1))


if __name__ == "__main__":
    unittest.main()
