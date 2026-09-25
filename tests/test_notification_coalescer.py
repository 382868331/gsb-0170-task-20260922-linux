"""Tests for the NotificationCoalescer.

Expected values (record identities, orders, fire times) are derived from
the scenario timelines by hand, never by calling the coalescer to compute
its own expectations.
"""

import unittest

from dma_ring import (
    CopyEngine,
    DmaRing,
    NotificationCoalescer,
)

SOURCE = bytes((i * 7 + 3) % 256 for i in range(256))


def make_ring(capacity=4, pool_size=64):
    ring = DmaRing(capacity, pool_size)
    ring.pool[:] = SOURCE[:pool_size]
    return ring


def complete(ring, index, *, offset=0, length=4, status="ok", error=None,
             bytes_done=None, next=None):
    """Drive one full handoff; returns nothing (completion is published)."""
    ring.cpu_write(index, offset=offset, length=length, next=next)
    ring.cpu_release(index)
    ring.device_acquire(index)
    ring.device_release(
        index, status=status, error=error,
        bytes_done=length if bytes_done is None else bytes_done)


def identities(notification):
    return [(r.index, r.generation) for r in notification.records]


class TestCoalescerValidation(unittest.TestCase):
    def test_threshold_and_timeout_must_be_positive_ints(self):
        ring = make_ring()
        for bad in (0, -1, -100):
            with self.assertRaises(ValueError, msg=f"threshold={bad}"):
                NotificationCoalescer(ring, threshold=bad, timeout=5)
            with self.assertRaises(ValueError, msg=f"timeout={bad}"):
                NotificationCoalescer(ring, threshold=2, timeout=bad)
        for bad in (True, 1.5, "2"):
            with self.assertRaises(TypeError, msg=f"bad={bad!r}"):
                NotificationCoalescer(ring, threshold=bad, timeout=5)
            with self.assertRaises(TypeError, msg=f"bad={bad!r}"):
                NotificationCoalescer(ring, threshold=2, timeout=bad)
        # Integral floats accepted, consistent with the rest of the module.
        NotificationCoalescer(ring, threshold=2.0, timeout=5.0)

    def test_ring_type_checked(self):
        with self.assertRaises(TypeError):
            NotificationCoalescer(object(), threshold=2, timeout=5)

    def test_advance_rejects_negative_and_non_int(self):
        coalescer = NotificationCoalescer(make_ring(), threshold=2, timeout=5)
        with self.assertRaises(ValueError):
            coalescer.advance(-1)
        with self.assertRaises(TypeError):
            coalescer.advance(True)
        coalescer.advance(0)  # zero advance is a legal no-op


class TestThresholdAndTimeout(unittest.TestCase):
    def test_threshold_fires_immediately_in_completion_order(self):
        ring = make_ring()
        coalescer = NotificationCoalescer(ring, threshold=3, timeout=100)
        complete(ring, 0)
        complete(ring, 1)
        self.assertIsNone(coalescer.read())       # 2 < threshold: silent
        self.assertEqual(coalescer.pending_count, 2)
        complete(ring, 2)
        note = coalescer.read()
        self.assertIsNotNone(note)
        self.assertEqual(note.reason, "threshold")
        self.assertEqual(note.time, 0)            # all completed at t=0
        self.assertEqual(identities(note), [(0, 1), (1, 1), (2, 1)])
        self.assertEqual([r.status for r in note.records],
                         ["ok", "ok", "ok"])
        self.assertEqual(coalescer.pending_count, 0)
        self.assertIsNone(coalescer.read())       # consumed, not re-read

    def test_timeout_boundary_equality_triggers(self):
        ring = make_ring()
        coalescer = NotificationCoalescer(ring, threshold=10, timeout=5)
        complete(ring, 0)                          # t=0, deadline = 5
        self.assertEqual(coalescer.next_deadline, 5)
        self.assertEqual(coalescer.advance(4), [])  # t=4 < 5: nothing
        fired = coalescer.advance(1)                # t=5 == deadline
        self.assertEqual(len(fired), 1)
        (note,) = fired
        self.assertEqual(note.reason, "timeout")
        self.assertEqual(note.time, 5)
        self.assertEqual(identities(note), [(0, 1)])
        self.assertEqual(coalescer.now, 5)

    def test_threshold_filled_one_tick_before_timeout(self):
        ring = make_ring()
        coalescer = NotificationCoalescer(ring, threshold=2, timeout=3)
        complete(ring, 0)                          # t=0, deadline = 3
        self.assertEqual(coalescer.advance(2), [])  # t=2: one tick to go
        complete(ring, 1)                          # threshold met at t=2
        note = coalescer.read()
        self.assertEqual(note.reason, "threshold")
        self.assertEqual(note.time, 2)
        self.assertEqual(identities(note), [(0, 1), (1, 1)])
        # The deadline is gone with the batch: crossing t=3 stays silent.
        self.assertEqual(coalescer.advance(1), [])
        self.assertIsNone(coalescer.read())

    def test_simultaneous_threshold_and_timeout_emit_one_notification(self):
        # threshold==1 degenerate: every completion notifies immediately,
        # so a timeout can never also be pending for the same record.
        ring = make_ring()
        coalescer = NotificationCoalescer(ring, threshold=1, timeout=2)
        complete(ring, 0)
        (note,) = coalescer.drain()
        self.assertEqual(note.reason, "threshold")
        self.assertEqual(coalescer.advance(2), [])  # no extra timeout event
        # A threshold batch emitted at the deadline's own tick is likewise
        # a single event: batch of one at t=0, timeout=4, second record
        # completes the threshold=2 batch at t=4 exactly.
        ring2 = make_ring()
        c2 = NotificationCoalescer(ring2, threshold=2, timeout=4)
        complete(ring2, 0)                          # t=0, deadline = 4
        c2.advance(3)                               # t=3
        (timeout_note,) = c2.advance(1)             # t=4: timeout fires
        self.assertEqual(timeout_note.reason, "timeout")
        self.assertEqual(identities(timeout_note), [(0, 1)])
        complete(ring2, 1)                          # t=4: fresh batch
        complete(ring2, 2)                          # threshold at t=4
        (threshold_note,) = c2.drain()
        self.assertEqual(threshold_note.reason, "threshold")
        self.assertEqual(identities(threshold_note), [(1, 1), (2, 1)])
        # Across the whole scenario every completion appears exactly once.
        self.assertEqual(identities(timeout_note) + identities(threshold_note),
                         [(0, 1), (1, 1), (2, 1)])

    def test_timeout_restarts_from_first_unnotified_record(self):
        ring = make_ring()
        coalescer = NotificationCoalescer(ring, threshold=2, timeout=3)
        complete(ring, 0)                          # t=0
        complete(ring, 1)                          # threshold at t=0
        coalescer.drain()
        coalescer.advance(5)                        # t=5, batch empty
        complete(ring, 2)                          # t=5, new deadline = 8
        self.assertEqual(coalescer.next_deadline, 8)
        self.assertEqual(coalescer.advance(2), [])  # t=7
        (note,) = coalescer.advance(1)              # t=8 == deadline
        self.assertEqual(note.time, 8)
        self.assertEqual(identities(note), [(2, 1)])

    def test_multiple_batches_queue_in_order(self):
        ring = make_ring()
        coalescer = NotificationCoalescer(ring, threshold=2, timeout=100)
        for idx in (0, 1, 2, 3):
            complete(ring, idx)
        self.assertEqual(coalescer.events_pending, 2)
        notes = coalescer.drain()
        self.assertEqual([n.reason for n in notes], ["threshold"] * 2)
        self.assertEqual(identities(notes[0]), [(0, 1), (1, 1)])
        self.assertEqual(identities(notes[1]), [(2, 1), (3, 1)])
        self.assertEqual(coalescer.drain(), [])


class TestFlushAndConsumption(unittest.TestCase):
    def test_explicit_flush_and_empty_flush(self):
        ring = make_ring()
        coalescer = NotificationCoalescer(ring, threshold=10, timeout=100)
        self.assertIsNone(coalescer.flush())        # empty flush: no event
        self.assertEqual(coalescer.events_pending, 0)
        complete(ring, 0)
        coalescer.advance(2)
        complete(ring, 1)                           # t=2
        note = coalescer.flush()
        self.assertEqual(note.reason, "flush")
        self.assertEqual(note.time, 2)
        self.assertEqual(identities(note), [(0, 1), (1, 1)])
        self.assertIsNone(coalescer.flush())        # empty again
        self.assertIsNone(coalescer.read())

    def test_repeated_read_does_not_reconsume(self):
        ring = make_ring()
        coalescer = NotificationCoalescer(ring, threshold=2, timeout=100)
        complete(ring, 0)
        complete(ring, 1)
        first = coalescer.read()
        self.assertIsNotNone(first)
        self.assertIsNone(coalescer.read())         # not re-consumed
        self.assertIsNone(coalescer.read())
        self.assertEqual(coalescer.events_pending, 0)


class TestIdentityOrderAndErrors(unittest.TestCase):
    def test_wraparound_reuse_distinguished_by_generation(self):
        ring = make_ring(capacity=2)
        coalescer = NotificationCoalescer(ring, threshold=3, timeout=100)
        complete(ring, 0)                           # (0, gen 1)
        complete(ring, 1)                           # (1, gen 1)
        ring.cpu_acquire(0)                         # recycle slot 0
        complete(ring, 0)                           # (0, gen 2)
        (note,) = coalescer.drain()
        self.assertEqual(identities(note), [(0, 1), (1, 1), (0, 2)])
        # Same slot, different generations: two distinct records.
        gens = [r.generation for r in note.records if r.index == 0]
        self.assertEqual(gens, [1, 2])

    def test_error_chain_adjacent_to_ok_chain_via_engine(self):
        ring = make_ring(capacity=4, pool_size=16)
        coalescer = NotificationCoalescer(ring, threshold=10, timeout=3)
        engine = CopyEngine(ring)
        # Chain A (head 0): ok.  Chain B (head 1): buffer out of range.
        ring.cpu_write(0, offset=0, length=8, next=None)
        ring.cpu_release(0)
        ring.cpu_write(1, offset=8, length=16, next=None)  # 8+16 > 16
        ring.cpu_release(1)
        engine.submit_chain(0)
        engine.submit_chain(1)
        engine.run()
        self.assertTrue(engine.halted)
        self.assertEqual(engine.output, SOURCE[0:8])
        # No notification yet: threshold not met, clock has not advanced.
        self.assertIsNone(coalescer.read())
        (note,) = coalescer.advance(3)              # deadline of first record
        self.assertEqual(note.reason, "timeout")
        # Error completions count, ordered by actual completion: ok, error.
        self.assertEqual(identities(note), [(0, 1), (1, 1)])
        ok_rec, err_rec = note.records
        self.assertEqual(ok_rec.status, "ok")
        self.assertEqual(ok_rec.bytes_done, 8)
        self.assertEqual(err_rec.status, "error")
        self.assertIsNotNone(err_rec.error)
        self.assertEqual(err_rec.bytes_done, 0)

    def test_each_completion_identity_appears_exactly_once(self):
        ring = make_ring(capacity=2)
        coalescer = NotificationCoalescer(ring, threshold=3, timeout=5)
        # t=0: three completions -> threshold batch [(0,1), (1,1), (0,2)].
        complete(ring, 0)
        complete(ring, 1)
        ring.cpu_acquire(0)
        complete(ring, 0)
        # t=2: one ok and one error completion -> pending batch of two.
        coalescer.advance(2)
        ring.cpu_acquire(1)
        complete(ring, 1)                           # (1, gen 2)
        ring.cpu_acquire(0)
        complete(ring, 0, status="error", error="boom",
                 bytes_done=0)                      # (0, gen 3)
        # t=7: timeout covers exactly the second batch.
        (timeout_note,) = coalescer.advance(5)
        self.assertEqual(timeout_note.reason, "timeout")
        self.assertEqual(timeout_note.time, 7)
        notes = coalescer.drain()
        self.assertEqual(len(notes), 1)             # only the threshold one
        all_ids = identities(notes[0]) + identities(timeout_note)
        self.assertEqual(all_ids,
                         [(0, 1), (1, 1), (0, 2), (1, 2), (0, 3)])
        self.assertEqual(len(all_ids), len(set(all_ids)))  # no duplicates
        # Ring-side truth: every slot completed as often as recorded.
        self.assertEqual(ring.completion_count(0), 3)
        self.assertEqual(ring.completion_count(1), 2)


class TestVisibilityNotDelayed(unittest.TestCase):
    def test_completion_readable_before_any_notification(self):
        ring = make_ring()
        coalescer = NotificationCoalescer(ring, threshold=5, timeout=100)
        complete(ring, 0, offset=4, length=4)
        # Nothing notified yet, but the completion is already published:
        # the original acquire path works exactly as without a coalescer.
        self.assertEqual(coalescer.events_pending, 0)
        comp = ring.cpu_acquire(0)
        self.assertEqual((comp.status, comp.bytes_done), ("ok", 4))
        self.assertEqual(ring.cpu_read_completion(0).status, "ok")
        # Notification still pending for later consumption.
        note = coalescer.flush()
        self.assertEqual(identities(note), [(0, 1)])

    def test_ring_without_coalescer_behaves_as_before(self):
        ring = make_ring()
        self.assertEqual(ring._coalescers, [])
        complete(ring, 0)
        self.assertEqual(ring.cpu_acquire(0).status, "ok")

    def test_detach_stops_recording(self):
        ring = make_ring()
        coalescer = NotificationCoalescer(ring, threshold=1, timeout=1)
        coalescer.detach()
        coalescer.detach()                          # idempotent
        complete(ring, 0)
        self.assertEqual(coalescer.events_pending, 0)
        self.assertEqual(coalescer.pending_count, 0)


if __name__ == "__main__":
    unittest.main()
