"""Tests for the completion-notification coalescer.

Expected orderings/counts below are hand-derived from the simulated call
sequences; the core is never called to produce its own expected values.
Every completion identity is the ``(index, generation)`` pair.
"""

import unittest

from dma_ring import (
    CompletionCoalescer,
    CopyEngine,
    DmaRing,
    OwnershipError,
)

SOURCE = bytes((i * 7 + 3) % 256 for i in range(256))


def make_ring(capacity=8, pool_size=256):
    ring = DmaRing(capacity, pool_size)
    ring.pool[:] = SOURCE[:pool_size]
    return ring


def release_descriptor(ring, index, *, offset=0, length=4, status="ok",
                        error=None):
    """Drive one full device-side completion at the current clock."""
    ring.cpu_write(index, offset=offset, length=length)
    ring.cpu_release(index)
    ring.device_acquire(index)
    ring.device_release(index, status=status, bytes_done=length if status == "ok" else 0,
                        error=error)


class TestCoalescerConstruction(unittest.TestCase):
    def test_threshold_and_timeout_validated(self):
        ring = make_ring()
        for bad in (0, -1, True, 1.5):
            with self.assertRaises((ValueError, TypeError)):
                CompletionCoalescer(ring, count_threshold=bad, timeout=3)
        with self.assertRaises((ValueError, TypeError)):
            CompletionCoalescer(ring, count_threshold=2, timeout=bad)
        with self.assertRaises(TypeError):
            CompletionCoalescer("not-a-ring", count_threshold=2, timeout=3)

    def test_ring_rejects_second_coalescer(self):
        ring = make_ring()
        CompletionCoalescer(ring, count_threshold=2, timeout=3)
        with self.assertRaises(ValueError):
            CompletionCoalescer(ring, count_threshold=2, timeout=3)

    def test_advance_validates_delta(self):
        ring = make_ring()
        coal = CompletionCoalescer(ring, count_threshold=2, timeout=3)
        with self.assertRaises((ValueError, TypeError)):
            coal.advance(0)
        with self.assertRaises((ValueError, TypeError)):
            coal.advance(-1)
        with self.assertRaises(TypeError):
            coal.advance(True)
        self.assertEqual(coal.advance(2), 2)
        self.assertEqual(coal.now, 2)


class TestWithoutCoalescer(unittest.TestCase):
    """The hook must be inert when no coalescer is attached."""

    def test_plain_ring_has_no_coalescer_and_no_ready_queue(self):
        ring = make_ring()
        self.assertIsNone(ring._coalescer)
        release_descriptor(ring, 0)
        # Ownership/completion protocol completely unchanged.
        comp = ring.cpu_acquire(0)
        self.assertEqual(comp.status, "ok")


class TestCountThreshold(unittest.TestCase):
    def test_batch_fires_exactly_at_threshold(self):
        ring = make_ring()
        coal = CompletionCoalescer(ring, count_threshold=3, timeout=10)
        release_descriptor(ring, 0)
        release_descriptor(ring, 1)
        self.assertEqual(coal.pending, 2)
        self.assertEqual(coal.unread, 0)
        release_descriptor(ring, 2)
        # Threshold reached immediately at completion time t=0.
        self.assertEqual(coal.pending, 0)
        batches = coal.notifications()
        self.assertEqual(len(batches), 1)
        (batch,) = batches
        self.assertEqual(batch.reason, "count")
        self.assertEqual(batch.time, 0)
        self.assertEqual([(e.index, e.generation) for e in batch.completions],
                         [(0, 1), (1, 1), (2, 1)])

    def test_threshold_one_tick_before_timeout_does_not_wait(self):
        # First completion at t=0 arms a deadline of t=3.  The threshold is
        # completed at t=2: the count batch fires one tick early and the
        # later timeout must leave no echo.
        ring = make_ring()
        coal = CompletionCoalescer(ring, count_threshold=3, timeout=3)
        release_descriptor(ring, 0)              # t=0, deadline t=3
        coal.advance(2)                          # t=2
        release_descriptor(ring, 1)              # t=2
        release_descriptor(ring, 2)              # t=2 -> count == 3
        batches = coal.notifications()
        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0].reason, "count")
        self.assertEqual(batches[0].time, 2)
        self.assertEqual([e.index for e in batches[0].completions], [0, 1, 2])
        # Crossing the old deadline produces nothing for that window.
        coal.advance(5)                          # t=7
        self.assertEqual(coal.notifications(), [])

    def test_threshold_window_does_not_double_notify_at_deadline(self):
        # A count batch at t=0 must not be followed by a timeout echo for
        # the same window when time reaches the deadline.
        ring = make_ring()
        coal = CompletionCoalescer(ring, count_threshold=2, timeout=3)
        release_descriptor(ring, 0)
        release_descriptor(ring, 1)              # count batch at t=0
        self.assertEqual(len(coal.notifications()), 1)
        coal.advance(3)
        self.assertEqual(coal.notifications(), [])


class TestTimeoutBoundary(unittest.TestCase):
    def test_timeout_fires_on_boundary_equality(self):
        ring = make_ring()
        coal = CompletionCoalescer(ring, count_threshold=4, timeout=3)
        release_descriptor(ring, 0)              # t=0, deadline t=3
        coal.advance(2)                          # t=2: not yet
        self.assertEqual(coal.notifications(), [])
        coal.advance(1)                          # t=3: equality fires
        batches = coal.notifications()
        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0].reason, "timeout")
        self.assertEqual(batches[0].time, 3)
        self.assertEqual([(e.index, e.generation)
                          for e in batches[0].completions], [(0, 1)])

    def test_timeout_does_not_fire_one_tick_early(self):
        ring = make_ring()
        coal = CompletionCoalescer(ring, count_threshold=4, timeout=3)
        release_descriptor(ring, 0)
        coal.advance(2)
        self.assertEqual(coal.pending, 1)
        self.assertEqual(coal.unread, 0)

    def test_large_jump_fires_at_boundary_time_not_jump_end(self):
        ring = make_ring()
        coal = CompletionCoalescer(ring, count_threshold=4, timeout=3)
        release_descriptor(ring, 0)              # t=0
        coal.advance(10)                         # jump past t=3
        batches = coal.notifications()
        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0].reason, "timeout")
        self.assertEqual(batches[0].time, 3)     # fires at boundary, not t=10
        self.assertEqual(coal.now, 10)

    def test_timeout_window_restarts_after_flush_then_new_completion(self):
        ring = make_ring()
        coal = CompletionCoalescer(ring, count_threshold=4, timeout=3)
        release_descriptor(ring, 0)              # t=0
        coal.advance(3)                          # t=3 -> timeout {0}
        self.assertEqual(coal.notifications()[0].reason, "timeout")
        release_descriptor(ring, 1)              # t=3, new deadline t=6
        coal.advance(2)                          # t=5
        self.assertEqual(coal.notifications(), [])
        coal.advance(1)                          # t=6
        (batch,) = coal.notifications()
        self.assertEqual(batch.reason, "timeout")
        self.assertEqual(batch.time, 6)
        self.assertEqual([e.index for e in batch.completions], [1])

    def test_timeout_holds_all_completions_since_window_start(self):
        ring = make_ring()
        coal = CompletionCoalescer(ring, count_threshold=10, timeout=5)
        release_descriptor(ring, 0)              # t=0
        coal.advance(2)                          # t=2
        release_descriptor(ring, 1)              # t=2
        coal.advance(3)                          # t=5 == deadline
        (batch,) = coal.notifications()
        self.assertEqual(batch.reason, "timeout")
        self.assertEqual([e.index for e in batch.completions], [0, 1])


class TestFlush(unittest.TestCase):
    def test_empty_flush_produces_nothing(self):
        ring = make_ring()
        coal = CompletionCoalescer(ring, count_threshold=3, timeout=3)
        self.assertIsNone(coal.flush())
        self.assertEqual(coal.notifications(), [])
        coal.advance(3)
        self.assertIsNone(coal.flush())
        self.assertEqual(coal.unread, 0)

    def test_flush_sends_pending_and_restarts_timer(self):
        ring = make_ring()
        coal = CompletionCoalescer(ring, count_threshold=5, timeout=10)
        release_descriptor(ring, 0)
        coal.advance(2)
        release_descriptor(ring, 1)
        batch = coal.flush()
        self.assertIsNotNone(batch)
        self.assertEqual(batch.reason, "flush")
        self.assertEqual(batch.time, 2)
        self.assertEqual([e.index for e in batch.completions], [0, 1])
        # Delivered directly: it must not also sit in the unread queue.
        self.assertEqual(coal.unread, 0)
        self.assertEqual(coal.notifications(), [])
        # Flushed window does not time out later.
        coal.advance(20)
        self.assertEqual(coal.notifications(), [])
        # A later completion arms a fresh deadline.
        release_descriptor(ring, 2)              # t=22
        coal.advance(10)                         # t=32 == 22+10
        (batch2,) = coal.notifications()
        self.assertEqual(batch2.reason, "timeout")
        self.assertEqual([e.index for e in batch2.completions], [2])


class TestConsumeOnRead(unittest.TestCase):
    def test_notifications_drain_and_repeat_read_is_empty(self):
        ring = make_ring()
        coal = CompletionCoalescer(ring, count_threshold=2, timeout=3)
        release_descriptor(ring, 0)
        release_descriptor(ring, 1)
        first = coal.notifications()
        self.assertEqual(len(first), 1)
        # Repeated reads do not redeliver (re-consume) the same batch.
        self.assertEqual(coal.notifications(), [])
        self.assertIsNone(coal.pop_notification())

    def test_pop_notification_delivers_each_batch_once(self):
        ring = make_ring()
        coal = CompletionCoalescer(ring, count_threshold=1, timeout=3)
        release_descriptor(ring, 0)
        release_descriptor(ring, 1)
        coal.advance(3)
        release_descriptor(ring, 2)              # count batch at t=3
        popped = []
        while True:
            b = coal.pop_notification()
            if b is None:
                break
            popped.append(b)
        self.assertEqual(len(popped), 3)
        self.assertEqual([b.reason for b in popped],
                         ["count", "count", "count"])
        self.assertIsNone(coal.pop_notification())

    def test_batches_delivered_oldest_first(self):
        ring = make_ring()
        coal = CompletionCoalescer(ring, count_threshold=2, timeout=3)
        release_descriptor(ring, 0)
        release_descriptor(ring, 1)              # count #1 at t=0
        coal.advance(3)
        release_descriptor(ring, 2)              # t=3
        coal.advance(3)                          # t=6 -> timeout
        batches = coal.notifications()
        self.assertEqual([b.time for b in batches], [0, 6])


class TestVisibilityNotDelayed(unittest.TestCase):
    def test_completion_visible_before_notification_consumed(self):
        # The merge affects notifications only: after device_release the CPU
        # can acquire/read even though the batch is still unread.
        ring = make_ring()
        coal = CompletionCoalescer(ring, count_threshold=10, timeout=10)
        release_descriptor(ring, 0, length=4)
        self.assertEqual(coal.pending, 1)
        comp = ring.cpu_acquire(0)               # no wait on notification
        self.assertEqual((comp.status, comp.bytes_done), ("ok", 4))
        self.assertEqual(ring.cpu_read_completion(0).status, "ok")


class TestWrapReuseAndGenerations(unittest.TestCase):
    def test_reused_slot_distinguished_by_generation(self):
        ring = DmaRing(2, 16)
        ring.pool[:] = SOURCE[:16]
        coal = CompletionCoalescer(ring, count_threshold=5, timeout=10)

        def cycle(index, gen):
            ring.cpu_write(index, offset=0, length=1)
            ring.cpu_release(index)
            ring.device_acquire(index)
            ring.device_release(index, status="ok", bytes_done=1)
            ring.cpu_acquire(index)

        cycle(0, 1)                              # t=0
        coal.advance(1)
        cycle(0, 2)                              # t=1, slot reused
        coal.advance(1)
        cycle(1, 1)                              # t=2
        coal.advance(1)
        cycle(0, 3)                              # t=3, slot reused again
        coal.advance(7)                          # t=10 == 0+10
        (batch,) = coal.notifications()
        self.assertEqual(batch.reason, "timeout")
        identities = [(e.index, e.generation)
                      for e in batch.completions]
        self.assertEqual(identities, [(0, 1), (0, 2), (1, 1), (0, 3)])
        # Every identity appears exactly once.
        self.assertEqual(len(identities), len(set(identities)))
        self.assertEqual(ring.completion_count(0), 3)
        self.assertEqual(ring.completion_count(1), 1)


class TestErrorChains(unittest.TestCase):
    def test_error_completion_counts_and_sits_next_to_ok_chain(self):
        ring = DmaRing(4, 16)
        ring.pool[:] = SOURCE[:16]
        coal = CompletionCoalescer(ring, count_threshold=10, timeout=10)
        # Chain 2 -> 3: descriptor 3 has an out-of-range buffer, so the
        # engine completes 2 with ok and 3 with error in one adjacent burst.
        ring.cpu_write(2, offset=0, length=8, next=3)
        ring.cpu_release(2)
        ring.cpu_write(3, offset=8, length=64, next=None)   # 8+64 > pool
        ring.cpu_release(3)
        engine = CopyEngine(ring)
        engine.submit_chain(2)
        engine.run()
        self.assertTrue(engine.halted)
        # Both completions are pending notification, error included, and in
        # actual completion order.
        self.assertEqual(coal.pending, 2)
        batch = coal.flush()
        self.assertEqual([(e.index, e.generation) for e in batch.completions],
                         [(2, 1), (3, 1)])
        # Statuses are owned by the release/acquire layer, not the notifier;
        # verify each identity completed with its real status exactly once.
        self.assertEqual(ring.cpu_acquire(2).status, "ok")
        bad = ring.cpu_acquire(3)
        self.assertEqual(bad.status, "error")
        self.assertTrue(bad.error)
        self.assertEqual(ring.completion_count(2), 1)
        self.assertEqual(ring.completion_count(3), 1)

    def test_mixed_error_and_ok_chains_each_identity_once(self):
        ring = DmaRing(4, 16)
        ring.pool[:] = SOURCE[:16]
        coal = CompletionCoalescer(ring, count_threshold=2, timeout=3)
        # Good chain: single descriptor 0.
        ring.cpu_write(0, offset=0, length=2)
        ring.cpu_release(0)
        # Bad chain 1 -> 2 (2 has bad range).
        ring.cpu_write(1, offset=0, length=2, next=2)
        ring.cpu_release(1)
        ring.cpu_write(2, offset=8, length=16, next=None)
        ring.cpu_release(2)
        engine = CopyEngine(ring)
        engine.submit_chain(0)
        engine.submit_chain(1)
        engine.run()                           # halts inside chain 1->2
        # Descriptor 0 (ok) then 1 (ok) already make a count pair at t=0.
        batches = coal.notifications()
        self.assertEqual(len(batches), 1)
        self.assertEqual([(e.index, e.generation)
                          for e in batches[0].completions], [(0, 1), (1, 1)])
        # The error descriptor is still pending after the halt.
        self.assertEqual(coal.pending, 1)
        coal.advance(3)                        # t=3 -> timeout
        (timeout_batch,) = coal.notifications()
        self.assertEqual([(e.index, e.generation)
                          for e in timeout_batch.completions], [(2, 1)])
        self.assertEqual(ring.cpu_acquire(2).status, "error")
        # Exactly-once across both batches.
        identities = ([(e.index, e.generation)
                       for e in batches[0].completions]
                      + [(e.index, e.generation)
                         for e in timeout_batch.completions])
        self.assertEqual(sorted(identities),
                         sorted(set(identities)))
        self.assertEqual(len(identities), 3)


class TestOwnershipUnchanged(unittest.TestCase):
    def test_missing_release_still_raises_with_coalescer(self):
        ring = make_ring()
        CompletionCoalescer(ring, count_threshold=2, timeout=3)
        ring.cpu_write(0, offset=0, length=4)
        with self.assertRaises(OwnershipError):
            ring.device_acquire(0)


class TestExactlyOnceAcrossWindows(unittest.TestCase):
    """Deterministic wrap/error/trigger interleaving; every identity once."""

    def test_count_timeout_flush_windows_partition_all_completions(self):
        ring = DmaRing(2, 16)
        ring.pool[:] = SOURCE[:16]
        coal = CompletionCoalescer(ring, count_threshold=2, timeout=3)

        def complete(index, *, status="ok", error=None):
            ring.cpu_write(index, offset=0, length=1)
            ring.cpu_release(index)
            ring.device_acquire(index)
            ring.device_release(
                index, status=status,
                bytes_done=1 if status == "ok" else 0, error=error)

        delivered = []                       # (batch, direct?)

        # t=0..1: count window over the first generation of both slots.
        complete(0)
        coal.advance(1)                      # t=1
        complete(1)                          # count batch at t=1
        delivered.extend((e, "count") for e in
                         coal.notifications()[0].completions)
        ring.cpu_acquire(0)
        ring.cpu_acquire(1)

        # t=2..5: slot 0 wrapped (gen 2), delivered by timeout.
        coal.advance(1)                      # t=2
        complete(0)
        coal.advance(3)                      # t=5 == 2+3
        delivered.extend((e, "timeout") for e in
                         coal.notifications()[0].completions)
        ring.cpu_acquire(0)

        # t=5..7: slot 1 wrapped (gen 2) as an error, delivered by flush.
        complete(1, status="error",
                 error="descriptor 1: simulated bad buffer")
        coal.advance(2)                      # t=7 (deadline t=8, still early)
        self.assertEqual(coal.notifications(), [])
        flushed = coal.flush()
        self.assertIsNotNone(flushed)
        delivered.extend((e, "flush") for e in flushed.completions)

        # Nothing ever escapes later.
        coal.advance(50)
        self.assertEqual(coal.notifications(), [])
        self.assertIsNone(coal.flush())

        identities = [(e.index, e.generation) for e, _ in delivered]
        self.assertEqual(identities,
                         [(0, 1), (1, 1), (0, 2), (1, 2)])
        self.assertEqual(len(identities), len(set(identities)))  # exactly once
        reasons = [r for _, r in delivered]
        self.assertEqual(reasons,
                         ["count", "count", "timeout", "flush"])
        # The release layer still owns status truth, each identity once.
        self.assertEqual(ring.completion_count(0), 2)
        self.assertEqual(ring.completion_count(1), 2)
        self.assertEqual(ring.cpu_acquire(1).status, "error")


if __name__ == "__main__":
    unittest.main()
