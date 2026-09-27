"""Tests for chain control: cancel of queued chains, stop requests and
confirmations for executing chains, stale/duplicate notification safety.

Expected values are hand-computed slices of the deterministic SOURCE data,
never derived by calling the implementation under test."""

import unittest

from dma_ring import (
    ChainStateError,
    CopyEngine,
    DmaRing,
    OwnershipError,
)

SOURCE = bytes((i * 7 + 3) % 256 for i in range(256))


def make_ring(capacity=8, pool_size=256):
    ring = DmaRing(capacity, pool_size)
    ring.pool[:] = SOURCE[:pool_size]
    return ring


def write_and_release(ring, index, *, offset, length, next=None):
    ring.cpu_write(index, offset=offset, length=length, next=next)
    ring.cpu_release(index)


def make_three_link_ring():
    """Ring with chain 0 -> 1 -> 2 (4 bytes each) fully released."""
    ring = make_ring()
    write_and_release(ring, 0, offset=0, length=4, next=1)
    write_and_release(ring, 1, offset=4, length=4, next=2)
    write_and_release(ring, 2, offset=8, length=4, next=None)
    return ring


class TestCancelQueuedChain(unittest.TestCase):
    def test_cancel_reclaims_descriptors_as_skipped(self):
        ring = make_three_link_ring()
        engine = CopyEngine(ring)
        token = engine.submit_chain(0)
        self.assertIsInstance(token, int)
        result = engine.cancel_chain(0)
        self.assertEqual(result.status, "cancelled")
        self.assertEqual(result.descriptors, ())
        self.assertEqual(result.skipped, (0, 1, 2))
        self.assertEqual(result.bytes_copied, 0)
        self.assertIs(engine.results[-1], result)
        self.assertEqual(engine.queued, 0)
        self.assertEqual(engine.output, b"")
        # Every descriptor is back with the CPU as a skipped completion.
        for idx in (0, 1, 2):
            comp = ring.cpu_acquire(idx)
            self.assertEqual(comp.status, "skipped")
            self.assertEqual(comp.bytes_done, 0)
            self.assertEqual(ring.completion_count(idx), 1)

    def test_cancel_only_queued_chain_leaves_others(self):
        ring = make_three_link_ring()
        write_and_release(ring, 3, offset=16, length=4, next=None)
        engine = CopyEngine(ring)
        engine.submit_chain(0)
        engine.submit_chain(3)
        engine.cancel_chain(3)
        self.assertEqual(engine.queued, 1)
        engine.run()
        self.assertEqual(engine.output, SOURCE[0:12])
        self.assertEqual([r.status for r in engine.results],
                         ["cancelled", "ok"])

    def test_cancel_not_queued_raises(self):
        ring = make_three_link_ring()
        engine = CopyEngine(ring)
        with self.assertRaises(ChainStateError):
            engine.cancel_chain(0)  # never submitted
        engine.submit_chain(0)
        engine.step()  # chain started: no longer merely queued
        with self.assertRaises(ChainStateError):
            engine.cancel_chain(0)

    def test_cancelled_slots_are_reusable(self):
        ring = make_three_link_ring()
        engine = CopyEngine(ring)
        engine.submit_chain(0)
        engine.cancel_chain(0)
        for idx in (0, 1, 2):
            ring.cpu_acquire(idx)
        # Reuse one slot for a fresh one-link chain.
        write_and_release(ring, 1, offset=32, length=4, next=None)
        engine.submit_chain(1)
        engine.run()
        self.assertEqual(engine.output, SOURCE[32:36])
        self.assertEqual(ring.completion_count(1), 2)


class TestStopRequest(unittest.TestCase):
    def test_stop_waits_for_confirmation_before_reclaim(self):
        ring = make_three_link_ring()
        engine = CopyEngine(ring)
        engine.submit_chain(0)
        self.assertTrue(engine.step())  # descriptor 0 completes
        self.assertEqual(engine.active_head, 0)
        token = engine.request_stop(0)
        self.assertIsNone(engine.active_head)
        self.assertEqual(engine.stopping_heads, (0,))
        # Handed to the device: not reclaimable before confirmation.
        for idx in (1, 2):
            with self.assertRaises(OwnershipError):
                ring.cpu_acquire(idx)
        self.assertEqual(ring.completion_count(1), 0)
        self.assertEqual(ring.completion_count(2), 0)
        # Device confirms it no longer accesses the chain.
        self.assertTrue(engine.confirm_stop(0, token))
        self.assertEqual(engine.stopping_heads, ())
        (result,) = engine.results
        self.assertEqual(result.status, "stopped")
        self.assertEqual(result.descriptors, (0,))
        self.assertEqual(result.skipped, (1, 2))
        self.assertEqual(result.bytes_copied, 4)
        self.assertEqual(engine.output, SOURCE[0:4])
        for idx in (1, 2):
            self.assertEqual(ring.cpu_acquire(idx).status, "skipped")
        self.assertEqual(ring.cpu_acquire(0).status, "ok")

    def test_request_stop_requires_executing_chain(self):
        ring = make_three_link_ring()
        engine = CopyEngine(ring)
        with self.assertRaises(ChainStateError):
            engine.request_stop(0)  # nothing executing
        engine.submit_chain(0)
        with self.assertRaises(ChainStateError):
            engine.request_stop(0)  # queued but not started

    def test_stopping_chain_does_not_block_queue(self):
        ring = make_three_link_ring()
        write_and_release(ring, 4, offset=64, length=8, next=None)
        engine = CopyEngine(ring)
        engine.submit_chain(0)
        engine.submit_chain(4)
        engine.step()
        token = engine.request_stop(0)
        engine.run()  # later queued chain proceeds while stop is pending
        self.assertEqual(engine.output, SOURCE[0:4] + SOURCE[64:72])
        self.assertEqual(engine.stopping_heads, (0,))
        self.assertTrue(engine.confirm_stop(0, token))
        self.assertEqual([r.status for r in engine.results],
                         ["ok", "stopped"])

    def test_stop_before_any_descriptor_skips_whole_chain(self):
        ring = make_three_link_ring()
        engine = CopyEngine(ring)
        engine.submit_chain(0)
        # Force the chain to become active without completing anything is
        # not possible via step(); stop right after submission is a cancel.
        engine.cancel_chain(0)
        (result,) = engine.results
        self.assertEqual(result.skipped, (0, 1, 2))


class TestStaleAndDuplicateNotifications(unittest.TestCase):
    def test_duplicate_confirmation_releases_nothing_twice(self):
        ring = make_three_link_ring()
        engine = CopyEngine(ring)
        engine.submit_chain(0)
        engine.step()
        token = engine.request_stop(0)
        self.assertTrue(engine.confirm_stop(0, token))
        counts = {i: ring.completion_count(i) for i in (0, 1, 2)}
        self.assertFalse(engine.confirm_stop(0, token))  # duplicate
        self.assertFalse(engine.confirm_stop(0, token))
        self.assertEqual({i: ring.completion_count(i) for i in (0, 1, 2)},
                         counts)
        self.assertEqual(len(engine.results), 1)

    def test_stale_confirmation_does_not_touch_new_chain(self):
        ring = make_three_link_ring()
        engine = CopyEngine(ring)
        engine.submit_chain(0)
        engine.step()
        old_token = engine.request_stop(0)
        self.assertTrue(engine.confirm_stop(0, old_token))
        # CPU reclaims everything and reuses slot 0 as head of a new chain.
        for idx in (0, 1, 2):
            ring.cpu_acquire(idx)
        write_and_release(ring, 0, offset=48, length=4, next=None)
        new_token = engine.submit_chain(0)
        self.assertNotEqual(old_token, new_token)
        # A late duplicate notification from the old chain must be ignored.
        self.assertFalse(engine.confirm_stop(0, old_token))
        engine.run()
        self.assertEqual(engine.output, SOURCE[0:4] + SOURCE[48:52])
        last = engine.results[-1]
        self.assertEqual(last.status, "ok")
        self.assertEqual(last.descriptors, (0,))
        self.assertEqual(last.skipped, ())

    def test_wrong_token_does_not_confirm_pending_stop(self):
        ring = make_three_link_ring()
        engine = CopyEngine(ring)
        engine.submit_chain(0)
        engine.step()
        token = engine.request_stop(0)
        self.assertFalse(engine.confirm_stop(0, token + 1))
        self.assertFalse(engine.confirm_stop(1, token))  # wrong head
        self.assertEqual(engine.stopping_heads, (0,))
        self.assertEqual(ring.completion_count(1), 0)
        self.assertTrue(engine.confirm_stop(0, token))

    def test_interleaved_partial_stop_and_slot_reuse(self):
        # Partial completion, stop confirmation and slot reuse interleaved:
        # old notifications must not corrupt the chain reusing the slots.
        ring = make_ring()
        write_and_release(ring, 0, offset=0, length=4, next=1)
        write_and_release(ring, 1, offset=4, length=4, next=None)
        engine = CopyEngine(ring)
        engine.submit_chain(0)
        engine.step()                       # desc 0 done (partial)
        token_a = engine.request_stop(0)
        self.assertTrue(engine.confirm_stop(0, token_a))
        ring.cpu_acquire(0)
        ring.cpu_acquire(1)
        # Reuse both slots for a different chain 1 -> 0.
        write_and_release(ring, 1, offset=32, length=4, next=0)
        write_and_release(ring, 0, offset=36, length=4, next=None)
        token_b = engine.submit_chain(1)
        engine.run()
        self.assertEqual(engine.output,
                         SOURCE[0:4] + SOURCE[32:36] + SOURCE[36:40])
        # Late notifications aimed at either old chain change nothing.
        self.assertFalse(engine.confirm_stop(0, token_a))
        self.assertFalse(engine.confirm_stop(1, token_b))  # chain B done ok
        self.assertEqual(ring.completion_count(0), 2)
        self.assertEqual(ring.completion_count(1), 2)
        statuses = [r.status for r in engine.results]
        self.assertEqual(statuses, ["stopped", "ok"])


class TestStepping(unittest.TestCase):
    def test_step_processes_one_descriptor_at_a_time(self):
        ring = make_three_link_ring()
        engine = CopyEngine(ring)
        engine.submit_chain(0)
        self.assertTrue(engine.step())
        self.assertEqual(engine.output, SOURCE[0:4])
        self.assertTrue(engine.step())
        self.assertEqual(engine.output, SOURCE[0:8])
        self.assertTrue(engine.step())
        self.assertEqual(engine.output, SOURCE[0:12])
        self.assertFalse(engine.step())  # queue drained
        (result,) = engine.results
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.descriptors, (0, 1, 2))

    def test_run_equivalent_to_stepping(self):
        ring = make_three_link_ring()
        engine = CopyEngine(ring)
        engine.submit_chain(0)
        engine.run()
        self.assertEqual(engine.output, SOURCE[0:12])
        self.assertFalse(engine.halted)


if __name__ == "__main__":
    unittest.main()

