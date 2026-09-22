"""Ownership handoff, visibility and generation/ABA protocol tests."""

import unittest

from dmaring import (
    CoherentMemory,
    DmaRing,
    OwnershipError,
    StaleGenerationError,
    VisibilityError,
)


class TestOwnershipHandoff(unittest.TestCase):
    def setUp(self):
        self.ring = DmaRing(4, ring_id=0)

    def test_full_cycle(self):
        self.assertEqual(self.ring.owner_of(0), "cpu")
        self.ring.cpu_write(0, offset=8, length=4, next=None)
        gen = self.ring.cpu_release(0)
        self.assertEqual(gen, 1)
        self.assertEqual(self.ring.owner_of(0), "device")
        snap = self.ring.device_acquire(0)
        self.assertEqual((snap.offset, snap.length, snap.next), (8, 4, None))
        self.assertEqual(snap.generation, 1)
        comp = self.ring.device_release(0, status="ok", bytes_copied=4)
        self.assertEqual(self.ring.owner_of(0), "cpu")
        got = self.ring.cpu_acquire(0)
        self.assertEqual(got, comp)
        self.assertEqual(self.ring.completion(0).status, "ok")

    def test_cpu_cannot_write_device_owned_slot(self):
        self.ring.cpu_write(0, offset=0, length=1)
        self.ring.cpu_release(0)
        with self.assertRaises(OwnershipError):
            self.ring.cpu_write(0, offset=0, length=1)
        with self.assertRaises(OwnershipError):
            self.ring.cpu_release(0)

    def test_device_cannot_acquire_before_release(self):
        # "missing release": fields written but never released.
        self.ring.cpu_write(0, offset=0, length=1)
        with self.assertRaises(OwnershipError):
            self.ring.device_acquire(0)

    def test_device_cannot_release_cpu_owned_slot(self):
        with self.assertRaises(OwnershipError):
            self.ring.device_release(0, status="ok", bytes_copied=0)

    def test_cpu_cannot_acquire_device_owned_slot(self):
        self.ring.cpu_write(0, offset=0, length=1)
        self.ring.cpu_release(0)
        with self.assertRaises(OwnershipError):
            self.ring.cpu_acquire(0)


class TestVisibility(unittest.TestCase):
    def setUp(self):
        self.ring = DmaRing(2, ring_id=0)
        self.ring.cpu_write(0, offset=0, length=3)
        self.ring.cpu_release(0)
        self.ring.device_acquire(0)
        self.ring.device_release(0, status="ok", bytes_copied=3)

    def test_completion_read_requires_acquire(self):
        # Device has released; CPU has not acquired -> read is refused.
        with self.assertRaises(VisibilityError):
            self.ring.completion(0)
        self.ring.cpu_acquire(0)
        self.assertEqual(self.ring.completion(0).bytes_copied, 3)

    def test_completion_read_with_no_completion(self):
        with self.assertRaises(VisibilityError):
            self.ring.completion(1)  # never used slot

    def test_double_acquire_without_new_cycle(self):
        self.ring.cpu_acquire(0)
        with self.assertRaises(VisibilityError):
            self.ring.cpu_acquire(0)  # nothing pending any more

    def test_new_cycle_retires_old_completion(self):
        self.ring.cpu_acquire(0)
        self.ring.cpu_write(0, offset=4, length=2)
        self.ring.cpu_release(0)
        # Old acquired completion must not be readable in the new cycle.
        with self.assertRaises(VisibilityError):
            self.ring.completion(0)


class TestMemoryCoherence(unittest.TestCase):
    def test_device_sees_nothing_before_flush(self):
        mem = CoherentMemory(16)
        mem.cpu_write(0, b"ABCD")
        self.assertEqual(mem.device_read(0, 4), b"\x00" * 4)
        mem.flush()
        self.assertEqual(mem.device_read(0, 4), b"ABCD")

    def test_cpu_sees_nothing_before_invalidate(self):
        mem = CoherentMemory(16)
        mem.cpu_write(0, b"ABCD")
        mem.flush()
        mem.cpu_write(0, b"WXYZ")  # CPU view diverges
        self.assertEqual(mem.cpu_read(0, 4), b"WXYZ")
        mem.invalidate()
        self.assertEqual(mem.cpu_read(0, 4), b"ABCD")


class TestGenerationAndABA(unittest.TestCase):
    def test_stale_reacquire_rejected(self):
        ring = DmaRing(2, ring_id=0)
        ring.cpu_write(0, offset=0, length=1)
        ring.cpu_release(0)
        ring.device_acquire(0)
        # Same generation, still device-owned: a stale second acquire is ABA.
        with self.assertRaises(StaleGenerationError):
            ring.device_acquire(0)

    def test_wraparound_reuse_with_new_generation(self):
        # Capacity 2 ring; cycle slot 0 many times so slot indices wrap.
        ring = DmaRing(2, ring_id=0)
        for cycle in range(1, 6):
            ring.cpu_write(0, offset=cycle, length=1)
            gen = ring.cpu_release(0)
            self.assertEqual(gen, cycle)
            snap = ring.device_acquire(0)  # new generation: accepted
            self.assertEqual(snap.generation, cycle)
            self.assertEqual(snap.offset, cycle)
            ring.device_release(0, status="ok", bytes_copied=1)
            comp = ring.cpu_acquire(0)
            self.assertEqual(comp.generation, cycle)

    def test_aba_same_fields_different_generation(self):
        # After wrap-around the slot holds identical field values; only the
        # generation distinguishes the new descriptor from the stale one.
        ring = DmaRing(2, ring_id=0)
        ring.cpu_write(0, offset=5, length=2)
        ring.cpu_release(0)
        first = ring.device_acquire(0)
        ring.device_release(0, status="ok", bytes_copied=2)
        ring.cpu_acquire(0)
        ring.cpu_write(0, offset=5, length=2)  # identical fields
        ring.cpu_release(0)
        second = ring.device_acquire(0)
        self.assertEqual((first.offset, first.length),
                         (second.offset, second.length))
        self.assertNotEqual(first.generation, second.generation)
        with self.assertRaises(StaleGenerationError):
            ring.device_acquire(0)  # stale view of the same generation

    def test_generations_tracked_per_slot(self):
        ring = DmaRing(2, ring_id=0)
        ring.cpu_write(0, offset=0, length=1)
        ring.cpu_release(0)
        ring.device_acquire(0)
        # Slot 1 was never released; acquiring it must fail with ownership,
        # not be confused by slot 0's generation state.
        ring.cpu_write(1, offset=0, length=1)
        with self.assertRaises(OwnershipError):
            ring.device_acquire(1)


if __name__ == "__main__":
    unittest.main()
