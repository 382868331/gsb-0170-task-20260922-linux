"""Input-validation contract tests: types, ranges, capacity, atomicity."""

import math
import unittest

from dmaring import (
    CapacityError,
    CoherentMemory,
    DmaRing,
    MAX_CAPACITY,
    MAX_MEMORY_SIZE,
    OwnershipError,
    SyntheticDevice,
    ValidationError,
)


class TestCapacity(unittest.TestCase):
    def test_valid_capacities(self):
        for cap in (1, 2, 4, 8, 16, 32, 64):
            ring = DmaRing(cap)
            self.assertEqual(ring.capacity, cap)

    def test_non_power_of_two_rejected(self):
        for cap in (3, 5, 6, 7, 12, 48, 63):
            with self.assertRaises(CapacityError, msg=f"capacity={cap}"):
                DmaRing(cap)

    def test_capacity_bounds(self):
        for cap in (0, -2, 65, 128):
            with self.assertRaises(ValidationError, msg=f"capacity={cap}"):
                DmaRing(cap)

    def test_capacity_bool_rejected(self):
        with self.assertRaises(ValidationError):
            DmaRing(True)

    def test_capacity_float_rejected(self):
        for cap in (4.0, float("nan"), float("inf")):
            with self.assertRaises(ValidationError, msg=f"capacity={cap}"):
                DmaRing(cap)


class TestIntegerContract(unittest.TestCase):
    """Integer parameters reject bool and floats (incl. NaN/Infinity)."""

    def setUp(self):
        self.ring = DmaRing(4, ring_id=0)

    def test_bool_rejected_for_int_params(self):
        with self.assertRaises(ValidationError):
            self.ring.cpu_write(True, offset=0, length=1)
        with self.assertRaises(ValidationError):
            self.ring.cpu_write(0, offset=False, length=1)
        with self.assertRaises(ValidationError):
            self.ring.cpu_write(0, offset=0, length=True)
        with self.assertRaises(ValidationError):
            CoherentMemory(True)

    def test_float_and_nan_inf_rejected(self):
        bad = [1.5, 2.0, math.nan, math.inf, -math.inf]
        for value in bad:
            with self.assertRaises(ValidationError, msg=f"offset={value}"):
                self.ring.cpu_write(0, offset=value, length=1)
            with self.assertRaises(ValidationError, msg=f"length={value}"):
                self.ring.cpu_write(0, offset=0, length=value)
            with self.assertRaises(ValidationError, msg=f"size={value}"):
                CoherentMemory(value)

    def test_string_and_none_rejected(self):
        for value in ("4", None, [1], object()):
            with self.assertRaises(ValidationError, msg=f"value={value!r}"):
                self.ring.cpu_write(0, offset=value, length=1)

    def test_negative_values_rejected(self):
        with self.assertRaises(ValidationError):
            self.ring.cpu_write(0, offset=-1, length=1)
        with self.assertRaises(ValidationError):
            self.ring.cpu_write(0, offset=0, length=-1)
        with self.assertRaises(ValidationError):
            DmaRing(4, ring_id=-1)

    def test_slot_out_of_range(self):
        with self.assertRaises(ValidationError):
            self.ring.cpu_write(4, offset=0, length=1)
        with self.assertRaises(ValidationError):
            self.ring.cpu_release(-1)

    def test_next_pointer_validation(self):
        self.ring.cpu_write(0, offset=0, length=1, next=None)
        self.ring.cpu_write(0, offset=0, length=1, next=(1, 2))
        for bad in (0, (1,), (1, 2, 3), ("a", 1), (1, 2.5), (True, 0),
                    (-1, 0), [0, 1]):
            with self.assertRaises(ValidationError, msg=f"next={bad!r}"):
                self.ring.cpu_write(0, offset=0, length=1, next=bad)

    def test_error_message_names_parameter(self):
        with self.assertRaises(ValidationError) as ctx:
            self.ring.cpu_write(0, offset=1.5, length=1)
        self.assertIn("offset", str(ctx.exception))


class TestMemoryValidation(unittest.TestCase):
    def test_size_bounds(self):
        with self.assertRaises(ValidationError):
            CoherentMemory(0)
        with self.assertRaises(ValidationError):
            CoherentMemory(MAX_MEMORY_SIZE + 1)
        mem = CoherentMemory(MAX_MEMORY_SIZE)
        self.assertEqual(mem.size, MAX_MEMORY_SIZE)

    def test_write_type_and_bounds(self):
        mem = CoherentMemory(16)
        mem.cpu_write(0, b"ab")
        mem.cpu_write(14, bytearray(b"cd"))
        mem.cpu_write(15, memoryview(b"e"))
        with self.assertRaises(ValidationError):
            mem.cpu_write(0, "text")
        with self.assertRaises(ValidationError):
            mem.cpu_write(15, b"xx")
        with self.assertRaises(ValidationError):
            mem.cpu_write(-1, b"x")
        with self.assertRaises(ValidationError):
            mem.cpu_read(0, 17)
        with self.assertRaises(ValidationError):
            mem.device_read(16, 1)


class TestBatchAtomicity(unittest.TestCase):
    def setUp(self):
        self.ring = DmaRing(4, ring_id=0)

    def test_valid_batch_applies(self):
        self.ring.cpu_write_many([
            {"slot": 0, "offset": 0, "length": 4},
            {"slot": 1, "offset": 8, "length": 2, "next": (0, 0)},
        ])
        self.assertEqual(self.ring.cpu_release(0), 1)
        self.assertEqual(self.ring.cpu_release(1), 1)

    def test_failed_batch_leaves_no_partial_change(self):
        self.ring.cpu_write(2, offset=40, length=4)
        before = self.ring.cpu_release(2)  # generation marker on a slot
        # Batch: first entry valid, second invalid -> nothing applied.
        with self.assertRaises(ValidationError):
            self.ring.cpu_write_many([
                {"slot": 0, "offset": 100, "length": 4},
                {"slot": 1, "offset": -1, "length": 4},
            ])
        # Slot 0 must not have been written: releasing it now must still be
        # generation 1 (i.e. it was untouched and still CPU-owned, but the
        # offset must not be the batch's 100).  We verify via device view.
        self.ring.cpu_release(0)
        snap = self.ring.device_acquire(0)
        self.assertEqual(snap.offset, 0)  # initial value, not 100
        self.assertEqual(before, 1)

    def test_batch_ownership_checked_atomically(self):
        self.ring.cpu_write(1, offset=0, length=1)
        self.ring.cpu_release(1)  # slot 1 now device-owned
        with self.assertRaises(OwnershipError):
            self.ring.cpu_write_many([
                {"slot": 0, "offset": 4, "length": 4},
                {"slot": 1, "offset": 4, "length": 4},
            ])
        # Slot 0 untouched by the failed batch.
        self.ring.cpu_release(0)
        snap = self.ring.device_acquire(0)
        self.assertEqual(snap.offset, 0)

    def test_batch_unknown_and_missing_keys(self):
        with self.assertRaises(ValidationError):
            self.ring.cpu_write_many([{"slot": 0, "offset": 0, "length": 1,
                                       "bogus": 1}])
        with self.assertRaises(ValidationError):
            self.ring.cpu_write_many([{"slot": 0, "offset": 0}])
        with self.assertRaises(ValidationError):
            self.ring.cpu_write_many([("not", "a", "dict")])


class TestDeviceValidation(unittest.TestCase):
    def test_memory_type_checked(self):
        with self.assertRaises(ValidationError):
            SyntheticDevice(b"not memory")

    def test_rings_argument_checked(self):
        dev = SyntheticDevice(CoherentMemory(64))
        with self.assertRaises(ValidationError):
            dev.process_chain(42, None)
        with self.assertRaises(ValidationError):
            dev.process_chain([object()], None)
        with self.assertRaises(ValidationError):
            dev.process_chain(DmaRing(2), (0, 0, 0))

    def test_device_release_status_checked(self):
        ring = DmaRing(2)
        ring.cpu_write(0, offset=0, length=1)
        ring.cpu_release(0)
        ring.device_acquire(0)
        with self.assertRaises(ValidationError):
            ring.device_release(0, status="maybe", bytes_copied=1)
        with self.assertRaises(ValidationError):
            ring.device_release(0, status="ok", bytes_copied=-1)
        with self.assertRaises(ValidationError):
            ring.device_release(0, status="ok", bytes_copied=True)


if __name__ == "__main__":
    unittest.main()
