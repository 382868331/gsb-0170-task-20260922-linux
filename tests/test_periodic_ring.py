"""Tests for the periodic capture mode of dma_ring.

Expected payloads are derived independently here: the oracle formula is
re-stated in this file (byte value (sequence * size + j) & 0xFF), never
pulled from the implementation under test.
"""

import unittest

from dma_ring import (
    ClaimResult,
    OverrunError,
    OverrunReport,
    OwnershipError,
    PeriodicRing,
)

SEG_SIZE = 8


def expected(sequence, size=SEG_SIZE):
    base = sequence * size
    return bytes((base + j) & 0xFF for j in range(size))


def started(capacity=4, size=SEG_SIZE, **kwargs):
    ring = PeriodicRing(capacity, size, **kwargs)
    ring.start()
    return ring


class TestHappyPath(unittest.TestCase):
    def test_claim_one_and_many_consecutive(self):
        ring = started()
        ring.hw_advance(3)
        result = ring.claim()
        self.assertEqual(len(result.segments), 3)
        self.assertEqual(result.lost_count, 0)
        for seg in result.segments:
            self.assertEqual(seg.data, expected(seg.sequence))
        self.assertEqual([s.sequence for s in result.segments], [0, 1, 2])
        self.assertEqual([s.generation for s in result.segments], [1, 1, 1])
        self.assertEqual(ring.claim_cursor, 3)

    def test_max_segments_limits_and_continues(self):
        ring = started(capacity=8)
        ring.hw_advance(5)
        first = ring.claim(2)
        self.assertEqual([s.sequence for s in first.segments], [0, 1])
        second = ring.claim(2)
        self.assertEqual([s.sequence for s in second.segments], [2, 3])
        third = ring.claim(10)  # only one remains
        self.assertEqual([s.sequence for s in third.segments], [4])
        self.assertEqual(ring.claim(10).segments, ())

    def test_segment_lap_slot_and_generation_fields(self):
        ring = started(capacity=2, size=4)
        seen = []
        for _ in range(5):
            ring.hw_step()
            seen.extend(ring.claim().segments)
        self.assertEqual([(s.slot, s.lap, s.generation) for s in seen],
                         [(0, 1, 1), (1, 1, 1), (0, 2, 2), (1, 2, 2), (0, 3, 3)])

    def test_capacity_one_periodic(self):
        ring = started(capacity=1, size=2)
        ring.hw_step()
        (seg,) = ring.claim().segments
        self.assertEqual(seg.data, b'\x00\x01')
        ring.hw_step()
        result = ring.claim()
        self.assertEqual(result.segments[0].sequence, 1)
        self.assertEqual(result.segments[0].data, b'\x02\x03')

    def test_data_copied_at_claim_not_live_view(self):
        ring = started(capacity=2, size=4)
        ring.hw_advance(2)
        result = ring.claim()
        # Later overwrites must not mutate the already-delivered bytes.
        ring.hw_advance(4)
        self.assertEqual(result.segments[0].data, expected(0, 4))


class TestInFlightSegment(unittest.TestCase):
    def test_segment_being_written_is_not_delivered(self):
        ring = started()
        ring.hw_write_bytes(expected(0)[:3])  # 3 of 8 bytes
        result = ring.claim()
        self.assertEqual(result.segments, ())
        ring.hw_write_bytes(expected(0)[3:])
        (seg,) = ring.claim().segments
        self.assertEqual(seg.sequence, 0)
        self.assertEqual(seg.data, expected(0))

    def test_consecutive_claim_stops_before_partial_segment(self):
        ring = started()
        ring.hw_advance(2)
        ring.hw_write_bytes(b'\x00\x01')  # partial seq 2
        result = ring.claim(5)
        self.assertEqual([s.sequence for s in result.segments], [0, 1])

    def test_cross_boundary_bulk_write_completes_and_starts(self):
        ring = started(capacity=4, size=4)
        ring.hw_write_bytes(expected(0, 4) + expected(1, 4) + b'\xff')
        result = ring.claim()
        self.assertEqual([s.sequence for s in result.segments], [0, 1])
        ring.hw_write_bytes(expected(2, 4)[1:])
        seqs = [s.sequence for s in ring.claim().segments]
        self.assertEqual(seqs, [2])


class TestOverwrite(unittest.TestCase):
    def test_slow_consumer_loses_exact_ranges(self):
        ring = started(capacity=4, size=4)
        ring.hw_advance(4)   # completes 0..3
        ring.hw_advance(4)   # seq 4..7 overwrite unclaimed 0..3
        result = ring.claim()
        self.assertEqual(result.lost, (OverrunReport(0, 3),))
        self.assertEqual(result.lost_count, 4)
        self.assertEqual([s.sequence for s in result.segments], [4, 5, 6, 7])
        for seg in result.segments:
            self.assertEqual(seg.data, expected(seg.sequence, 4))
        self.assertEqual(ring.claim_cursor, 8)
        self.assertEqual(ring.total_lost, 4)

    def test_exactly_one_lap_boundary(self):
        ring = started(capacity=4, size=4)
        ring.hw_advance(4)
        self.assertEqual(ring.claim().lost, ())   # one lap, all delivered
        ring.hw_advance(4)   # lap 2 completes; nothing lost (claimed in time)
        ring.hw_advance(4)   # lap 3 overwrites unclaimed lap 2 (4..7)
        result = ring.claim()
        self.assertEqual(result.lost, (OverrunReport(4, 7),))
        self.assertEqual([s.sequence for s in result.segments], [8, 9, 10, 11])

    def test_losses_merging_and_reporting_once(self):
        ring = started(capacity=4, size=4)
        ring.hw_advance(10)  # overwrites 0..5 (contiguous, merged)
        first = ring.claim(2)
        self.assertEqual(first.lost, (OverrunReport(0, 5),))
        self.assertEqual([s.sequence for s in first.segments], [6, 7])
        second = ring.claim()  # loss already consumed by the first claim
        self.assertEqual(second.lost, ())
        self.assertEqual([s.sequence for s in second.segments], [8, 9])

    def test_overwritten_payload_never_returned(self):
        ring = started(capacity=4, size=4)
        ring.hw_advance(8)  # lap 2 overwrites lap 1
        result = ring.claim()
        self.assertEqual([s.sequence for s in result.segments], [4, 5, 6, 7])
        for seg in result.segments:
            self.assertEqual(seg.data, expected(seg.sequence, 4))


class TestCounterWrap(unittest.TestCase):
    def test_counter_value_wraps_but_sequences_stay_infinite(self):
        ring = started(capacity=4, size=2, counter_bits=4)  # modulus 16
        got = []
        for _ in range(5):
            ring.hw_advance(4)
            got.extend(ring.claim().segments)
        self.assertEqual(ring.hw_counter(), 20 % 16)
        self.assertEqual([s.sequence for s in got], list(range(20)))
        for seg in got:
            self.assertEqual(seg.data, expected(seg.sequence, 2))

    def test_full_modulus_lag_is_ambiguous_and_raises(self):
        ring = started(capacity=4, size=2, counter_bits=3)  # modulus 8
        with self.assertRaises(OverrunError):
            ring.hw_advance(9)  # starting seq 8: lag reaches the modulus

    def test_claims_keep_wrap_simulation_unambiguous(self):
        ring = started(capacity=4, size=2, counter_bits=3)
        for _ in range(20):
            ring.hw_advance(4)
            result = ring.claim()
            self.assertEqual(result.lost, ())
        self.assertEqual(ring.claim_cursor, 80)
        self.assertEqual(ring.hw_counter(), 80 % 8)


class TestStopRestart(unittest.TestCase):
    def test_stop_reports_half_segment_and_keeps_completed(self):
        ring = started()
        ring.hw_advance(2)
        ring.hw_write_bytes(b'\x10\x20')  # 2/8 bytes of seq 2
        partial = ring.stop()
        self.assertEqual((partial.sequence, partial.bytes_written), (2, 2))
        self.assertFalse(ring.running)
        result = ring.claim()  # completed segments stay available
        self.assertEqual([s.sequence for s in result.segments], [0, 1])
        # Half-written segment is not deliverable while stopped.
        self.assertEqual(ring.claim().segments, ())

    def test_restart_rewrites_partial_and_continues_sequence(self):
        ring = started()
        ring.hw_advance(2)
        ring.hw_write_bytes(b'\x10\x20')
        self.assertIsNotNone(ring.stop())
        ring.start()
        self.assertEqual([s.sequence for s in ring.claim().segments], [0, 1])
        ring.hw_step()  # rewrites seq 2 from byte 0
        (seg,) = ring.claim().segments
        self.assertEqual(seg.sequence, 2)
        self.assertEqual(seg.data, expected(2))  # no stale partial bytes
        ring.hw_step()
        self.assertEqual(ring.claim().segments[0].sequence, 3)

    def test_stop_at_segment_boundary_returns_none(self):
        ring = started()
        ring.hw_advance(1)
        self.assertIsNone(ring.stop())

    def test_double_start_and_stop_are_errors(self):
        ring = PeriodicRing(4, 4)
        with self.assertRaises(OwnershipError):
            ring.stop()
        ring.start()
        with self.assertRaises(OwnershipError):
            ring.start()
        ring.stop()
        with self.assertRaises(OwnershipError):
            ring.hw_step()

    def test_completed_segments_survive_stop_restart_cycle(self):
        ring = started(capacity=4, size=4)
        ring.hw_advance(3)
        ring.stop()
        ring.start()
        ring.hw_advance(2)  # seq 3,4 (seq 4 overwrites unclaimed seq 0)
        result = ring.claim()
        self.assertEqual(result.lost, (OverrunReport(0, 0),))
        self.assertEqual([s.sequence for s in result.segments], [1, 2, 3, 4])


class TestOrdering(unittest.TestCase):
    def test_completion_then_claim_observes_new_segment(self):
        ring = started()
        ring.hw_advance(2)
        ring.claim()  # drain 0,1
        result = ring.hw_step_and_claim()  # completion happens first
        self.assertEqual([s.sequence for s in result.segments], [2])


class TestValidation(unittest.TestCase):
    def test_bad_construction_arguments(self):
        with self.assertRaises(ValueError):
            PeriodicRing(3, 4)       # capacity not power of two
        with self.assertRaises(ValueError):
            PeriodicRing(4, 0)       # empty segment
        with self.assertRaises(ValueError):
            PeriodicRing(4, 4, counter_bits=0)
        with self.assertRaises(TypeError):
            PeriodicRing(True, 4)

    def test_bad_claim_arguments(self):
        ring = started()
        with self.assertRaises(ValueError):
            ring.claim(-1)
        with self.assertRaises(TypeError):
            ring.claim(True)
        self.assertIsInstance(ring.claim(0), ClaimResult)

    def test_negative_segment_payload_rejected_by_formula(self):
        from dma_ring import periodic_segment_bytes
        with self.assertRaises(ValueError):
            periodic_segment_bytes(-1, 4)

    def test_hw_advance_rejects_unaligned_partial(self):
        ring = started()
        ring.hw_write_bytes(b'\x00')
        with self.assertRaises(OwnershipError):
            ring.hw_advance(1)


if __name__ == "__main__":
    unittest.main()
