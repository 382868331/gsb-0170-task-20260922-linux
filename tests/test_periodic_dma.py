"""Tests for periodic_dma.  Expected segment bytes are derived
independently from infinite sequence numbers, never by calling the
implementation under test."""

import math
import unittest

from periodic_dma import (
    ClaimResult,
    LostRange,
    PartialState,
    PeriodicAcquisition,
    StateError,
)


def expected_segment(seq, size):
    """Independent reference: deterministic incrementing bytes for seq."""
    base = seq * size
    return bytes((base + i) % 256 for i in range(size))


class TestBasicDelivery(unittest.TestCase):
    def test_empty_claim_returns_nothing(self):
        acq = PeriodicAcquisition(4, 8)
        result = acq.claim()
        self.assertIsInstance(result, ClaimResult)
        self.assertEqual(result.segments, ())
        self.assertEqual(result.lost, ())
        self.assertIsNone(result.partial)

    def test_single_step_same_step_ordering(self):
        acq = PeriodicAcquisition(4, 8)
        # Nothing completed yet: the frontier segment is not deliverable.
        self.assertEqual(acq.claim().segments, ())
        # step(): hardware completes first, then the claim sees it.
        result = acq.step()
        (seg,) = result.segments
        self.assertEqual((seg.seq, seg.generation, seg.slot), (0, 0, 0))
        self.assertEqual(seg.data, expected_segment(0, 8))
        self.assertEqual(acq.read_seq, 1)

    def test_claim_multiple_consecutive_segments(self):
        acq = PeriodicAcquisition(4, 8)
        for _ in range(3):
            acq.hw_step()
        result = acq.claim(max_segments=2)
        self.assertEqual([s.seq for s in result.segments], [0, 1])
        for s in result.segments:
            self.assertEqual(s.data, expected_segment(s.seq, 8))
        rest = acq.claim()
        self.assertEqual([s.seq for s in rest.segments], [2])

    def test_segments_wrap_around_slots_and_generations(self):
        acq = PeriodicAcquisition(2, 4)
        seen = []
        for _ in range(5):  # claim per step so nothing is overwritten
            result = acq.step()
            seen.extend((s.seq, s.generation, s.slot) for s in result.segments)
        self.assertEqual(seen, [(0, 0, 0), (1, 0, 1), (2, 1, 0),
                                (3, 1, 1), (4, 2, 0)])

    def test_byte_counter_wraps_at_256(self):
        acq = PeriodicAcquisition(2, 4)
        delivered = {}
        for _ in range(65):  # claim per step so nothing is overwritten
            for s in acq.step().segments:
                delivered[s.seq] = s.data
        self.assertEqual(delivered[0], bytes([0, 1, 2, 3]))
        # seq 64 starts at stream position 256 -> bytes 0,1,2,3 again
        self.assertEqual(delivered[64], bytes([0, 1, 2, 3]))
        self.assertEqual(delivered[64], expected_segment(64, 4))
        self.assertEqual(sorted(delivered), list(range(65)))


class TestOverrun(unittest.TestCase):
    def test_exactly_one_lap_is_still_lossless(self):
        n, size = 4, 8
        acq = PeriodicAcquisition(n, size)
        for _ in range(n):
            acq.hw_step()
        result = acq.claim()
        self.assertEqual([s.seq for s in result.segments], [0, 1, 2, 3])
        self.assertEqual(result.lost, ())
        self.assertEqual(acq.lost_total, 0)
        # One more completed segment puts seq 0 exactly one lap behind...
        # One more completed segment overwrites the oldest of the lap...
        acq2 = PeriodicAcquisition(n, size)
        for _ in range(n + 1):
            acq2.hw_step()
        result = acq2.claim()  # seq 0 overwritten by seq 4 in slot 0
        (rng,) = result.lost
        self.assertEqual((rng.first_seq, rng.last_seq, rng.count),
                         (0, 0, 1))
        self.assertEqual(acq2.lost_total, 1)
        self.assertEqual([s.seq for s in result.segments], [1, 2, 3, 4])

    def test_slow_consumer_exact_overwritten_interval(self):
        n, size = 4, 8
        acq = PeriodicAcquisition(n, size)
        for _ in range(3):
            acq.hw_step()  # seqs 0,1,2
        acq.claim()  # software has consumed through seq 2
        # Hardware runs 6 segments ahead: seqs 3..8; slots 3,0,1,2,3,0.
        for _ in range(6):
            acq.hw_step()
        result = acq.claim()
        # Seqs 3 (slot3 now holds seq7) and 4 (slot0 now holds seq8) lost.
        (rng,) = result.lost
        self.assertEqual((rng.first_seq, rng.last_seq, rng.count),
                         (3, 4, 2))
        self.assertEqual([s.seq for s in result.segments], [5, 6, 7, 8])
        for s in result.segments:
            self.assertEqual(s.data, expected_segment(s.seq, size))
            self.assertEqual(s.generation, s.seq // n)
        self.assertEqual(acq.lost_total, 2)
        self.assertEqual(acq.read_seq, 9)

    def test_never_returns_stale_data_and_covers_whole_stream(self):
        n, size = 3, 5
        acq = PeriodicAcquisition(n, size)
        for _ in range(11):  # seqs 0..10
            acq.hw_step()
        result = acq.claim()
        delivered = [s.seq for s in result.segments]
        lost_seqs = []
        for rng in result.lost:
            lost_seqs.extend(range(rng.first_seq, rng.last_seq + 1))
        # Lost + delivered partitions every seq up to the frontier with no
        # overlap and no gaps.
        self.assertEqual(sorted(lost_seqs + delivered), list(range(11)))
        self.assertEqual(len(lost_seqs) + len(delivered), 11)
        # Anything delivered must still match its independent expectation;
        # stale overwritten bytes can never satisfy this.
        for s in result.segments:
            self.assertEqual(s.data, expected_segment(s.seq, size))
        self.assertEqual(acq.lost_total, len(lost_seqs))

    def test_loss_detected_more_than_two_laps_behind(self):
        n, size = 2, 4
        acq = PeriodicAcquisition(n, size)
        for _ in range(7):  # seqs 0..6
            acq.hw_step()
        result = acq.claim()
        (rng,) = result.lost
        # slots: 0->seq6, 1->seq5; seqs 0..4 overwritten (lag > one lap),
        # only seqs 5,6 still valid.
        self.assertEqual((rng.first_seq, rng.last_seq, rng.count),
                         (0, 4, 5))
        self.assertEqual([s.seq for s in result.segments], [5, 6])
        self.assertEqual(acq.lost_total, 5)


class TestPartialStopRestart(unittest.TestCase):
    def test_partial_segment_not_delivered_but_reported(self):
        acq = PeriodicAcquisition(4, 8)
        acq.hw_step()  # seq 0 complete
        acq.hw_write(3)  # seq 1 half-written
        result = acq.claim()
        self.assertEqual([s.seq for s in result.segments], [0])
        self.assertEqual(result.partial, PartialState(1, 1, 3, 8))
        # Claiming again does not hand out the partial segment.
        self.assertEqual(acq.claim().segments, ())
        self.assertEqual(acq.status()["partial"].bytes_written, 3)

    def test_stop_keeps_completed_and_reports_partial(self):
        acq = PeriodicAcquisition(2, 4)
        acq.hw_step()  # seq 0
        acq.hw_write(2)  # seq 1 partial
        report = acq.stop()
        self.assertEqual(report.completed_pending, 1)
        self.assertEqual(report.partial, PartialState(1, 1, 2, 4))
        self.assertEqual(report.lost_total, 0)
        # No hardware progress while stopped; completed segments remain.
        with self.assertRaises(StateError):
            acq.hw_step()
        result = acq.claim()
        self.assertEqual([s.seq for s in result.segments], [0])
        with self.assertRaises(StateError):
            acq.stop()

    def test_restart_rewrites_abandoned_partial_and_continues(self):
        size = 4
        acq = PeriodicAcquisition(2, size)
        acq.hw_step()  # seq 0
        acq.hw_write(2)  # seq 1 partial: bytes 0,1 written
        report = acq.stop()
        self.assertEqual(report.partial.seq, 1)
        acq.start()
        # seq 1 is re-written from byte 0 with exactly one full segment.
        acq.hw_write(size)
        result = acq.claim()
        self.assertEqual([s.seq for s in result.segments], [0, 1])
        for s in result.segments:
            self.assertEqual(s.data, expected_segment(s.seq, size))
        acq.hw_step()  # seq 2
        result = acq.claim()
        self.assertEqual([s.seq for s in result.segments], [2])
        self.assertEqual(result.segments[0].data, expected_segment(2, size))
        self.assertEqual(acq.lost_total, 0)
        with self.assertRaises(StateError):
            acq.start()

    def test_stop_on_boundary_has_no_partial(self):
        acq = PeriodicAcquisition(2, 4)
        for _ in range(3):
            acq.hw_step()
        report = acq.stop()
        self.assertIsNone(report.partial)
        self.assertEqual(report.completed_pending, 3)


class TestCounterWraparound(unittest.TestCase):
    def test_wrapping_counter_with_exact_infinite_expectations(self):
        n, size, modulus = 2, 4, 3
        acq = PeriodicAcquisition(n, size, counter_modulus=modulus)
        delivered = []
        for step in range(10):
            result = acq.step()
            self.assertEqual(acq.hw_counter, (step + 1) % modulus)
            for s in result.segments:
                self.assertEqual(s.data, expected_segment(s.seq, size))
                delivered.append(s.seq)
        self.assertEqual(delivered, list(range(10)))
        self.assertEqual(acq.lost_total, 0)
        # Infinite sequence keeps growing even though hw_counter wrapped.
        self.assertEqual(acq.hw_seq, 10)

    def test_wrapping_counter_with_slow_consumer(self):
        acq = PeriodicAcquisition(3, 4, counter_modulus=2)
        for _ in range(8):  # seqs 0..7, modulus does not affect overrun math
            acq.hw_step()
        result = acq.claim()
        (rng,) = result.lost
        # slots: 0->seq6, 1->seq7, 2->seq5; seqs 0..4 gone, 5..7 valid.
        self.assertEqual((rng.first_seq, rng.last_seq), (0, 4))
        self.assertEqual([s.seq for s in result.segments], [5, 6, 7])


class TestValidation(unittest.TestCase):
    def test_constructor_validation(self):
        for bad_count in (0, -1, True, 1.5, float("nan"), float("inf")):
            with self.assertRaises((ValueError, TypeError)):
                PeriodicAcquisition(bad_count, 4)
        for bad_size in (0, -2, False, 2.5):
            with self.assertRaises((ValueError, TypeError)):
                PeriodicAcquisition(4, bad_size)
        with self.assertRaises(ValueError):
            PeriodicAcquisition(4, 4, counter_modulus=1)
        with self.assertRaises(TypeError):
            PeriodicAcquisition(4, 4, counter_modulus=True)
        PeriodicAcquisition(4.0, 4.0, counter_modulus=4.0)  # integral ok

    def test_hw_write_and_claim_validation(self):
        acq = PeriodicAcquisition(4, 8)
        with self.assertRaises(ValueError):
            acq.hw_write(-1)
        with self.assertRaises(TypeError):
            acq.hw_write(True)
        with self.assertRaises(ValueError):
            acq.hw_write(math.inf)
        with self.assertRaises(ValueError):
            acq.claim(max_segments=0)
        with self.assertRaises(TypeError):
            acq.claim(max_segments=False)


if __name__ == "__main__":
    unittest.main()
