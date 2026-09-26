"""Tests for the BlitEngine 2D strided blit increment.

Expected pools are computed by an independent reference that enumerates
every (row, column) address explicitly: it snapshots each source byte by
its generated address, then writes destination addresses in ascending
logical row order and ascending column order (later writes overwrite
earlier ones).  The reference never calls the implementation under test.
"""

import unittest

from dma_ring import (
    BlitEngine,
    CopyEngine,
    DmaRing,
    OwnershipError,
    VisibilityError,
)

# Deterministic pool contents generated in-repo.
SOURCE = bytes((i * 7 + 3) % 256 for i in range(256))


def make_ring(capacity=8, pool_size=256):
    ring = DmaRing(capacity, pool_size)
    ring.pool[:] = SOURCE[:pool_size]
    return ring


def ref_blit(pool_before, *, src, dst, rows, bpr, src_stride, dst_stride):
    """Independent per-(row, col) address enumeration of the blit contract."""
    expected = bytearray(pool_before)
    # Snapshot phase: read every source byte by its generated address.
    snapshot = {}
    for r in range(rows):
        for c in range(bpr):
            snapshot[(r, c)] = pool_before[src + r * src_stride + c]
    # Write phase: ascending logical rows, ascending columns; later wins.
    for r in range(rows):
        for c in range(bpr):
            expected[dst + r * dst_stride + c] = snapshot[(r, c)]
    return bytes(expected)


def write_blit(ring, index, *, src, dst, rows, bpr, src_stride, dst_stride,
               next=None):
    ring.cpu_write_blit(index, src=src, dst=dst, rows=rows, bpr=bpr,
                        src_stride=src_stride, dst_stride=dst_stride,
                        next=next)
    ring.cpu_release(index)


def run_blit(ring, head):
    engine = BlitEngine(ring)
    engine.submit_chain(head)
    engine.run()
    return engine


class TestBlitInputValidation(unittest.TestCase):
    def test_negative_geometry_rejected_at_write(self):
        ring = make_ring()
        for field in ("src", "dst", "rows", "bpr"):
            kwargs = dict(src=0, dst=0, rows=1, bpr=1,
                          src_stride=0, dst_stride=0)
            kwargs[field] = -1
            with self.assertRaises(ValueError, msg=field):
                ring.cpu_write_blit(0, **kwargs)

    def test_bool_and_float_rejected(self):
        ring = make_ring()
        with self.assertRaises(TypeError):
            ring.cpu_write_blit(0, src=True, dst=0, rows=1, bpr=1,
                                src_stride=0, dst_stride=0)
        with self.assertRaises(ValueError):
            ring.cpu_write_blit(0, src=0, dst=0, rows=1, bpr=1,
                                src_stride=float("nan"), dst_stride=0)
        with self.assertRaises(TypeError):
            ring.cpu_write_blit(0, src=0, dst=0, rows=1, bpr=1.5,
                                src_stride=0, dst_stride=0)

    def test_negative_strides_accepted(self):
        ring = make_ring()
        ring.cpu_write_blit(0, src=10, dst=20, rows=2, bpr=4,
                            src_stride=-8, dst_stride=-8)  # no raise

    def test_failed_write_leaves_no_partial_state(self):
        ring = make_ring()
        ring.cpu_write_blit(0, src=1, dst=2, rows=3, bpr=4,
                            src_stride=5, dst_stride=6)
        before = ring.inspect(0)
        with self.assertRaises(ValueError):
            ring.cpu_write_blit(0, src=-1, dst=2, rows=3, bpr=4,
                                src_stride=5, dst_stride=6)
        self.assertEqual(ring.inspect(0), before)


class TestBlitTransfers(unittest.TestCase):
    def test_basic_block_copy(self):
        ring = make_ring()
        write_blit(ring, 0, src=0, dst=128, rows=4, bpr=8,
                   src_stride=8, dst_stride=8)
        engine = run_blit(ring, 0)
        self.assertEqual(bytes(ring.pool),
                         ref_blit(SOURCE, src=0, dst=128, rows=4, bpr=8,
                                  src_stride=8, dst_stride=8))
        (result,) = engine.results
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.bytes_copied, 32)
        comp = ring.cpu_acquire(0)
        self.assertEqual((comp.status, comp.bytes_done), ("ok", 32))

    def test_vertical_flip_negative_dst_stride(self):
        # 4x4 block at src=0 (stride 4) flipped vertically into dst=64.
        ring = make_ring()
        write_blit(ring, 0, src=0, dst=64 + 3 * 4, rows=4, bpr=4,
                   src_stride=4, dst_stride=-4)
        run_blit(ring, 0)
        self.assertEqual(bytes(ring.pool),
                         ref_blit(SOURCE, src=0, dst=76, rows=4, bpr=4,
                                  src_stride=4, dst_stride=-4))
        # Spot-check the flip semantics independently of the reference too.
        self.assertEqual(bytes(ring.pool[64:68]), SOURCE[12:16])
        self.assertEqual(bytes(ring.pool[76:80]), SOURCE[0:4])

    def test_negative_src_stride(self):
        ring = make_ring()
        write_blit(ring, 0, src=96 + 5 * 16, dst=0, rows=6, bpr=16,
                   src_stride=-16, dst_stride=16)
        run_blit(ring, 0)
        self.assertEqual(bytes(ring.pool),
                         ref_blit(SOURCE, src=176, dst=0, rows=6, bpr=16,
                                  src_stride=-16, dst_stride=16))

    def test_partial_overlap_src_dst(self):
        # dst overlaps the source rectangle; result must follow the
        # acceptance-time source snapshot.
        ring = make_ring()
        write_blit(ring, 0, src=0, dst=8, rows=4, bpr=16,
                   src_stride=16, dst_stride=16)
        run_blit(ring, 0)
        self.assertEqual(bytes(ring.pool),
                         ref_blit(SOURCE, src=0, dst=8, rows=4, bpr=16,
                                  src_stride=16, dst_stride=16))
        # Untouched padding keeps its original value.
        self.assertEqual(bytes(ring.pool[72:]), SOURCE[72:])

    def test_dst_rows_self_overlap_later_rows_win(self):
        # dst_stride < bpr: destination rows overlap each other.
        ring = make_ring()
        write_blit(ring, 0, src=0, dst=100, rows=4, bpr=8,
                   src_stride=8, dst_stride=2)
        run_blit(ring, 0)
        self.assertEqual(bytes(ring.pool),
                         ref_blit(SOURCE, src=0, dst=100, rows=4, bpr=8,
                                  src_stride=8, dst_stride=2))
        # Hand-computed: dst bytes 100..101 are row 0 cols 0..1; every later
        # byte is overwritten by a later row, so 102..113 hold rows 1..3
        # shifted by 2, with row 3 winning the tail.
        self.assertEqual(ring.pool[100], SOURCE[0])
        self.assertEqual(ring.pool[101], SOURCE[1])
        self.assertEqual(ring.pool[102], SOURCE[8])   # row1 col0 overwrites
        self.assertEqual(ring.pool[113], SOURCE[31])  # row3 col7 last write
        self.assertEqual(ring.pool[114], SOURCE[114])  # padding untouched

    def test_chain_wraps_ring(self):
        # Chain 7 -> 0 -> 1 wraps the ring; each descriptor blits 2 rows.
        ring = make_ring(capacity=8)
        write_blit(ring, 7, src=0, dst=128, rows=2, bpr=8,
                   src_stride=8, dst_stride=8, next=0)
        write_blit(ring, 0, src=32, dst=144, rows=2, bpr=8,
                   src_stride=8, dst_stride=8, next=1)
        write_blit(ring, 1, src=64, dst=160, rows=2, bpr=8,
                   src_stride=8, dst_stride=8, next=None)
        engine = run_blit(ring, 7)
        (result,) = engine.results
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.descriptors, (7, 0, 1))
        expected = ref_blit(SOURCE, src=0, dst=128, rows=2, bpr=8,
                            src_stride=8, dst_stride=8)
        expected = ref_blit(expected, src=32, dst=144, rows=2, bpr=8,
                            src_stride=8, dst_stride=8)
        expected = ref_blit(expected, src=64, dst=160, rows=2, bpr=8,
                            src_stride=8, dst_stride=8)
        self.assertEqual(bytes(ring.pool), expected)

    def test_empty_rectangle_is_successful_zero_transfer(self):
        for rows, bpr in ((0, 8), (4, 0), (0, 0)):
            ring = make_ring()
            write_blit(ring, 0, src=0, dst=64, rows=rows, bpr=bpr,
                       src_stride=8, dst_stride=8)
            engine = run_blit(ring, 0)
            (result,) = engine.results
            self.assertEqual(result.status, "ok")
            self.assertEqual(result.bytes_copied, 0)
            comp = ring.cpu_acquire(0)
            self.assertEqual((comp.status, comp.bytes_done), ("ok", 0))
            self.assertEqual(ring.completion_count(0), 1)
            self.assertEqual(bytes(ring.pool), SOURCE)  # nothing written

    def test_out_of_range_row_writes_nothing(self):
        # Row 2 of the destination falls outside the pool: the whole
        # descriptor must fail without writing any byte.
        ring = make_ring(pool_size=100)
        ring.pool[:] = SOURCE[:100]
        write_blit(ring, 0, src=0, dst=80, rows=3, bpr=16,
                   src_stride=16, dst_stride=16)
        engine = run_blit(ring, 0)
        (result,) = engine.results
        self.assertEqual(result.status, "error")
        self.assertEqual(result.error_index, 0)
        self.assertIn("row 1", result.error)
        self.assertTrue(engine.halted)
        self.assertEqual(bytes(ring.pool), SOURCE[:100])
        comp = ring.cpu_acquire(0)
        self.assertEqual((comp.status, comp.bytes_done), ("error", 0))

    def test_negative_stride_below_zero_rejected(self):
        ring = make_ring()
        write_blit(ring, 0, src=8, dst=0, rows=4, bpr=8,
                   src_stride=-4, dst_stride=8)  # row 2 src -> -1
        engine = run_blit(ring, 0)
        (result,) = engine.results
        self.assertEqual(result.status, "error")
        self.assertEqual(bytes(ring.pool), SOURCE)

    def test_bad_descriptor_halts_and_later_chain_stays_queued(self):
        ring = make_ring()
        write_blit(ring, 0, src=0, dst=250, rows=2, bpr=16,
                   src_stride=8, dst_stride=8)  # dst row 1 out of range
        write_blit(ring, 1, src=0, dst=64, rows=2, bpr=8,
                   src_stride=8, dst_stride=8)
        engine = BlitEngine(ring)
        engine.submit_chain(0)
        engine.submit_chain(1)
        engine.run()
        self.assertTrue(engine.halted)
        self.assertEqual(engine.queued, 1)
        self.assertEqual(bytes(ring.pool), SOURCE)
        engine.reset_halt()
        engine.run()
        self.assertEqual(engine.queued, 0)
        self.assertEqual(len(engine.results), 2)
        self.assertEqual(engine.results[1].status, "ok")
        self.assertEqual(bytes(ring.pool),
                         ref_blit(SOURCE, src=0, dst=64, rows=2, bpr=8,
                                  src_stride=8, dst_stride=8))

    def test_submit_requires_device_ownership(self):
        ring = make_ring()
        ring.cpu_write_blit(0, src=0, dst=64, rows=1, bpr=8,
                            src_stride=8, dst_stride=8)
        engine = BlitEngine(ring)
        with self.assertRaises(OwnershipError):
            engine.submit_chain(0)  # missing cpu_release

    def test_completion_visibility_protocol_reused(self):
        ring = make_ring()
        write_blit(ring, 0, src=0, dst=64, rows=1, bpr=8,
                   src_stride=8, dst_stride=8)
        with self.assertRaises(VisibilityError):
            ring.cpu_read_completion(0)  # device has not released yet
        engine = run_blit(ring, 0)
        comp = ring.cpu_acquire(0)
        self.assertEqual(comp.status, "ok")
        self.assertEqual(comp.bytes_done, 8)
        self.assertEqual(ring.completion_count(0), 1)

    def test_copy_engine_unaffected_by_blit_fields(self):
        # A copy descriptor and a blit descriptor share the same ring.
        ring = make_ring()
        ring.cpu_write(0, offset=0, length=16)
        ring.cpu_release(0)
        write_blit(ring, 1, src=0, dst=64, rows=2, bpr=8,
                   src_stride=8, dst_stride=8)
        copy = CopyEngine(ring)
        copy.submit_chain(0)
        copy.run()
        self.assertEqual(copy.output, SOURCE[:16])
        run_blit(ring, 1)
        self.assertEqual(bytes(ring.pool),
                         ref_blit(SOURCE, src=0, dst=64, rows=2, bpr=8,
                                  src_stride=8, dst_stride=8))

    def test_blit_engine_rejects_copy_descriptor(self):
        ring = make_ring()
        ring.cpu_write(0, offset=0, length=8)
        ring.cpu_release(0)
        engine = run_blit(ring, 0)
        (result,) = engine.results
        self.assertEqual(result.status, "error")
        self.assertIn("not a blit", result.error)
        self.assertEqual(bytes(ring.pool), SOURCE)


if __name__ == "__main__":
    unittest.main()
