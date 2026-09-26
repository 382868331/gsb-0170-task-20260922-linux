"""Tests for the BlitEngine 2D strided blit increment.

Expected pools are derived by an independent per-row/per-column address
enumeration over a snapshot copy of the pool (see expected_blit), never by
calling the implementation under test.
"""

import unittest

from dma_ring import (
    BlitEngine,
    ChainError,
    CopyEngine,
    DmaRing,
    OwnershipError,
    VisibilityError,
)

# Deterministic source data generated in-repo (no external inputs).
SOURCE = bytes((i * 5 + 1) % 256 for i in range(256))


def make_ring(capacity=8, pool_size=128):
    ring = DmaRing(capacity, pool_size)
    ring.pool[:] = SOURCE[:pool_size]
    return ring


def expected_blit(pool_before, *, src, dst, rows, row_bytes,
                  src_stride, dst_stride):
    """Independent reference: enumerate every (row, column) address and
    write from a snapshot taken before the transfer."""
    snapshot = bytes(pool_before)
    out = bytearray(pool_before)
    for r in range(rows):
        for c in range(row_bytes):
            out[dst + r * dst_stride + c] = snapshot[src + r * src_stride + c]
    return bytes(out)


def write_and_release_blit(ring, index, **fields):
    ring.cpu_write_blit(index, **fields)
    ring.cpu_release(index)


class TestBlitInputValidation(unittest.TestCase):
    def test_integer_fields_reject_bool_and_non_finite(self):
        ring = make_ring()
        base = dict(src=0, dst=32, rows=2, row_bytes=4,
                    src_stride=4, dst_stride=4)
        for key in ("src", "dst", "rows", "row_bytes",
                    "src_stride", "dst_stride"):
            with self.assertRaises(TypeError, msg=key):
                ring.cpu_write_blit(0, **{**base, key: True})
            with self.assertRaises(ValueError, msg=key):
                ring.cpu_write_blit(0, **{**base, key: float("nan")})
            with self.assertRaises(TypeError, msg=key):
                ring.cpu_write_blit(0, **{**base, key: 1.5})
        with self.assertRaises(TypeError):
            ring.cpu_write_blit(0, **base, next=False)

    def test_negative_rows_and_row_bytes_rejected(self):
        ring = make_ring()
        with self.assertRaises(ValueError):
            ring.cpu_write_blit(0, src=0, dst=0, rows=-1, row_bytes=4,
                                src_stride=4, dst_stride=4)
        with self.assertRaises(ValueError):
            ring.cpu_write_blit(0, src=0, dst=0, rows=1, row_bytes=-4,
                                src_stride=4, dst_stride=4)

    def test_negative_strides_and_starts_accepted_at_write_time(self):
        # Bounds are a submit-time (device-side) check, not a write-time
        # check; only type/sign of rows/row_bytes are validated here.
        ring = make_ring()
        ring.cpu_write_blit(0, src=60, dst=3, rows=4, row_bytes=4,
                            src_stride=-4, dst_stride=-1)
        ring.cpu_release(0)
        self.assertEqual(ring.inspect(0)["kind"], "blit")

    def test_failed_write_leaves_slot_unchanged(self):
        ring = make_ring()
        ring.cpu_write_blit(2, src=0, dst=32, rows=2, row_bytes=4,
                            src_stride=4, dst_stride=4)
        before = ring.inspect(2)
        with self.assertRaises(ValueError):
            ring.cpu_write_blit(2, src=0, dst=32, rows=-1, row_bytes=4,
                                src_stride=4, dst_stride=4)
        self.assertEqual(ring.inspect(2), before)


class TestBlitOwnershipReuse(unittest.TestCase):
    def test_cpu_write_blit_after_release_raises(self):
        ring = make_ring()
        write_and_release_blit(ring, 0, src=0, dst=32, rows=1, row_bytes=4,
                               src_stride=4, dst_stride=4)
        with self.assertRaises(OwnershipError):
            ring.cpu_write_blit(0, src=0, dst=32, rows=1, row_bytes=4,
                                src_stride=4, dst_stride=4)

    def test_device_read_blit_requires_acquire(self):
        ring = make_ring()
        write_and_release_blit(ring, 0, src=0, dst=32, rows=1, row_bytes=4,
                               src_stride=4, dst_stride=4)
        with self.assertRaises(VisibilityError):
            ring.device_read_blit(0)

    def test_cpu_acquire_before_device_release_raises(self):
        ring = make_ring()
        write_and_release_blit(ring, 0, src=0, dst=32, rows=1, row_bytes=4,
                               src_stride=4, dst_stride=4)
        with self.assertRaises(OwnershipError):
            ring.cpu_acquire(0)

    def test_full_handoff_and_completion_fields(self):
        ring = make_ring()
        write_and_release_blit(ring, 1, src=0, dst=64, rows=3, row_bytes=8,
                               src_stride=8, dst_stride=16)
        engine = BlitEngine(ring)
        engine.submit_chain(1)
        engine.run()
        comp = ring.cpu_acquire(1)
        self.assertEqual((comp.status, comp.bytes_done, comp.error),
                         ("ok", 24, None))
        self.assertEqual(ring.completion_count(1), 1)


class TestBlitTransfers(unittest.TestCase):
    def test_vertical_flip_negative_dst_stride(self):
        # 4x4 block at src rows [0..16) flipped vertically into dst,
        # landing with its last row at address 48.
        ring = make_ring()
        before = bytes(ring.pool)
        write_and_release_blit(ring, 0, src=0, dst=48, rows=4, row_bytes=4,
                               src_stride=4, dst_stride=-4)
        engine = BlitEngine(ring)
        engine.submit_chain(0)
        engine.run()
        expected = expected_blit(before, src=0, dst=48, rows=4, row_bytes=4,
                                 src_stride=4, dst_stride=-4)
        self.assertEqual(bytes(ring.pool), expected)
        # Hand-check the flip: dst row 0 (addr 48) is src row 0, dst row 3
        # (addr 36) is src row 3; everything else untouched.
        self.assertEqual(ring.pool[48:52], SOURCE[0:4])
        self.assertEqual(ring.pool[36:40], SOURCE[12:16])
        self.assertEqual(ring.pool[52:64], SOURCE[52:64])

    def test_partial_overlap_uses_submit_snapshot(self):
        # dst overlaps src; result must reflect the source bytes as of the
        # successful submit, not bytes mutated by earlier rows of the same
        # transfer or afterwards.
        ring = make_ring()
        before = bytes(ring.pool)
        write_and_release_blit(ring, 0, src=0, dst=6, rows=4, row_bytes=8,
                               src_stride=8, dst_stride=8)
        engine = BlitEngine(ring)
        engine.submit_chain(0)
        # Host mutates the source region after submit, before run: the
        # snapshot taken at submit must still determine the result.
        ring.pool[0:40] = b"\xAA" * 40
        engine.run()
        expected = expected_blit(before, src=0, dst=6, rows=4, row_bytes=8,
                                 src_stride=8, dst_stride=8)
        got = bytes(ring.pool)
        # Bytes outside the dst rectangle keep the post-submit mutation.
        self.assertEqual(got[:6], b"\xAA" * 6)
        self.assertEqual(got[6:38], expected[6:38])
        # Deterministic overlap: dst row 0 == original src row 0.
        self.assertEqual(got[6:14], before[0:8])

    def test_dst_rows_self_overlap_later_row_wins(self):
        # dst_stride < row_bytes: destination rows overlap each other.
        ring = make_ring()
        before = bytes(ring.pool)
        write_and_release_blit(ring, 0, src=0, dst=64, rows=4, row_bytes=8,
                               src_stride=8, dst_stride=2)
        engine = BlitEngine(ring)
        engine.submit_chain(0)
        engine.run()
        expected = expected_blit(before, src=0, dst=64, rows=4, row_bytes=8,
                                 src_stride=8, dst_stride=2)
        self.assertEqual(bytes(ring.pool), expected)
        # Row r covers [64+2r, 72+2r).  Byte 66 is covered by rows 0 and 1,
        # so row 1 (written later) wins; byte 70 is covered by rows 1..3,
        # so row 3 wins.
        self.assertEqual(ring.pool[66], SOURCE[8])    # row 1, col 0
        self.assertEqual(ring.pool[70], SOURCE[24])   # row 3, col 0
        # Padding beyond the last row keeps its original value.
        self.assertEqual(ring.pool[78:96], SOURCE[78:96])

    def test_chain_wraps_ring_and_shares_pool_with_copy_engine(self):
        ring = make_ring(capacity=4)
        before = bytes(ring.pool)
        # Blit chain wraps the ring: 3 -> 0.
        write_and_release_blit(ring, 3, src=0, dst=96, rows=2, row_bytes=8,
                               src_stride=8, dst_stride=8, next=0)
        write_and_release_blit(ring, 0, src=32, dst=112, rows=2,
                               row_bytes=8, src_stride=8, dst_stride=8,
                               next=None)
        engine = BlitEngine(ring)
        engine.submit_chain(3)
        engine.run()
        (result,) = engine.results
        self.assertEqual(result.descriptors, (3, 0))
        self.assertEqual(result.bytes_blitted, 32)
        expected = expected_blit(before, src=0, dst=96, rows=2, row_bytes=8,
                                 src_stride=8, dst_stride=8)
        expected = expected_blit(expected, src=32, dst=112, rows=2,
                                 row_bytes=8, dst_stride=8, src_stride=8)
        self.assertEqual(bytes(ring.pool), expected)
        # The same pool still serves a classic CopyEngine chain afterwards.
        ring.cpu_write(1, offset=0, length=16)
        ring.cpu_release(1)
        copier = CopyEngine(ring)
        copier.submit_chain(1)
        copier.run()
        self.assertEqual(copier.output, bytes(ring.pool[0:16]))

    def test_empty_rectangle_is_successful_zero_transfer(self):
        for rows, row_bytes in ((0, 8), (4, 0), (0, 0)):
            ring = make_ring()
            before = bytes(ring.pool)
            write_and_release_blit(ring, 0, src=10, dst=20, rows=rows,
                                   row_bytes=row_bytes, src_stride=8,
                                   dst_stride=8)
            engine = BlitEngine(ring)
            engine.submit_chain(0)
            engine.run()
            (result,) = engine.results
            self.assertEqual(result.status, "ok")
            self.assertEqual(result.bytes_blitted, 0)
            self.assertEqual(bytes(ring.pool), before)
            comp = ring.cpu_acquire(0)
            self.assertEqual((comp.status, comp.bytes_done), ("ok", 0))
            self.assertEqual(ring.completion_count(0), 1)

    def test_bytes_done_counts_payload_not_stride_padding(self):
        ring = make_ring()
        write_and_release_blit(ring, 0, src=0, dst=64, rows=3, row_bytes=4,
                               src_stride=16, dst_stride=16)
        engine = BlitEngine(ring)
        engine.submit_chain(0)
        engine.run()
        comp = ring.cpu_acquire(0)
        self.assertEqual(comp.bytes_done, 12)  # 3 rows * 4 bytes, no padding
        # Stride padding inside the dst span keeps its original value.
        self.assertEqual(ring.pool[68:80], SOURCE[68:80])


class TestBlitSubmitRejection(unittest.TestCase):
    def test_out_of_bounds_row_rejected_nothing_written(self):
        ring = make_ring()
        before = bytes(ring.pool)
        # rows=4 with dst_stride=-16 from dst=48: last row starts at 0
        # (ok), but rows=5 would reach -16.  Use rows=5 to force one row
        # out of bounds.
        write_and_release_blit(ring, 0, src=0, dst=48, rows=5, row_bytes=8,
                               src_stride=8, dst_stride=-16)
        engine = BlitEngine(ring)
        with self.assertRaises(ChainError) as ctx:
            engine.submit_chain(0)
        self.assertEqual(ctx.exception.reason, "buffer-out-of-range")
        self.assertEqual(ctx.exception.index, 0)
        self.assertEqual(engine.queued, 0)
        engine.run()
        self.assertEqual(engine.results, [])
        self.assertEqual(bytes(ring.pool), before)
        comp = ring.cpu_acquire(0)
        self.assertEqual(comp.status, "error")
        self.assertEqual(comp.bytes_done, 0)

    def test_source_out_of_bounds_rejected(self):
        ring = make_ring()
        before = bytes(ring.pool)
        write_and_release_blit(ring, 0, src=120, dst=0, rows=2, row_bytes=16,
                               src_stride=0, dst_stride=16)
        engine = BlitEngine(ring)
        with self.assertRaises(ChainError):
            engine.submit_chain(0)
        self.assertEqual(bytes(ring.pool), before)

    def test_mid_chain_bad_descriptor_completes_predecessors_skipped(self):
        ring = make_ring()
        before = bytes(ring.pool)
        write_and_release_blit(ring, 0, src=0, dst=64, rows=1, row_bytes=8,
                               src_stride=8, dst_stride=8, next=1)
        write_and_release_blit(ring, 1, src=0, dst=200, rows=1, row_bytes=8,
                               src_stride=8, dst_stride=8, next=None)
        engine = BlitEngine(ring)
        with self.assertRaises(ChainError) as ctx:
            engine.submit_chain(0)
        self.assertEqual(ctx.exception.index, 1)
        self.assertEqual(bytes(ring.pool), before)
        self.assertEqual(ring.cpu_acquire(0).status, "skipped")
        self.assertEqual(ring.cpu_acquire(1).status, "error")

    def test_unreleased_chain_member_raises_ownership(self):
        ring = make_ring()
        before = bytes(ring.pool)
        write_and_release_blit(ring, 0, src=0, dst=64, rows=1, row_bytes=8,
                               src_stride=8, dst_stride=8, next=1)
        ring.cpu_write_blit(1, src=0, dst=72, rows=1, row_bytes=8,
                            src_stride=8, dst_stride=8)  # never released
        engine = BlitEngine(ring)
        with self.assertRaises(OwnershipError):
            engine.submit_chain(0)
        self.assertEqual(bytes(ring.pool), before)
        self.assertEqual(ring.cpu_acquire(0).status, "skipped")

    def test_cycle_and_bad_next_rejected(self):
        ring = make_ring()
        write_and_release_blit(ring, 0, src=0, dst=64, rows=1, row_bytes=8,
                               src_stride=8, dst_stride=8, next=0)
        engine = BlitEngine(ring)
        with self.assertRaises(ChainError) as ctx:
            engine.submit_chain(0)
        self.assertEqual(ctx.exception.reason, "cycle")

        ring2 = make_ring()
        write_and_release_blit(ring2, 0, src=0, dst=64, rows=1, row_bytes=8,
                               src_stride=8, dst_stride=8, next=99)
        engine2 = BlitEngine(ring2)
        with self.assertRaises(ChainError) as ctx2:
            engine2.submit_chain(0)
        self.assertEqual(ctx2.exception.reason, "next-out-of-range")

    def test_copy_descriptor_rejected_as_wrong_kind(self):
        ring = make_ring()
        ring.cpu_write(0, offset=0, length=8)
        ring.cpu_release(0)
        engine = BlitEngine(ring)
        with self.assertRaises(ChainError) as ctx:
            engine.submit_chain(0)
        self.assertEqual(ctx.exception.reason, "wrong-kind")
        self.assertEqual(ring.cpu_acquire(0).status, "error")

    def test_submit_requires_device_owned_head(self):
        ring = make_ring()
        ring.cpu_write_blit(0, src=0, dst=64, rows=1, row_bytes=8,
                            src_stride=8, dst_stride=8)
        engine = BlitEngine(ring)
        with self.assertRaises(OwnershipError):
            engine.submit_chain(0)


if __name__ == "__main__":
    unittest.main()
