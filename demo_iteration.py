"""Demo: BlitEngine 2D strided blits on the shared DMA ring pool.

Shows, with real offline computation (no sleeps, no recorded output):
  1. a vertical flip via a negative destination stride,
  2. a destination-row self-overlap (later rows overwrite earlier ones),
  3. an out-of-range descriptor rejected without writing a single byte,
  4. an empty rectangle completing as a successful zero transfer.
"""

from dma_ring import BlitEngine, DmaRing


def show(pool, lo, hi):
    return " ".join(f"{b:02x}" for b in pool[lo:hi])


def main() -> None:
    ring = DmaRing(capacity=8, pool_size=128)
    # Deterministic source block: 4 rows x 8 bytes at offset 0.
    ring.pool[:] = bytes((i * 7 + 3) % 256 for i in range(128))
    before = bytes(ring.pool)

    # --- 1. Vertical flip into offset 64 (negative dst stride) -----------
    ring.cpu_write_blit(0, src=0, dst=64 + 3 * 8, rows=4, bpr=8,
                        src_stride=8, dst_stride=-8, next=1)
    ring.cpu_release(0)

    # --- 2. Self-overlapping destination rows (dst_stride < bpr) ---------
    ring.cpu_write_blit(1, src=0, dst=96, rows=4, bpr=8,
                        src_stride=8, dst_stride=2)
    ring.cpu_release(1)

    engine = BlitEngine(ring)
    engine.submit_chain(0)
    engine.run()
    (result,) = engine.results
    print("=== vertical flip (dst_stride=-8) ===")
    print(f"chain status={result.status} bytes={result.bytes_copied} "
          f"descriptors={result.descriptors}")
    for row in range(4):
        print(f"  dst row {row}: {show(ring.pool, 64 + row * 8, 72 + row * 8)}")
    print(f"  src row {3 - 0} was: {show(before, 24, 32)} (now dst row 0)")

    print("\n=== self-overlapping dst rows (dst_stride=2, bpr=8) ===")
    print("(completed as descriptor 1 of the chain above)")
    print(f"  dst window: {show(ring.pool, 96, 114)}")
    print(f"  padding after window untouched: "
          f"{show(ring.pool, 114, 120)} == {show(before, 114, 120)}")

    # --- 3. Out-of-range descriptor: rejected, nothing written -----------
    pool_before_bad = bytes(ring.pool)
    ring.cpu_write_blit(2, src=0, dst=120, rows=2, bpr=16,
                        src_stride=8, dst_stride=16)
    ring.cpu_release(2)
    engine.submit_chain(2)
    engine.run()
    print("\n=== out-of-range blit rejected ===")
    print(f"chain status={engine.results[-1].status} "
          f"error={engine.results[-1].error!r}")
    print(f"engine halted={engine.halted} "
          f"pool unchanged={bytes(ring.pool) == pool_before_bad}")
    comp = ring.cpu_acquire(2)
    print(f"completion: status={comp.status} bytes_done={comp.bytes_done}")

    # --- 4. Empty rectangle: successful zero transfer --------------------
    engine.reset_halt()
    ring.cpu_write_blit(3, src=0, dst=64, rows=0, bpr=8,
                        src_stride=8, dst_stride=8)
    ring.cpu_release(3)
    engine.submit_chain(3)
    engine.run()
    comp = ring.cpu_acquire(3)
    print("\n=== empty rectangle (rows=0) ===")
    print(f"chain status={engine.results[-1].status} "
          f"bytes_done={comp.bytes_done} "
          f"completions={ring.completion_count(3)}")


if __name__ == "__main__":
    main()
