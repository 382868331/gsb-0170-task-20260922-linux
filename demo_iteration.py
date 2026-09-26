"""Demo: BlitEngine 2D strided blits on the shared DMA-ring byte pool.

Shows one vertical flip with self-overlapping destination rows, one
submit-time source snapshot with partial overlap, and one real
out-of-bounds rejection.  Runs in well under 90 seconds, fully offline,
no sleeps, no recorded output; all data is generated deterministically
in-repo.
"""

from dma_ring import BlitEngine, ChainError, DmaRing


def show_pool(ring, lo, hi, label):
    print(f"  {label} pool[{lo}:{hi}] = {bytes(ring.pool[lo:hi]).hex()}")


def main() -> None:
    ring = DmaRing(capacity=8, pool_size=96)
    ring.pool[:] = bytes((i * 7 + 3) % 256 for i in range(96))
    engine = BlitEngine(ring)

    # --- 1. Vertical flip into self-overlapping destination rows ---------
    # 4 rows of 8 bytes from src (stride 8) land at dst=40 with
    # dst_stride=-6: row order is flipped and destination rows overlap, so
    # later logical rows overwrite earlier ones in the overlap.
    ring.cpu_write_blit(0, src=0, dst=40, rows=4, row_bytes=8,
                        src_stride=8, dst_stride=-6)
    ring.cpu_release(0)
    engine.submit_chain(0)
    engine.run()
    (result,) = engine.results
    print("=== vertical flip with self-overlapping dst rows ===")
    print(f"chain head=0 status={result.status} "
          f"bytes_blitted={result.bytes_blitted}")
    show_pool(ring, 22, 48, "dst span")
    comp = ring.cpu_acquire(0)
    print(f"  desc 0: status={comp.status} bytes_done={comp.bytes_done} "
          f"(payload only, no stride padding)")

    # --- 2. Partial overlap interpreted from the submit-time snapshot ----
    # src [48, 72) and dst [54, 78) overlap; the transfer uses the source
    # bytes as of the successful submit, so the result is deterministic.
    ring.cpu_write_blit(1, src=48, dst=54, rows=3, row_bytes=8,
                        src_stride=8, dst_stride=8)
    ring.cpu_release(1)
    before = bytes(ring.pool)
    engine.submit_chain(1)
    ring.pool[48] = 0xEE  # host mutation after submit: must not matter
    engine.run()
    print("\n=== partial overlap from submit-time snapshot ===")
    print(f"  src[48] before submit = 0x{before[48]:02x}; "
          f"mutated to 0xee after submit")
    print(f"  dst[54] after run     = 0x{ring.pool[54]:02x} "
          f"(snapshot value, not 0xee)")
    comp = ring.cpu_acquire(1)
    print(f"  desc 1: status={comp.status} bytes_done={comp.bytes_done}")

    # --- 3. A real out-of-bounds rejection --------------------------------
    # rows=6 with dst_stride=-16 from dst=64 generates a row at -16:
    # the whole request is rejected at submit, nothing is written.
    ring.cpu_write_blit(2, src=0, dst=64, rows=6, row_bytes=8,
                        src_stride=8, dst_stride=-16)
    ring.cpu_release(2)
    print("\n=== out-of-bounds rejection ===")
    pool_before = bytes(ring.pool)
    try:
        engine.submit_chain(2)
        print("ERROR: submit unexpectedly succeeded")
    except ChainError as exc:
        print(f"submit_chain rejected: {exc} "
              f"(index={exc.index}, reason={exc.reason})")
    print(f"  pool unchanged: {bytes(ring.pool) == pool_before}; "
          f"queued chains: {engine.queued}")
    comp = ring.cpu_acquire(2)
    print(f"  desc 2 completed: status={comp.status} "
          f"bytes_done={comp.bytes_done}")


if __name__ == "__main__":
    main()

