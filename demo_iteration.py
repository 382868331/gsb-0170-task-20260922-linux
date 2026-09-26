"""Offline demo of the periodic capture mode (runs in well under 90 s).

Scenarios, all computed live with deterministic data, no sleeps, no
network, no recorded output:

1. lossless consumption (claim keeps up)
2. slow consumer: exact lost sequence ranges are reported, never faked
3. exactly one lap boundary
4. counter wrap simulation (finite hardware counter width)
5. stop with a half-written segment, then restart
"""

from dma_ring import OverrunError, PeriodicRing


def expected(sequence, size):
    base = sequence * size
    return bytes((base + j) & 0xFF for j in range(size))


def show(result):
    for report in result.lost:
        print(f"  LOST sequences [{report.lost_start}..{report.lost_end}] "
              f"({report.lost_count} segments overwritten)")
    for seg in result.segments:
        ok = "ok" if seg.data == expected(seg.sequence, len(seg.data)) else "CORRUPT"
        print(f"  seq={seg.sequence:3d} slot={seg.slot} lap={seg.lap} "
              f"gen={seg.generation} data={seg.data.hex()} [{ok}]")


def main() -> None:
    print("=== 1. lossless consumption (capacity=4, 8-byte segments) ===")
    ring = PeriodicRing(4, 8)
    ring.start()
    for _ in range(3):
        ring.hw_advance(2)
        show(ring.claim())
    print(f"  totals: completed={ring.total_completed} lost={ring.total_lost}")
    ring.stop()

    print("\n=== 2. slow consumer: losses reported, not hidden ===")
    ring = PeriodicRing(4, 8)
    ring.start()
    ring.hw_advance(10)  # software never claims; lap 2/3 overwrite lap 1
    result = ring.claim()
    show(result)
    print(f"  delivered={len(result.segments)} lost_count={result.lost_count}")
    ring.stop()

    print("\n=== 3. exactly one lap boundary ===")
    ring = PeriodicRing(4, 4)
    ring.start()
    ring.hw_advance(4)  # one full lap, software claims in time
    print("  after one lap, claimed in time:",
          "lost", ring.claim().lost_count)
    ring.hw_advance(4)  # lap 2 left unclaimed
    ring.hw_advance(4)  # lap 3 overwrites exactly one unclaimed lap
    result = ring.claim()
    show(result)
    ring.stop()

    print("\n=== 4. counter wrap simulation (3-bit counter, modulus 8) ===")
    ring = PeriodicRing(4, 2, counter_bits=3)
    ring.start()
    for _ in range(3):
        ring.hw_advance(4)
        ring.claim()
    print(f"  infinite next_sequence={ring.next_sequence} "
          f"hw_counter={ring.hw_counter()} (wrapped)")
    ring.stop()
    ring2 = PeriodicRing(4, 2, counter_bits=4)
    ring2.start()
    try:
        ring2.hw_advance(17)  # starting seq 16: lag hits the modulus
    except OverrunError as exc:
        print(f"  OverrunError raised as specified: {exc}")

    print("\n=== 5. stop with half segment, then restart ===")
    ring = PeriodicRing(4, 8)
    ring.start()
    ring.hw_advance(2)
    ring.hw_write_bytes(expected(2, 8)[:5])  # half of segment 2
    partial = ring.stop()
    print(f"  stopped mid-segment: seq={partial.sequence} "
          f"bytes_written={partial.bytes_written}/{partial.segment_size}")
    show(ring.claim())  # completed segments survive the stop
    ring.start()
    ring.hw_step()  # seq 2 rewritten from byte 0
    show(ring.claim())
    ring.stop()


if __name__ == "__main__":
    main()
