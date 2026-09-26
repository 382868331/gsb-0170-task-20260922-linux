"""Offline demo of the periodic acquisition mode (well under 90 seconds).

No sleeps, no recorded output, no accounts or keys: every byte is
generated deterministically in-process and every delivered segment is
checked against an independently computed expectation.  Loss is counted
and printed, not hidden behind a fake lossless stream.
"""

from periodic_dma import PeriodicAcquisition


def expected_segment(seq, size):
    """Independent reference for the deterministic incrementing stream."""
    base = seq * size
    return bytes((base + i) % 256 for i in range(size))


def check(result, size):
    """Verify a claim against the independent expectation; raise on drift."""
    for seg in result.segments:
        assert seg.data == expected_segment(seg.seq, size), seg.seq


def fmt(result):
    got = [s.seq for s in result.segments]
    lost = [f"[{r.first_seq}..{r.last_seq}] x{r.count}" for r in result.lost]
    return f"delivered={got} lost={lost or 'none'}"


def main():
    # --- 1. Slow consumer: exact overwritten intervals --------------------
    print("=== slow consumer (4 segments x 16 bytes) ===")
    acq = PeriodicAcquisition(segment_count=4, segment_size=16)
    for round_no in range(4):
        # Round 0 fills exactly one lap (lossless); later rounds overrun.
        for _ in range(4 if round_no == 0 else 5):
            acq.hw_step()
        result = acq.claim(max_segments=2)  # software only takes 2
        check(result, 16)
        print(f"round {round_no}: {fmt(result)}")
    print(f"cumulative lost segments: {acq.lost_total} "
          f"(loss is reported, not hidden)")

    # --- 2. Exactly one lap behind: still lossless -------------------------
    print("\n=== exactly one lap ===")
    acq = PeriodicAcquisition(4, 8)
    for _ in range(4):
        acq.hw_step()
    result = acq.claim()
    check(result, 8)
    print(f"one full lap buffered: {fmt(result)}")
    # Fresh run: one lap plus one segment without claiming -> seq 0 lost.
    acq = PeriodicAcquisition(4, 8)
    for _ in range(5):
        acq.hw_step()
    result = acq.claim()
    check(result, 8)
    print(f"one segment too many:  {fmt(result)}")

    # --- 3. Stop with a half-written segment, then restart -----------------
    print("\n=== stop / restart ===")
    acq = PeriodicAcquisition(2, 8)
    acq.hw_step()       # seq 0 complete
    acq.hw_write(3)     # seq 1 half-written
    report = acq.stop()
    print(f"stop: completed_pending={report.completed_pending} "
          f"partial={report.partial}")
    result = acq.claim()  # completed segments survive the stop
    check(result, 8)
    print(f"claim while stopped: {fmt(result)}")
    acq.start()
    acq.hw_write(8)     # seq 1 re-written from byte 0
    result = acq.claim()
    check(result, 8)
    print(f"after restart: {fmt(result)} "
          f"(abandoned half segment rewritten deterministically)")

    # --- 4. Wrapping hardware counter --------------------------------------
    print("\n=== counter wraparound (modulus 4) ===")
    acq = PeriodicAcquisition(2, 8, counter_modulus=4)
    for _ in range(9):
        result = acq.step()
        check(result, 8)
    print(f"hw_seq={acq.hw_seq} hw_counter={acq.hw_counter} "
          f"lost_total={acq.lost_total} (infinite seq survives the wrap)")
    print("\nall delivered segments matched the independent expectation")


if __name__ == "__main__":
    main()
