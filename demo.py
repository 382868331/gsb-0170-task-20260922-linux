"""Demo: one correct DMA-ring copy, one real visibility rejection, and one
coalesced-notification run.

Runs in well under 8 seconds, no sleeps, no recorded output.  All data is
generated deterministically in-repo.
"""

from dma_ring import CopyEngine, DmaRing, NotificationCoalescer, VisibilityError


def main() -> None:
    # Deterministic pool contents generated locally.
    ring = DmaRing(capacity=8, pool_size=96)
    ring.pool[:] = bytes((i * 7 + 3) % 256 for i in range(96))

    # --- 1. A correct scatter/gather copy across the ring wrap ------------
    # Chain 6 -> 7 -> 0 gathers three non-contiguous pool ranges.
    ring.cpu_write(6, offset=0, length=16, next=7)
    ring.cpu_release(6)
    ring.cpu_write(7, offset=32, length=16, next=0)
    ring.cpu_release(7)
    ring.cpu_write(0, offset=64, length=16, next=None)
    ring.cpu_release(0)

    engine = CopyEngine(ring)
    engine.submit_chain(6)
    engine.run()

    (result,) = engine.results
    print("=== correct copy ===")
    print(f"chain head=6 status={result.status} "
          f"descriptors={result.descriptors} bytes={result.bytes_copied}")
    print(f"output: {engine.output.hex()}")
    for idx in result.descriptors:
        comp = ring.cpu_acquire(idx)
        print(f"  desc {idx}: status={comp.status} "
              f"bytes_done={comp.bytes_done} generation={comp.generation} "
              f"completions={ring.completion_count(idx)}")

    # --- 2. A real visibility rejection ------------------------------------
    # CPU releases descriptor 1 to the device, then illegally tries to read
    # completion fields without the device having released (and without a
    # cpu_acquire): the read is rejected.
    ring.cpu_write(1, offset=0, length=8)
    ring.cpu_release(1)
    print("\n=== visibility rejection ===")
    try:
        ring.cpu_read_completion(1)
        print("ERROR: read unexpectedly succeeded")
    except VisibilityError as exc:
        print(f"cpu_read_completion rejected: {exc}")

    # Legal sequence on the same descriptor, for contrast.
    snap = ring.device_acquire(1)
    ring.device_release(1, status="ok", bytes_done=snap.length)
    comp = ring.cpu_acquire(1)
    print(f"legal handoff afterwards: status={comp.status} "
          f"bytes_done={comp.bytes_done}")

    # --- 3. Coalesced completion notifications -----------------------------
    # Threshold 3, timeout 4 ticks: one notification per 3 completions, or
    # one when the oldest unnotified completion has waited 4 ticks.
    ring2 = DmaRing(capacity=4, pool_size=64)
    ring2.pool[:] = bytes((i * 5 + 1) % 256 for i in range(64))
    coalescer = NotificationCoalescer(ring2, threshold=3, timeout=4)
    engine2 = CopyEngine(ring2)

    print("\n=== coalesced notifications (threshold=3, timeout=4) ===")
    for idx in (0, 1, 2):
        ring2.cpu_write(idx, offset=idx * 8, length=8)
        ring2.cpu_release(idx)
        engine2.submit_chain(idx)
    engine2.run()  # 3 completions at t=0 -> one threshold notification

    # Data visibility is NOT delayed: the CPU can acquire completion 0
    # before any notification has been read.
    early = ring2.cpu_acquire(0)
    print(f"cpu_acquire before notification read: status={early.status} "
          f"bytes_done={early.bytes_done} (visibility not delayed)")

    note = coalescer.read()
    print(f"t={note.time} [{note.reason}] "
          f"records={[(r.index, r.generation, r.status) for r in note.records]}")
    print(f"re-read consumes nothing: {coalescer.read()}")

    # One more completion at t=0: below threshold, waits for the timeout.
    ring2.cpu_write(3, offset=24, length=8)
    ring2.cpu_release(3)
    engine2.submit_chain(3)
    engine2.run()
    print(f"after 1 more completion: events_pending={coalescer.events_pending}"
          f" pending_count={coalescer.pending_count}"
          f" deadline={coalescer.next_deadline}")
    fired = coalescer.advance(4)  # t=4 == deadline: boundary equality fires
    for note in fired:
        print(f"t={note.time} [{note.reason}] "
              f"records={[(r.index, r.generation, r.status) for r in note.records]}")

    # Explicit flush on an empty batch produces no event.
    print(f"empty flush -> {coalescer.flush()}")


if __name__ == "__main__":
    main()
