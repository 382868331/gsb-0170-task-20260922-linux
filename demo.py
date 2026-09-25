"""Demo: DMA-ring copy, a visibility rejection, and notification coalescing.

Runs in well under 90 seconds, no sleeps, no recorded output.  All data is
generated deterministically in-repo.
"""

from dma_ring import (
    CompletionCoalescer,
    CopyEngine,
    DmaRing,
    VisibilityError,
)


def complete(ring, index, *, offset=0, length=4, status="ok", error=None):
    """Drive one descriptor to a device completion at the current clock."""
    ring.cpu_write(index, offset=offset, length=length)
    ring.cpu_release(index)
    ring.device_acquire(index)
    ring.device_release(
        index,
        status=status,
        bytes_done=length if status == "ok" else 0,
        error=error,
    )


def show_batches(title, coal):
    print(title)
    batches = coal.notifications()
    if not batches:
        print("  (no unread notifications)")
    for b in batches:
        ids = ", ".join(f"slot{e.index}/gen{e.generation}"
                        for e in b.completions)
        print(f"  t={b.time} reason={b.reason}: [{ids}]")
    # A repeated read never re-consumes the same events.
    assert coal.notifications() == [], "notification was consumed twice"


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
    # CPU releases descriptor 1, then illegally reads completion fields
    # without any device release / cpu_acquire: the read is rejected.
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

    # --- 3. Notification coalescing ----------------------------------------
    # A fresh small ring so every generation is easy to read.  Threshold =
    # 3 completions, timeout = 4 logical ticks after the oldest pending one.
    print("\n=== notification coalescer (threshold=3, timeout=4) ===")
    cring = DmaRing(capacity=2, pool_size=16)
    cring.pool[:] = bytes((i * 7 + 3) % 256 for i in range(16))
    coal = CompletionCoalescer(cring, count_threshold=3, timeout=4)

    # An empty flush produces no event.
    assert coal.flush() is None
    print("empty flush -> None")

    complete(cring, 0, length=2)                 # t=0, deadline t=4
    coal.advance(2)                             # t=2
    complete(cring, 1, length=2)                # t=2
    coal.advance(1)                             # t=3: one tick early
    show_batches("at t=3 (threshold/timeout not yet due):", coal)

    # Slot 0 is wrapped and reused: generation disambiguates the old record.
    cring.cpu_acquire(0)
    complete(cring, 0, length=2)                # t=3, same slot, gen 2
    show_batches("3rd completion at t=3 -> count batch:", coal)

    # A device error counts as a completion and sits next to the ok chain.
    cring.cpu_acquire(1)
    complete(cring, 1, length=2, status="error",
             error="descriptor 1: simulated bad buffer")  # t=3
    coal.advance(4)                             # t=7 == 3+4 -> timeout
    show_batches("at t=7 -> timeout batch (error included, ordered):", coal)

    # Data visibility was never gated on notifications: the errored slot is
    # acquirable directly regardless of notification state.
    bad = cring.cpu_acquire(1)
    print(f"error slot readable independently of notification: "
          f"status={bad.status} generation={bad.generation}")

    # Explicit flush on an empty queue is still a no-op.
    print(f"empty flush after drain -> {coal.flush()}")


if __name__ == "__main__":
    main()
