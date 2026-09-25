"""Demo: one correct DMA-ring copy, one real visibility rejection, and one
step-driven cancellable copy (partial copy + cancel confirmation).

Runs in well under 8 seconds, no sleeps, no recorded output.  All data is
generated deterministically in-repo.
"""

from dma_ring import (
    CopyEngine,
    DmaRing,
    OwnershipError,
    SteppedCopyEngine,
    VisibilityError,
)


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

    # --- 3. Stepped engine: partial copy, then a confirmed cancel ---------
    # Chain 2 -> 3 -> 4 -> 5, four 16-byte segments.
    print("\n=== stepped cancellable copy ===")
    stepped = SteppedCopyEngine(ring)
    for pos, idx in enumerate((2, 3, 4, 5)):
        nxt = (2, 3, 4, 5)[pos + 1] if pos + 1 < 4 else None
        ring.cpu_write(idx, offset=pos * 16, length=16, next=nxt)
        ring.cpu_release(idx)
    req = stepped.submit(2)
    print(f"submitted {req!r}; chain slots frozen at generation 1")

    stepped.step()  # device copies descriptor 2 only
    print(f"after 1 step: {len(stepped.output)} bytes copied "
          f"(partial: {stepped.output.hex()})")

    req.cancel()
    print("cancel registered; slots are still device-owned until the "
          "device confirms:")
    try:
        ring.cpu_write(3, offset=0, length=1)
        print("ERROR: cpu_write unexpectedly succeeded")
    except OwnershipError as exc:
        print(f"  cpu_write rejected before confirmation: {exc}")

    stepped.step()  # device confirms the cancel
    result = req.result()
    print(f"cancel confirmed: status={result.status} "
          f"bytes_copied={result.bytes_copied} "
          f"descriptors={result.descriptors} skipped={result.skipped}")
    for idx in result.skipped:
        comp = ring.cpu_acquire(idx)
        print(f"  desc {idx}: handed back with status={comp.status}")
    ring.cpu_acquire(2)  # the copied prefix keeps its ok completion

    # --- 4. Completion wins over a later cancel ---------------------------
    print("\n=== completion beats a late cancel ===")
    for pos, idx in enumerate((2, 3, 4, 5)):
        nxt = (2, 3, 4, 5)[pos + 1] if pos + 1 < 4 else None
        ring.cpu_write(idx, offset=48 + pos * 12, length=12, next=nxt)
        ring.cpu_release(idx)  # generation 2 on the recycled slots
    req2 = stepped.submit(2)
    stepped.drain()  # run to completion
    print(f"drained: status={req2.result().status} "
          f"bytes_copied={req2.result().bytes_copied}")
    print(f"cancel after completion is a no-op: {req.cancel()!r} (old "
          f"handle), {req2.cancel()!r} (finished handle)")
    print(f"old handle still reports its own terminal state: "
          f"{req.result().status}; new request stays "
          f"{req2.result().status}")


if __name__ == "__main__":
    main()
