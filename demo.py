"""Demo: one correct DMA-ring copy, one real visibility rejection, and one
incrementally stepped copy that is cancelled halfway through.

Runs in well under 8 seconds, no sleeps, no recorded output.  All data is
generated deterministically in-repo.
"""

from dma_ring import CopyEngine, DmaRing, SteppedCopyEngine, VisibilityError


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

    # --- 3. Stepped copy cancelled halfway through --------------------------
    # Chain 2 -> 3 -> 4, three 16-byte descriptors, driven one step at a
    # time.  After the first descriptor is copied the caller cancels; the
    # next step confirms the cancellation and hands the remaining slots
    # back as "skipped".
    ring.cpu_write(2, offset=0, length=16, next=3)
    ring.cpu_release(2)
    ring.cpu_write(3, offset=16, length=16, next=4)
    ring.cpu_release(3)
    ring.cpu_write(4, offset=32, length=16, next=None)
    ring.cpu_release(4)

    stepped = SteppedCopyEngine(ring)
    req = stepped.submit(2)
    print("\n=== stepped copy with cancellation ===")
    print(f"submitted request id={req.request_id} slots={req.slots} "
          f"generations={req.generations}")

    event = stepped.step()
    print(f"step 1: {event.kind} descriptor {event.index} "
          f"({event.bytes_copied} bytes copied so far)")
    print(f"partial output: {stepped.output.hex()}")

    print(f"cancel registered: {stepped.cancel(req)} "
          f"(repeat: {stepped.cancel(req)})")
    print(f"result before confirmation: {stepped.result(req)}")

    event = stepped.step()
    print(f"step 2: {event.kind}, unprocessed slots returned {event.skipped}")
    result = stepped.result(req)
    print(f"final: status={result.status} "
          f"bytes_copied={result.bytes_copied} "
          f"descriptors={result.descriptors} skipped={result.skipped}")
    for idx in (2, 3, 4):
        comp = ring.cpu_acquire(idx)
        print(f"  desc {idx}: status={comp.status} "
              f"bytes_done={comp.bytes_done}")


if __name__ == "__main__":
    main()
