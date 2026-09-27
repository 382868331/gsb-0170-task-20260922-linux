"""Demo: a correct copy, a real visibility rejection, and chain control
(cancel of a queued chain, stop request + device confirmation, stale and
duplicate stop notifications).

Runs in well under 8 seconds, no sleeps, no recorded output.  All data is
generated deterministically in-repo.
"""

from dma_ring import CopyEngine, DmaRing, OwnershipError, VisibilityError


def write_and_release(ring, index, *, offset, length, next=None):
    ring.cpu_write(index, offset=offset, length=length, next=next)
    ring.cpu_release(index)


def main() -> None:
    source = bytes((i * 7 + 3) % 256 for i in range(256))
    ring = DmaRing(capacity=8, pool_size=256)
    ring.pool[:] = source

    # --- 1. A correct scatter/gather copy across the ring wrap ------------
    # Chain 6 -> 7 -> 0 gathers three non-contiguous pool ranges.
    write_and_release(ring, 6, offset=0, length=16, next=7)
    write_and_release(ring, 7, offset=32, length=16, next=0)
    write_and_release(ring, 0, offset=64, length=16, next=None)

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

    # --- 3. Stop request for an executing chain ---------------------------
    # Fresh chain 2 -> 3 -> 4, three 4-byte descriptors.
    write_and_release(ring, 2, offset=16, length=4, next=3)
    write_and_release(ring, 3, offset=20, length=4, next=4)
    write_and_release(ring, 4, offset=24, length=4, next=None)
    engine2 = CopyEngine(ring)
    engine2.submit_chain(2)
    engine2.step()  # descriptor 2 completes; the chain is mid-execution
    token = engine2.request_stop(2)
    print("\n=== stop request: handed-out descriptors not reclaimable ===")
    try:
        ring.cpu_acquire(3)
        print("ERROR: cpu_acquire unexpectedly succeeded")
    except OwnershipError as exc:
        print(f"cpu_acquire before device confirmation rejected: {exc}")
    accepted = engine2.confirm_stop(2, token)
    stopped = engine2.results[-1]
    print(f"confirm_stop accepted={accepted} status={stopped.status} "
          f"completed={stopped.descriptors} skipped={stopped.skipped} "
          f"bytes={stopped.bytes_copied}")
    print(f"descriptor 3 completion: {ring.cpu_acquire(3).status}, "
          f"descriptor 4 completion: {ring.cpu_acquire(4).status}")

    # --- 4. Cancel a never-started chain; stale/duplicate notifications ---
    ring.cpu_acquire(2)  # 3 and 4 were already acquired above
    write_and_release(ring, 2, offset=32, length=4, next=3)
    write_and_release(ring, 3, offset=36, length=4, next=None)
    write_and_release(ring, 5, offset=40, length=4, next=None)
    engine3 = CopyEngine(ring)
    engine3.submit_chain(2)
    engine3.submit_chain(5)
    cancelled = engine3.cancel_chain(2)
    print("\n=== cancel queued chain; old notifications ignored ===")
    print(f"cancelled status={cancelled.status} skipped={cancelled.skipped}")
    # A late confirmation carrying the old chain's token must be ignored
    # (head 2 is now reused by a new submission), and a confirmation that
    # has already been accepted must not release resources twice.
    print(f"stale confirm_stop(2, old token) -> "
          f"{engine3.confirm_stop(2, token)}")
    engine3.run()
    print(f"remaining queued chain output: {engine3.output.hex()} "
          f"(expect {source[40:44].hex()})")
    print(f"completion counts after reuse: "
          f"slot2={ring.completion_count(2)} slot3={ring.completion_count(3)}")


if __name__ == "__main__":
    main()
