"""Demo: DMA ring handoff plus multi-descriptor requests.

Runs in well under 90 seconds, no sleeps, no recorded output.  All data is
generated deterministically in-repo.  Sections:

1. a correct scatter/gather chain copy (original ring protocol);
2. a real visibility rejection (original ring protocol);
3. a multi-descriptor request completed out of order (new);
4. boundary cases: a request filling the whole ring and a zero-length
   descriptor (new);
5. real request errors: stale-generation completion after ring wrap, a
   duplicate completion, and an errored request that releases every slot
   it occupied (new).
"""

from dma_ring import (
    CopyEngine,
    DmaRing,
    RequestEngine,
    RequestError,
    VisibilityError,
)


def write_release(ring, index, offset, length, next=None):
    ring.cpu_write(index, offset=offset, length=length, next=next)
    return ring.cpu_release(index)


def main() -> None:
    # Deterministic pool contents generated locally.
    ring = DmaRing(capacity=8, pool_size=96)
    ring.pool[:] = bytes((i * 7 + 3) % 256 for i in range(96))

    # --- 1. A correct scatter/gather copy across the ring wrap ------------
    # Chain 6 -> 7 -> 0 gathers three non-contiguous pool ranges.
    write_release(ring, 6, 0, 16, next=7)
    write_release(ring, 7, 32, 16, next=0)
    write_release(ring, 0, 64, 16, next=None)

    engine = CopyEngine(ring)
    engine.submit_chain(6)
    engine.run()

    (result,) = engine.results
    print("=== 1. correct chain copy ===")
    print(f"chain head=6 status={result.status} "
          f"descriptors={result.descriptors} bytes={result.bytes_copied}")
    print(f"output: {engine.output.hex()}")
    for idx in result.descriptors:
        comp = ring.cpu_acquire(idx)
        print(f"  desc {idx}: status={comp.status} "
              f"bytes_done={comp.bytes_done} generation={comp.generation} "
              f"completions={ring.completion_count(idx)}")

    # --- 2. A real visibility rejection ------------------------------------
    write_release(ring, 1, 0, 8)
    print("\n=== 2. visibility rejection ===")
    try:
        ring.cpu_read_completion(1)
        print("ERROR: read unexpectedly succeeded")
    except VisibilityError as exc:
        print(f"cpu_read_completion rejected: {exc}")

    snap = ring.device_acquire(1)
    ring.device_release(1, status="ok", bytes_done=snap.length)
    comp = ring.cpu_acquire(1)
    print(f"legal handoff afterwards: status={comp.status} "
          f"bytes_done={comp.bytes_done}")

    # --- 3. Request: descriptors complete out of order ---------------------
    # Descriptors 2,3,4 describe three ranges; the device completes them in
    # scrambled order (4 -> 2 -> 3).  The request result appears only after
    # the final completion.
    print("\n=== 3. request with out-of-order completion ===")
    specs = {2: (0, 4), 3: (8, 4), 4: (16, 4)}
    req = RequestEngine(ring)
    generations = {}
    for idx, (off, length) in specs.items():
        generations[idx] = write_release(ring, idx, off, length)
    acquired = req.submit_request([2, 3, 4])
    print(f"submitted request 0: descriptors={[s.index for s in acquired]} "
          f"generations={[s.generation for s in acquired]}")
    order = [4, 2, 3]
    outcome = None
    for step, idx in enumerate(order, start=1):
        outcome = req.complete_descriptor(
            0, idx, generation=generations[idx], bytes_done=4)
        print(f"  step {step}: descriptor {idx} completed -> "
              f"{'request published' if outcome is not None else 'still pending'}")
    print(f"request status={outcome.status} bytes_copied={outcome.bytes_copied}")
    print("  per-descriptor (submission order):")
    for d in outcome.descriptors:
        print(f"    desc {d.index}: {d.status} bytes_done={d.bytes_done}")
    # Output follows actual completion order 4,2,3; the request total is the
    # sum regardless of order.
    print(f"engine output (completion order): {req.output.hex()}")
    for idx in specs:
        ring.cpu_acquire(idx)

    # --- 4. Boundary cases --------------------------------------------------
    print("\n=== 4. boundary cases ===")
    small = DmaRing(capacity=4, pool_size=16)
    small.pool[:] = bytes((i * 7 + 3) % 256 for i in range(16))
    small_req = RequestEngine(small)
    # A request occupying every slot of the ring, including a zero-length
    # descriptor, completed in reverse order.
    for idx in range(4):
        write_release(small, idx, idx * 4, 0 if idx == 2 else 4)
    small_req.submit_request([0, 1, 2, 3])
    out = None
    for idx in (3, 2, 0, 1):
        length = small_req.acquired_snapshot(0, idx).length
        out = small_req.complete_descriptor(
            0, idx, generation=1, bytes_done=length)
    print(f"full-ring request (4/4 slots, one zero-length): "
          f"status={out.status} bytes_copied={out.bytes_copied}")
    for idx in range(4):
        small.cpu_acquire(idx)

    # --- 5. Real request errors ---------------------------------------------
    print("\n=== 5. request error handling ===")

    # 5a. Stale generation: recycle a slot around the ring, then report a
    # completion carrying the old generation.
    old_generation = write_release(small, 0, 0, 4)
    small_req.submit_request([0], request_id=2)
    small_req.complete_descriptor(2, 0, generation=old_generation,
                                  bytes_done=4)
    small.cpu_acquire(0)
    new_generation = write_release(small, 0, 0, 4)  # recycling bumps gen
    small_req.submit_request([0], request_id=3)
    try:
        small_req.complete_descriptor(3, 0, generation=old_generation,
                                      bytes_done=4)
        print("ERROR: stale completion unexpectedly accepted")
    except RequestError as exc:
        print(f"stale completion rejected ({exc.reason}): {exc}")
    # Slot is still held by request 3 and completes fine with the new gen.
    good = small_req.complete_descriptor(3, 0, generation=new_generation,
                                         bytes_done=4)
    print(f"correct generation then completes: status={good.status}")
    small.cpu_acquire(0)

    # 5b. Duplicate completion of a descriptor inside a still-live request
    # (needs >= 2 descriptors: with one, the first completion would already
    # finish and remove the request).
    gen0 = write_release(small, 0, 0, 4)
    gen1 = write_release(small, 1, 4, 4)
    small_req.submit_request([0, 1], request_id=4)
    small_req.complete_descriptor(4, 1, generation=gen1, bytes_done=4)
    try:
        small_req.complete_descriptor(4, 1, generation=gen1, bytes_done=4)
        print("ERROR: duplicate completion unexpectedly accepted")
    except RequestError as exc:
        print(f"duplicate completion rejected ({exc.reason}): {exc}")
    finished = small_req.complete_descriptor(4, 0, generation=gen0,
                                             bytes_done=4)
    print(f"request then completes normally: status={finished.status}")
    small.cpu_acquire(0)
    small.cpu_acquire(1)

    # 5c. An errored request releases every occupied resource: descriptor 0
    # errors while 1 is already ok and 2,3 were never touched; 2 and 3 come
    # back as 'skipped'.
    gens = {idx: write_release(small, idx, idx * 4, 4)
            for idx in range(4)}
    small_req.submit_request([0, 1, 2, 3], request_id=5)
    small_req.complete_descriptor(5, 1, generation=gens[1], bytes_done=4)
    failed = small_req.complete_descriptor(
        5, 0, generation=gens[0], status="error",
        error="simulated device fault")
    print(f"errored request: status={failed.status} "
          f"error_index={failed.error_index} pending_now={small_req.pending}")
    for d in failed.descriptors:
        print(f"    desc {d.index}: {d.status}"
              + (f" ({d.error})" if d.error else ""))
    # Every slot is back with the CPU and recyclable immediately — the
    # errored request leaked no resources.
    for idx in range(4):
        comp = small.cpu_acquire(idx)
        assert small.inspect(idx)["owner"] == "cpu"
    print("all four slots returned to CPU; ring ready for reuse")


if __name__ == "__main__":
    main()
