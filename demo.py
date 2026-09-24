"""Demo: DMA-ring ownership handoff, chains and multi-descriptor requests.

Runs in well under a second (no sleeps, no recorded output).  All pool
data is generated deterministically in-repo; expected bytes in the output
below are hand-specified slices of that data, never produced by the code
under test.
"""

from dma_ring import (
    CopyEngine,
    DmaRing,
    DuplicateCompletionError,
    GenerationError,
    VisibilityError,
)

POOL = bytes((i * 7 + 3) % 256 for i in range(96))


def release(ring, index, *, offset, length, next=None):
    ring.cpu_write(index, offset=offset, length=length, next=next)
    return ring.cpu_release(index)


def section(title):
    print(f"\n=== {title} ===")


def main() -> None:
    ring = DmaRing(capacity=8, pool_size=96)
    ring.pool[:] = POOL

    # --- 1. Legacy entry point: one correct scatter/gather chain ----------
    # Chain 6 -> 7 -> 0 gathers three non-contiguous pool ranges.
    release(ring, 6, offset=0, length=8, next=7)
    release(ring, 7, offset=32, length=8, next=0)
    release(ring, 0, offset=64, length=8, next=None)
    engine = CopyEngine(ring)
    engine.submit_chain(6)
    engine.run()
    chain_result = engine.results[0]
    section("1. correct scatter/gather chain (legacy interface)")
    print(f"status={chain_result.status} "
          f"descriptors={chain_result.descriptors} "
          f"bytes={chain_result.bytes_copied}")
    print(f"output: {engine.output.hex()}")
    print("hand expectation : "
          f"{(POOL[0:8] + POOL[32:40] + POOL[64:72]).hex()}")

    # --- 2. Request: descriptors complete out of order --------------------
    # Request members [2, 4, 5]; the device completes them in the order
    # 5, 2, 4.  The aggregate result appears only after ALL three finish,
    # and lists members in request (membership) order.
    for idx, (off, ln) in ((2, (8, 4)), (4, (40, 8)), (5, (72, 2))):
        release(ring, idx, offset=off, length=ln)
    rid = engine.submit_request([2, 4, 5], order=[5, 2, 4])
    print(f"\nrequest {rid} submitted; before run, "
          f"done={ring.request_is_done(rid)} (no result yet)")
    engine.run()
    result = ring.request_acquire(rid)
    section("2. request with out-of-order completion")
    print(f"request {rid}: status={result.status} "
          f"bytes_copied={result.bytes_copied}")
    for comp in result.completions:
        print(f"  member {comp.index}: status={comp.status} "
              f"generation={comp.generation} bytes_done={comp.bytes_done}")
    print(f"engine output this request: "
          f"{engine.output[24:].hex()}")
    print("hand expectation (order 5,2,4): "
          f"{(POOL[72:74] + POOL[8:12] + POOL[40:48]).hex()}")
    # The request result is available and the per-slot completions are ok.
    assert [c.index for c in result.completions] == [2, 4, 5]

    # --- 3. Boundary: half-completed request, then ring wrap --------------
    # A fresh ring: error request frees every slot, which then wrap into a
    # healthy request whose recorded generations are 2.
    section("3. boundary: error request frees slots; they wrap immediately")
    r2 = DmaRing(capacity=4, pool_size=64)
    r2.pool[:] = POOL[:64]
    # member 2 points out of the pool; completion order reaches it second.
    specs = {0: (0, 4), 1: (8, 4), 2: (32, 64), 3: (16, 4)}
    for idx, (off, ln) in specs.items():
        release(r2, idx, offset=off, length=ln)
    eng2 = CopyEngine(r2)
    rid2 = eng2.submit_request([0, 1, 2, 3], order=[1, 2, 0, 3])
    eng2.run()
    bad = r2.request_acquire(rid2)
    print(f"request {rid2}: status={bad.status} "
          f"error_index={bad.error_index} bytes_copied={bad.bytes_copied}")
    print("  error: " + str(bad.error))
    for comp in bad.completions:
        print(f"  member {comp.index}: status={comp.status} "
              f"bytes_done={comp.bytes_done}")
    print(f"engine halted={eng2.halted}, queued={eng2.queued}")
    print("half-completed prefix copied (hand): "
          f"{POOL[8:12].hex()}")
    # Every slot is back to the CPU; recycle all four into request #2.
    for idx in range(4):
        r2.cpu_acquire(idx)
        release(r2, idx, offset=4 * idx, length=2)
    eng2.reset_halt()
    rid3 = eng2.submit_request([0, 1, 2, 3], order=[3, 2, 1, 0])
    eng2.run()
    good = r2.request_acquire(rid3)
    print(f"wrapped request {rid3}: status={good.status} "
          f"bytes_copied={good.bytes_copied} "
          f"generations={[c.generation for c in good.completions]}")
    print("hand expectation: "
          f"{(POOL[6:8] + POOL[4:6] + POOL[2:4] + POOL[0:2]).hex()}")

    # --- 4. Real errors: stale and duplicate completion rejections --------
    section("4. stale and duplicate completions are rejected")
    r3 = DmaRing(capacity=4, pool_size=32)
    r3.pool[:] = POOL[:32]
    g0 = release(r3, 0, offset=0, length=4)
    g1 = release(r3, 1, offset=8, length=4)
    rid4, gens = r3.register_request([0, 1])
    print(f"request {rid4} members=[0,1] generations={gens}")
    # Complete member 0 once, correctly.
    snap = r3.device_acquire(0)
    r3.device_release(0, status="ok", bytes_done=snap.length, request_id=rid4)
    print(f"member 0 completed ok at generation {g0}")
    # Replay the same completion: duplicate, rejected, no second record.
    try:
        r3.device_release(0, status="ok", bytes_done=4, request_id=rid4)
        print("ERROR: duplicate completion was accepted")
    except DuplicateCompletionError as exc:
        print(f"duplicate rejected: {exc}")
    # CPU recycles slot 0 (generation advances) while the request is open.
    r3.cpu_acquire(0)
    g0b = release(r3, 0, offset=16, length=2)
    try:
        r3.device_release(0, status="ok", bytes_done=2, request_id=rid4)
        print("ERROR: stale completion was accepted")
    except GenerationError as exc:
        print(f"stale rejected: {exc}")
    print(f"completion_count(0)={r3.completion_count(0)} "
          f"(exactly 1); slot 0 is now generation {g0b}, untouched")
    # The open request still finishes normally on member 1 plus the gen-1
    # member 0 record; the request result covers generation 1 only.
    snap = r3.device_acquire(1)
    r3.device_release(1, status="ok", bytes_done=snap.length, request_id=rid4)
    final = r3.request_acquire(rid4)
    print(f"request {rid4} still published: status={final.status} "
          f"bytes_copied={final.bytes_copied} "
          f"(member gens {[c.generation for c in final.completions]})")

    # --- 5. Legacy visibility rejection still enforced --------------------
    section("5. visibility rejection (legacy protocol)")
    release(ring, 3, offset=0, length=8)
    try:
        ring.cpu_read_completion(3)
        print("ERROR: read unexpectedly succeeded")
    except VisibilityError as exc:
        print(f"cpu_read_completion rejected: {exc}")


if __name__ == "__main__":
    main()
