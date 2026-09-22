"""demo.py -- fixed-input demonstration of the dmaring module.

Runs entirely offline with deterministic, in-code data.  Shows:
  1. one correct cross-ring scatter/gather copy (checked against the
     independent reference implementation), and
  2. one actually-triggered visibility rejection (CPU reads completion
     fields without a cpu_acquire), plus a half-chain error boundary.

No sleeps, no recorded output: everything is computed on each run.
"""

from dmaring import (
    CoherentMemory,
    DmaRing,
    SyntheticDevice,
    VisibilityError,
    reference_gather_copy,
)


def build_source():
    """Deterministic 256-byte buffer; readable fragments at fixed offsets."""
    buf = bytearray(((i * 31 + 7) ^ (i >> 1)) & 0xFF for i in range(256))
    buf[0:10] = b"HELLO-DMA "
    buf[32:37] = b"RING "
    buf[64:72] = b"COPY-OK"
    return bytes(buf)


def main():
    print("=== dmaring demo: DMA ring descriptor ownership handoff ===")

    # -- setup ---------------------------------------------------------------
    src = build_source()
    mem = CoherentMemory(256)
    mem.cpu_write(0, src)
    mem.flush()  # publish CPU writes to the device view

    rings = {0: DmaRing(4, ring_id=0), 1: DmaRing(2, ring_id=1)}
    dev = SyntheticDevice(mem)

    # -- 1. correct cross-ring scatter/gather copy ---------------------------
    # Chain: (ring 0, slot 0) -> (ring 1, slot 0) -> (ring 0, slot 1) -> END
    rings[0].cpu_write(0, offset=0, length=10, next=(1, 0))
    rings[1].cpu_write(0, offset=32, length=5, next=(0, 1))
    rings[0].cpu_write(1, offset=64, length=7, next=None)
    for ring_id, slot in ((0, 0), (1, 0), (0, 1)):
        rings[ring_id].cpu_release(slot)

    result = dev.process_chain(rings, (0, 0))
    print(f"\n[1] chain status={result.status} bytes_copied={result.bytes_copied}")
    print(f"    copied bytes : {bytes(dev.output)!r}")
    print(f"    decoded text : {bytes(dev.output).decode('ascii')}")

    expected = reference_gather_copy(src, [[(0, 10), (32, 5), (64, 7)]])
    print(f"    reference    : {expected!r}")
    assert bytes(dev.output) == expected, "core output disagrees with reference"
    print("    reference comparison: MATCH")

    counts = {k: dev.completion_counts[k] for k in sorted(dev.completion_counts)}
    print(f"    per-item completion counts (ring,slot)->n: {counts}")

    # -- 2. visibility rejection (actually triggered) ------------------------
    # The device has released every descriptor back to the CPU, but the CPU
    # has not acquired any completion yet: reading completion fields now
    # must be refused.
    print("\n[2] reading completion fields before cpu_acquire ...")
    try:
        rings[0].completion(0)
    except VisibilityError as exc:
        print(f"    REJECTED with VisibilityError: {exc}")
    else:
        raise AssertionError("visibility violation was not rejected")

    # Legal path: acquire first, then read.
    for ring_id, slot in ((0, 0), (1, 0), (0, 1)):
        comp = rings[ring_id].cpu_acquire(slot)
        print(f"    acquired (ring={comp.ring_id}, slot={comp.slot}): "
              f"gen={comp.generation} status={comp.status} "
              f"bytes={comp.bytes_copied}")

    # -- 3. boundary: half-chain error ---------------------------------------
    # A fresh device on a fresh memory; chain of three where the middle
    # descriptor's buffer range is out of bounds.
    print("\n[3] half-chain error boundary ...")
    mem2 = CoherentMemory(64)
    mem2.cpu_write(0, src[:64])
    mem2.flush()
    ring = DmaRing(4, ring_id=0)
    dev2 = SyntheticDevice(mem2)
    ring.cpu_write(0, offset=0, length=8, next=(0, 1))
    ring.cpu_write(1, offset=60, length=16, next=(0, 2))  # 60+16 > 64: bad
    ring.cpu_write(2, offset=8, length=8, next=None)
    for slot in (0, 1, 2):
        ring.cpu_release(slot)
    bad = dev2.process_chain(ring, (0, 0))
    print(f"    chain status={bad.status}")
    print(f"    error: {bad.error}")
    print(f"    items: {[(c.slot, c.status) for c in bad.items]}")
    print(f"    tail slot 2 consumed? completions="
          f"{dev2.completion_count(0, 2)} (0 means not consumed)")
    print(f"    device halted: {dev2.halted}")

    print("\n=== demo finished: all outputs computed live ===")


if __name__ == "__main__":
    main()
