"""Synthesised DMA ring descriptor ownership-handoff model.

This module models, in pure software, the ownership protocol of a DMA
descriptor ring as used by offline device-driver kernel work.  No real
hardware, threads, or I/O are involved; visibility of descriptor fields is
simulated with explicit release (flush) / acquire events:

* The CPU stages field writes into a private *pending* view.  Nothing the
  CPU writes is visible to the device until ``cpu_release`` commits the
  pending view (the "flush") and flips ownership to the device.
* The device must ``device_acquire`` a descriptor before reading it; the
  snapshot it sees is exactly what the last CPU release committed.
* The device completes the descriptor with ``device_release``, publishing a
  completion record and flipping ownership back to the CPU.
* The CPU must ``cpu_acquire`` before it may read the completion fields.

Generations: every ``cpu_release`` increments the slot generation, so a
consumer holding a stale expected generation detects the ABA reuse that
occurs when a ring slot wraps and is recycled.

Chains: descriptors carry a ``next`` index (or ``None``) allowing
scatter/gather chains that may wrap around the ring.  ``DmaRing.walk_chain``
validates a whole chain up front; ``CopyEngine`` validates incrementally
while copying.  A bad descriptor completes its chain with an error and the
engine halts without consuming subsequent queued chains.

Only the contract documented here is implemented; this is not a complete
model of any industry-standard DMA controller.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from enum import Enum

MAX_CAPACITY = 64
VALID_STATUSES = ("ok", "error", "skipped")


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class DmaRingError(Exception):
    """Base class for all errors raised by this module."""


class OwnershipError(DmaRingError):
    """An operation was attempted by the side that does not own the slot."""


class VisibilityError(DmaRingError):
    """A side read descriptor/completion fields without a prior acquire."""


class GenerationError(DmaRingError):
    """Expected generation did not match the committed generation (ABA)."""


class ChainError(DmaRingError):
    """A descriptor chain is malformed.

    Attributes:
        index:  ring index of the offending descriptor (or ``None`` when the
                problem is the head index itself).
        reason: short machine-stable reason string.
    """

    def __init__(self, message: str, *, index: int | None = None,
                 reason: str | None = None) -> None:
        super().__init__(message)
        self.index = index
        self.reason = reason


# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------

class Owner(Enum):
    """Which side currently owns a descriptor slot."""

    CPU = "cpu"
    DEVICE = "device"


@dataclass(frozen=True)
class DescriptorSnapshot:
    """Immutable view of the committed descriptor fields."""

    index: int
    generation: int
    offset: int
    length: int
    next: int | None


@dataclass(frozen=True)
class CompletionSnapshot:
    """Immutable view of the committed completion record."""

    index: int
    generation: int
    status: str            # one of VALID_STATUSES
    error: str | None
    bytes_done: int


@dataclass(frozen=True)
class BlitSnapshot:
    """Immutable view of the committed 2D blit descriptor fields.

    src/dst are the byte offsets of row 0 in the pool, rows the number
    of rows, bpr the bytes per row, and src_stride/dst_stride the signed
    per-row address steps (negative allowed).  Row r of the source spans
    [src + r*src_stride, +bpr) and of the destination [dst + r*dst_stride,
    +bpr).
    """

    index: int
    generation: int
    src: int
    dst: int
    rows: int
    bpr: int
    src_stride: int
    dst_stride: int
    next: int | None


@dataclass(frozen=True)
class ChainResult:
    """Outcome of one chain processed by :class:`CopyEngine`.

    ``descriptors`` lists the slots that were completed (ok or error) in
    processing order; slots after a bad descriptor are *not* consumed and do
    not appear here.
    """

    head: int
    status: str            # "ok" or "error"
    error: str | None
    error_index: int | None
    bytes_copied: int
    descriptors: tuple[int, ...]


# ---------------------------------------------------------------------------
# Input validation helpers
# ---------------------------------------------------------------------------

def _as_int(value, name: str) -> int:
    """Coerce an integer argument, rejecting bool and NaN/Infinity.

    Integral floats are accepted; non-integral floats are not.  Nothing is
    silently corrected: every rejection raises a located error naming the
    parameter.
    """
    if isinstance(value, bool):
        raise TypeError(f"{name}: bool is not accepted as an integer (got {value!r})")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise ValueError(f"{name}: NaN/Infinity are not accepted (got {value!r})")
        if value.is_integer():
            return int(value)
        raise TypeError(f"{name}: non-integral float {value!r} is not an integer")
    raise TypeError(f"{name}: expected int, got {type(value).__name__}")


def _check_capacity(value) -> int:
    capacity = _as_int(value, "capacity")
    if not 1 <= capacity <= MAX_CAPACITY:
        raise ValueError(
            f"capacity: must be in [1, {MAX_CAPACITY}], got {capacity}")
    if capacity & (capacity - 1):
        raise ValueError(f"capacity: must be a power of two, got {capacity}")
    return capacity


# ---------------------------------------------------------------------------
# Ring slot (internal)
# ---------------------------------------------------------------------------

class _Slot:
    """One ring slot: committed state plus the CPU's staged (pending) view."""

    __slots__ = (
        "owner", "cpu_held", "device_held", "generation",
        "offset", "length", "next",          # committed (visible) fields
        "p_offset", "p_length", "p_next",    # pending (staged by CPU)
        "completion",                        # committed completion triple
        "completions",                       # lifetime completion count
        "kind", "src", "dst", "rows", "bpr", "src_stride", "dst_stride",
        "p_kind", "p_src", "p_dst", "p_rows", "p_bpr",
        "p_src_stride", "p_dst_stride",        # blit fields (committed/pending)
    )

    def __init__(self) -> None:
        self.owner = Owner.CPU
        self.cpu_held = True
        self.device_held = False
        self.generation = 0
        self.offset = 0
        self.length = 0
        self.next = None
        self.p_offset = 0
        self.p_length = 0
        self.p_next = None
        self.completion: tuple[str, str | None, int] | None = None
        self.completions = 0
        # 2D blit fields (committed + pending); kind selects which field
        # set a release commits as the descriptor's operation.
        self.kind = "copy"
        self.src = 0
        self.dst = 0
        self.rows = 0
        self.bpr = 0
        self.src_stride = 0
        self.dst_stride = 0
        self.p_kind = "copy"
        self.p_src = 0
        self.p_dst = 0
        self.p_rows = 0
        self.p_bpr = 0
        self.p_src_stride = 0
        self.p_dst_stride = 0


# ---------------------------------------------------------------------------
# The ring
# ---------------------------------------------------------------------------

class DmaRing:
    """A synthesised DMA descriptor ring with an attached byte pool.

    Parameters:
        capacity:  number of descriptor slots; a power of two in
                   [1, :data:`MAX_CAPACITY`].
        pool_size: size in bytes of the shared buffer pool that descriptor
                   ``offset``/``length`` ranges refer to.
    """

    def __init__(self, capacity, pool_size) -> None:
        self.capacity = _check_capacity(capacity)
        pool_size = _as_int(pool_size, "pool_size")
        if pool_size < 0:
            raise ValueError(f"pool_size: must be >= 0, got {pool_size}")
        self.pool = bytearray(pool_size)
        self._slots = [_Slot() for _ in range(self.capacity)]

    # -- internal helpers --------------------------------------------------

    def _slot(self, index) -> tuple[_Slot, int]:
        index = _as_int(index, "index")
        if not 0 <= index < self.capacity:
            raise IndexError(
                f"index {index} out of range for capacity {self.capacity}")
        return self._slots[index], index

    # -- CPU side -----------------------------------------------------------

    def cpu_write(self, index, *, offset, length, next=None) -> None:
        """Stage descriptor fields.  Requires CPU ownership.

        All arguments are validated before any state changes, so a rejected
        call leaves the slot untouched.  Buffer-range validity against the
        pool is *not* checked here; that is device-side chain validation
        (see :meth:`walk_chain` and :class:`CopyEngine`).
        """
        slot, i = self._slot(index)
        if slot.owner is not Owner.CPU or not slot.cpu_held:
            raise OwnershipError(
                f"descriptor {i}: CPU cannot write "
                f"(owner={slot.owner.value}, cpu_held={slot.cpu_held})")
        offset = _as_int(offset, "offset")
        length = _as_int(length, "length")
        if offset < 0:
            raise ValueError(f"descriptor {i}: offset {offset} < 0")
        if length < 0:
            raise ValueError(f"descriptor {i}: length {length} < 0")
        if next is not None:
            next = _as_int(next, "next")
        slot.p_offset = offset
        slot.p_length = length
        slot.p_next = next
        slot.p_kind = "copy"

    def cpu_write_blit(self, index, *, src, dst, rows, bpr,
                       src_stride, dst_stride, next=None) -> None:
        """Stage 2D blit descriptor fields.  Requires CPU ownership.

        ``src``/``dst`` are pool byte offsets of row 0, ``rows`` the row
        count, ``bpr`` the bytes per row, and the strides the signed
        per-row address steps (negative allowed, e.g. a vertical flip).
        All arguments are validated before any state changes.  Generated
        row addresses are *not* checked against the pool here; that
        happens when the device accepts the request (see
        :class:`BlitEngine`), mirroring the copy-descriptor contract.
        """
        slot, i = self._slot(index)
        if slot.owner is not Owner.CPU or not slot.cpu_held:
            raise OwnershipError(
                f"descriptor {i}: CPU cannot write "
                f"(owner={slot.owner.value}, cpu_held={slot.cpu_held})")
        src = _as_int(src, "src")
        dst = _as_int(dst, "dst")
        rows = _as_int(rows, "rows")
        bpr = _as_int(bpr, "bpr")
        src_stride = _as_int(src_stride, "src_stride")
        dst_stride = _as_int(dst_stride, "dst_stride")
        if src < 0:
            raise ValueError(f"descriptor {i}: src {src} < 0")
        if dst < 0:
            raise ValueError(f"descriptor {i}: dst {dst} < 0")
        if rows < 0:
            raise ValueError(f"descriptor {i}: rows {rows} < 0")
        if bpr < 0:
            raise ValueError(f"descriptor {i}: bpr {bpr} < 0")
        if next is not None:
            next = _as_int(next, "next")
        slot.p_src = src
        slot.p_dst = dst
        slot.p_rows = rows
        slot.p_bpr = bpr
        slot.p_src_stride = src_stride
        slot.p_dst_stride = dst_stride
        slot.p_next = next
        slot.p_kind = "blit"

    def cpu_release(self, index) -> int:
        """Flush staged fields and hand the descriptor to the device.

        Returns the new generation.  After this call the CPU may not read or
        write the descriptor until the device releases it back.
        """
        slot, i = self._slot(index)
        if slot.owner is not Owner.CPU or not slot.cpu_held:
            raise OwnershipError(
                f"descriptor {i}: CPU cannot release "
                f"(owner={slot.owner.value}, cpu_held={slot.cpu_held})")
        slot.offset = slot.p_offset
        slot.length = slot.p_length
        slot.next = slot.p_next
        slot.kind = slot.p_kind
        slot.src = slot.p_src
        slot.dst = slot.p_dst
        slot.rows = slot.p_rows
        slot.bpr = slot.p_bpr
        slot.src_stride = slot.p_src_stride
        slot.dst_stride = slot.p_dst_stride
        slot.generation += 1
        slot.completion = None
        slot.cpu_held = False
        slot.owner = Owner.DEVICE
        return slot.generation

    def cpu_acquire(self, index) -> CompletionSnapshot:
        """Acquire a descriptor the device has released back.

        Raises :class:`OwnershipError` if the device has not released the
        descriptor yet.
        """
        slot, i = self._slot(index)
        if slot.owner is not Owner.CPU:
            raise OwnershipError(
                f"descriptor {i}: CPU cannot acquire; device has not "
                f"released (owner={slot.owner.value})")
        if slot.cpu_held:
            raise OwnershipError(
                f"descriptor {i}: CPU already holds this descriptor")
        slot.cpu_held = True
        if slot.completion is None:
            raise VisibilityError(
                f"descriptor {i}: no completion has been published")
        status, error, bytes_done = slot.completion
        return CompletionSnapshot(i, slot.generation, status, error, bytes_done)

    def cpu_read_completion(self, index) -> CompletionSnapshot:
        """Read the published completion.  Requires a prior ``cpu_acquire``."""
        slot, i = self._slot(index)
        if not slot.cpu_held:
            raise VisibilityError(
                f"descriptor {i}: CPU must cpu_acquire before reading "
                f"completion fields")
        if slot.completion is None:
            raise VisibilityError(
                f"descriptor {i}: no completion has been published")
        status, error, bytes_done = slot.completion
        return CompletionSnapshot(i, slot.generation, status, error, bytes_done)

    # -- device side --------------------------------------------------------

    def device_acquire(self, index, *, expected_generation=None
                       ) -> DescriptorSnapshot:
        """Acquire a descriptor handed to the device.

        Raises :class:`OwnershipError` if the CPU has not released the
        descriptor (e.g. a missing ``cpu_release``), and
        :class:`GenerationError` if ``expected_generation`` is given and does
        not match the committed generation (stale handle / ABA).
        """
        slot, i = self._slot(index)
        if slot.owner is not Owner.DEVICE:
            raise OwnershipError(
                f"descriptor {i}: device cannot acquire; CPU has not "
                f"released (owner={slot.owner.value})")
        if slot.device_held:
            raise OwnershipError(
                f"descriptor {i}: device already holds this descriptor")
        if expected_generation is not None:
            expected = _as_int(expected_generation, "expected_generation")
            if expected != slot.generation:
                raise GenerationError(
                    f"descriptor {i}: expected generation {expected}, "
                    f"committed generation is {slot.generation} (stale "
                    f"handle / ABA)")
        slot.device_held = True
        return DescriptorSnapshot(
            i, slot.generation, slot.offset, slot.length, slot.next)

    def device_read(self, index) -> DescriptorSnapshot:
        """Read committed descriptor fields.  Requires ``device_acquire``."""
        slot, i = self._slot(index)
        if not slot.device_held:
            raise VisibilityError(
                f"descriptor {i}: device must device_acquire before reading")
        return DescriptorSnapshot(
            i, slot.generation, slot.offset, slot.length, slot.next)

    def device_read_blit(self, index) -> BlitSnapshot:
        """Read committed 2D blit fields.  Requires ``device_acquire``."""
        slot, i = self._slot(index)
        if not slot.device_held:
            raise VisibilityError(
                f"descriptor {i}: device must device_acquire before reading")
        return BlitSnapshot(
            i, slot.generation, slot.src, slot.dst, slot.rows, slot.bpr,
            slot.src_stride, slot.dst_stride, slot.next)

    def device_release(self, index, *, status, bytes_done=0, error=None
                       ) -> None:
        """Publish a completion and hand the descriptor back to the CPU."""
        slot, i = self._slot(index)
        if slot.owner is not Owner.DEVICE or not slot.device_held:
            raise OwnershipError(
                f"descriptor {i}: device cannot release "
                f"(owner={slot.owner.value}, device_held={slot.device_held})")
        if not isinstance(status, str) or status not in VALID_STATUSES:
            raise ValueError(
                f"descriptor {i}: status must be one of {VALID_STATUSES}, "
                f"got {status!r}")
        bytes_done = _as_int(bytes_done, "bytes_done")
        max_done = slot.length if slot.kind == "copy" else slot.rows * slot.bpr
        if not 0 <= bytes_done <= max_done:
            raise ValueError(
                f"descriptor {i}: bytes_done {bytes_done} outside "
                f"[0, {max_done}]")
        if status == "error" and not error:
            raise ValueError(
                f"descriptor {i}: status 'error' requires an error message")
        if status != "error" and error is not None:
            raise ValueError(
                f"descriptor {i}: error message only allowed with "
                f"status 'error'")
        slot.completion = (status, error, bytes_done)
        slot.completions += 1
        slot.device_held = False
        slot.owner = Owner.CPU

    # -- chain validation / introspection -----------------------------------

    def walk_chain(self, head) -> list[int]:
        """Validate a whole scatter/gather chain; return its slot indices.

        Raises :class:`ChainError` (carrying the offending ``index`` and a
        ``reason``) on a bad ``next`` pointer, a cycle, or a buffer range
        outside the pool.  Cycle detection is exact (a visited-set proof of
        a loop), not a search budget; a chain can never exceed ``capacity``
        entries without repeating a slot.
        """
        slot, head = self._slot(head)
        order: list[int] = []
        seen: set[int] = set()
        idx: int | None = head
        prev: int | None = None
        while idx is not None:
            if not isinstance(idx, int) or isinstance(idx, bool) \
                    or not 0 <= idx < self.capacity:
                raise ChainError(
                    f"descriptor {prev}: next={idx!r} is not a valid ring "
                    f"index", index=prev, reason="next-out-of-range")
            if idx in seen:
                raise ChainError(
                    f"descriptor {prev}: next={idx} forms a cycle",
                    index=prev, reason="cycle")
            s = self._slots[idx]
            end = s.offset + s.length
            if end > len(self.pool):
                raise ChainError(
                    f"descriptor {idx}: buffer range [{s.offset}, {end}) "
                    f"exceeds pool size {len(self.pool)}",
                    index=idx, reason="buffer-out-of-range")
            seen.add(idx)
            order.append(idx)
            prev = idx
            idx = s.next
        return order

    def completion_count(self, index) -> int:
        """Lifetime number of completions (any status) for a slot."""
        slot, _ = self._slot(index)
        return slot.completions

    def inspect(self, index) -> dict:
        """Out-of-band snapshot of internal slot state.

        Introspection for tests and debugging only; it is *not* part of the
        ownership protocol and bypasses all visibility rules.
        """
        slot, i = self._slot(index)
        return {
            "index": i,
            "owner": slot.owner.value,
            "cpu_held": slot.cpu_held,
            "device_held": slot.device_held,
            "generation": slot.generation,
            "offset": slot.offset,
            "length": slot.length,
            "next": slot.next,
            "completion": slot.completion,
            "completions": slot.completions,
        }


# ---------------------------------------------------------------------------
# Copy engine (synthesised device)
# ---------------------------------------------------------------------------

class CopyEngine:
    """A synchronous device model that copies chain bytes into an output.

    Chains are submitted by head index and processed in submission order by
    :meth:`run`.  Each descriptor is acquired, validated (buffer range and
    ``next`` pointer), copied from the ring pool into the engine output, and
    released back to the CPU with status ``ok``.  On the first bad
    descriptor the engine releases *that* descriptor with status ``error``,
    completes the chain as an error, and halts: the rest of the broken
    chain and all subsequently queued chains are left unconsumed until
    :meth:`reset_halt` is called.  Descriptors completed before the bad one
    keep their ``ok`` completions and copied bytes (documented partial
    completion semantics).
    """

    def __init__(self, ring: DmaRing) -> None:
        if not isinstance(ring, DmaRing):
            raise TypeError(
                f"ring: expected DmaRing, got {type(ring).__name__}")
        self.ring = ring
        self._output = bytearray()
        self._queue: deque[int] = deque()
        self.results: list[ChainResult] = []
        self.halted = False

    @property
    def output(self) -> bytes:
        """All bytes copied so far, in chain/descriptor order."""
        return bytes(self._output)

    @property
    def queued(self) -> int:
        """Number of submitted chains not yet processed."""
        return len(self._queue)

    def submit_chain(self, head) -> None:
        """Queue a chain by head index.  The head must be device-owned."""
        slot, head = self.ring._slot(head)
        if slot.owner is not Owner.DEVICE:
            raise OwnershipError(
                f"chain head {head}: not released to device "
                f"(owner={slot.owner.value})")
        self._queue.append(head)

    def run(self) -> None:
        """Process queued chains in order until the queue drains or a chain
        error halts the engine."""
        while self._queue and not self.halted:
            self._process_chain(self._queue.popleft())

    def reset_halt(self) -> None:
        """Clear the halt flag so queued chains can be processed again."""
        self.halted = False

    # -- internals ----------------------------------------------------------

    def _validate_descriptor(self, idx: int, snap: DescriptorSnapshot,
                             seen: set[int]) -> str | None:
        """Return an error string if the descriptor is bad, else ``None``."""
        end = snap.offset + snap.length
        if end > len(self.ring.pool):
            return (f"descriptor {idx}: buffer range "
                    f"[{snap.offset}, {end}) exceeds pool size "
                    f"{len(self.ring.pool)}")
        nxt = snap.next
        if nxt is not None:
            if not isinstance(nxt, int) or isinstance(nxt, bool) \
                    or not 0 <= nxt < self.ring.capacity:
                return (f"descriptor {idx}: next={nxt!r} is not a valid "
                        f"ring index")
            if nxt in seen:
                return f"descriptor {idx}: next={nxt} forms a cycle"
        return None

    def _process_chain(self, head: int) -> None:
        ring = self.ring
        seen: set[int] = set()
        idx: int | None = head
        done: list[int] = []
        copied = 0
        while idx is not None:
            try:
                snap = ring.device_acquire(idx)
            except OwnershipError as exc:
                # e.g. a descriptor in the chain was never cpu_release'd
                self._fail_chain(head, done, copied, idx,
                                 f"descriptor {idx}: {exc}")
                return
            seen.add(idx)
            problem = self._validate_descriptor(idx, snap, seen)
            if problem is not None:
                ring.device_release(idx, status="error", error=problem,
                                    bytes_done=0)
                done.append(idx)
                self._fail_chain(head, done, copied, idx, problem)
                return
            data = bytes(ring.pool[snap.offset: snap.offset + snap.length])
            self._output += data
            copied += len(data)
            ring.device_release(idx, status="ok", bytes_done=snap.length)
            done.append(idx)
            idx = snap.next
        self.results.append(ChainResult(
            head=head, status="ok", error=None, error_index=None,
            bytes_copied=copied, descriptors=tuple(done)))

    def _fail_chain(self, head: int, done: list[int], copied: int,
                    idx: int, problem: str) -> None:
        self.results.append(ChainResult(
            head=head, status="error", error=problem, error_index=idx,
            bytes_copied=copied, descriptors=tuple(done)))
        self.halted = True


# ---------------------------------------------------------------------------
# Blit engine (synthesised 2D device)
# ---------------------------------------------------------------------------

class BlitEngine:
    """A synchronous device model that blits 2D strided blocks in the pool.

    Shares the ring and its ownership handoff with :class:`CopyEngine` but
    consumes descriptors staged with :meth:`DmaRing.cpu_write_blit`.  For
    each descriptor the engine:

    1. validates *every* generated row address (source and destination,
       all rows) against the pool when the request is accepted; if any row
       is out of range the descriptor completes with an error and **no
       byte of it is written**,
    2. snapshots the source bytes (only the valid source rows are cached,
       not the whole pool), so overlapping source/destination — including
       self-overlapping transfers — has a deterministic result: the whole
       blit is interpreted against the source bytes as of acceptance,
    3. writes destination rows in ascending logical row order, columns
       ascending within a row; when destination rows overlap, later rows
       overwrite earlier ones.  Bytes outside the written rectangles keep
       their previous values.

    An empty rectangle (``rows == 0`` or ``bpr == 0``) is a successful
    zero transfer; ``bytes_done`` counts only the ``rows * bpr`` payload
    bytes, never row/stride padding.  Error and halt semantics (bad
    descriptor completes the chain as an error, later chains stay queued
    until :meth:`reset_halt`) mirror :class:`CopyEngine`.
    """

    def __init__(self, ring: DmaRing) -> None:
        if not isinstance(ring, DmaRing):
            raise TypeError(
                f"ring: expected DmaRing, got {type(ring).__name__}")
        self.ring = ring
        self._queue: deque[int] = deque()
        self.results: list[ChainResult] = []
        self.halted = False

    @property
    def queued(self) -> int:
        """Number of submitted chains not yet processed."""
        return len(self._queue)

    def submit_chain(self, head) -> None:
        """Queue a blit chain by head index.  The head must be device-owned."""
        slot, head = self.ring._slot(head)
        if slot.owner is not Owner.DEVICE:
            raise OwnershipError(
                f"chain head {head}: not released to device "
                f"(owner={slot.owner.value})")
        self._queue.append(head)

    def run(self) -> None:
        """Process queued chains in order until the queue drains or a chain
        error halts the engine."""
        while self._queue and not self.halted:
            self._process_chain(self._queue.popleft())

    def reset_halt(self) -> None:
        """Clear the halt flag so queued chains can be processed again."""
        self.halted = False

    # -- internals ----------------------------------------------------------

    def _validate_blit(self, idx: int, snap: BlitSnapshot,
                       seen: set[int]) -> str | None:
        """Return an error string if the blit descriptor is bad, else None.

        Every generated row address is checked up front so a rejected
        descriptor never writes a single byte.
        """
        if self.ring._slots[idx].kind != "blit":
            return f"descriptor {idx}: not a blit descriptor"
        pool_size = len(self.ring.pool)
        for row in range(snap.rows):
            for label, base, stride in (
                    ("src", snap.src, snap.src_stride),
                    ("dst", snap.dst, snap.dst_stride)):
                start = base + row * stride
                end = start + snap.bpr
                if not (0 <= start and end <= pool_size):
                    return (f"descriptor {idx}: {label} row {row} range "
                            f"[{start}, {end}) outside pool size {pool_size}")
        nxt = snap.next
        if nxt is not None:
            if not isinstance(nxt, int) or isinstance(nxt, bool) \
                    or not 0 <= nxt < self.ring.capacity:
                return (f"descriptor {idx}: next={nxt!r} is not a valid "
                        f"ring index")
            if nxt in seen:
                return f"descriptor {idx}: next={nxt} forms a cycle"
        return None

    def _process_chain(self, head: int) -> None:
        ring = self.ring
        seen: set[int] = set()
        idx: int | None = head
        done: list[int] = []
        moved = 0
        while idx is not None:
            try:
                ring.device_acquire(idx)
            except OwnershipError as exc:
                self._fail_chain(head, done, moved, idx,
                                 f"descriptor {idx}: {exc}")
                return
            seen.add(idx)
            snap = ring.device_read_blit(idx)
            problem = self._validate_blit(idx, snap, seen)
            if problem is not None:
                ring.device_release(idx, status="error", error=problem,
                                    bytes_done=0)
                done.append(idx)
                self._fail_chain(head, done, moved, idx, problem)
                return
            # Snapshot the valid source rows before any destination write,
            # so overlapping blits see the acceptance-time source bytes.
            src_rows = [bytes(ring.pool[snap.src + r * snap.src_stride:
                                        snap.src + r * snap.src_stride + snap.bpr])
                        for r in range(snap.rows)]
            for r in range(snap.rows):
                d = snap.dst + r * snap.dst_stride
                ring.pool[d:d + snap.bpr] = src_rows[r]
            moved += snap.rows * snap.bpr
            ring.device_release(idx, status="ok",
                                bytes_done=snap.rows * snap.bpr)
            done.append(idx)
            idx = snap.next
        self.results.append(ChainResult(
            head=head, status="ok", error=None, error_index=None,
            bytes_copied=moved, descriptors=tuple(done)))

    def _fail_chain(self, head: int, done: list[int], moved: int,
                    idx: int, problem: str) -> None:
        self.results.append(ChainResult(
            head=head, status="error", error=problem, error_index=idx,
            bytes_copied=moved, descriptors=tuple(done)))
        self.halted = True
