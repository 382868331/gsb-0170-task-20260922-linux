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


@dataclass(frozen=True)
class BlitDescriptorSnapshot:
    """Immutable view of a committed 2D blit descriptor.

    Addresses are generated as ``start + row * stride`` for each logical
    row ``0 <= r < rows``; each row spans ``row_bytes`` bytes.  Strides may
    be negative.  ``rows == 0`` or ``row_bytes == 0`` is an empty
    rectangle: a successful zero transfer generating no addresses.
    """

    index: int
    generation: int
    src: int
    dst: int
    rows: int
    row_bytes: int
    src_stride: int
    dst_stride: int
    next: int | None


@dataclass(frozen=True)
class BlitChainResult:
    """Outcome of one chain processed by :class:`BlitEngine`."""

    head: int
    status: str            # "ok" (rejections happen at submit time)
    error: str | None
    error_index: int | None
    bytes_blitted: int
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
        "kind", "src", "dst", "rows", "row_bytes",
        "src_stride", "dst_stride",          # committed blit fields
        "p_kind", "p_src", "p_dst", "p_rows", "p_row_bytes",
        "p_src_stride", "p_dst_stride",      # pending blit fields
        "completion",                        # committed completion triple
        "completions",                       # lifetime completion count
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
        self.kind = "copy"
        self.src = 0
        self.dst = 0
        self.rows = 0
        self.row_bytes = 0
        self.src_stride = 0
        self.dst_stride = 0
        self.p_kind = "copy"
        self.p_src = 0
        self.p_dst = 0
        self.p_rows = 0
        self.p_row_bytes = 0
        self.p_src_stride = 0
        self.p_dst_stride = 0
        self.completion: tuple[str, str | None, int] | None = None
        self.completions = 0


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
        slot.p_kind = "copy"
        slot.p_offset = offset
        slot.p_length = length
        slot.p_next = next

    def cpu_write_blit(self, index, *, src, dst, rows, row_bytes,
                       src_stride, dst_stride, next=None) -> None:
        """Stage a 2D blit descriptor.  Requires CPU ownership.

        ``src``/``dst`` are the start addresses of the first source and
        destination rows in the pool; row ``r`` starts at
        ``start + r * stride`` and spans ``row_bytes`` bytes.  Strides may
        be negative.  ``rows`` and ``row_bytes`` must be >= 0; a zero of
        either is an empty rectangle (a successful zero transfer).

        As with :meth:`cpu_write`, all arguments are validated before any
        state changes, and generated-address bounds against the pool are
        *not* checked here: every generated address is validated by
        :class:`BlitEngine` when the request is submitted.
        """
        slot, i = self._slot(index)
        if slot.owner is not Owner.CPU or not slot.cpu_held:
            raise OwnershipError(
                f"descriptor {i}: CPU cannot write "
                f"(owner={slot.owner.value}, cpu_held={slot.cpu_held})")
        src = _as_int(src, "src")
        dst = _as_int(dst, "dst")
        rows = _as_int(rows, "rows")
        row_bytes = _as_int(row_bytes, "row_bytes")
        src_stride = _as_int(src_stride, "src_stride")
        dst_stride = _as_int(dst_stride, "dst_stride")
        if rows < 0:
            raise ValueError(f"descriptor {i}: rows {rows} < 0")
        if row_bytes < 0:
            raise ValueError(f"descriptor {i}: row_bytes {row_bytes} < 0")
        if next is not None:
            next = _as_int(next, "next")
        slot.p_kind = "blit"
        slot.p_src = src
        slot.p_dst = dst
        slot.p_rows = rows
        slot.p_row_bytes = row_bytes
        slot.p_src_stride = src_stride
        slot.p_dst_stride = dst_stride
        # Mirror into the generic offset/length fields so introspection and
        # the device_release bytes_done bound stay meaningful: length is the
        # total transferred bytes (row payload only, no stride padding).
        slot.p_offset = src
        slot.p_length = rows * row_bytes
        slot.p_next = next

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
        slot.row_bytes = slot.p_row_bytes
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

    def device_read_blit(self, index) -> BlitDescriptorSnapshot:
        """Read committed 2D blit fields.  Requires ``device_acquire``."""
        slot, i = self._slot(index)
        if not slot.device_held:
            raise VisibilityError(
                f"descriptor {i}: device must device_acquire before reading")
        return BlitDescriptorSnapshot(
            i, slot.generation, slot.src, slot.dst, slot.rows,
            slot.row_bytes, slot.src_stride, slot.dst_stride, slot.next)

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
        if not 0 <= bytes_done <= slot.length:
            raise ValueError(
                f"descriptor {i}: bytes_done {bytes_done} outside "
                f"[0, length={slot.length}]")
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
            "kind": slot.kind,
            "src": slot.src,
            "dst": slot.dst,
            "rows": slot.rows,
            "row_bytes": slot.row_bytes,
            "src_stride": slot.src_stride,
            "dst_stride": slot.dst_stride,
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

def _blit_range_ok(pool_size: int, start: int, rows: int, row_bytes: int,
                   stride: int) -> bool:
    """Check every generated row address of one rectangle.

    Row addresses form the arithmetic progression start + r * stride;
    the extremes are the two endpoints, so checking both covers every
    row.  An empty rectangle generates no addresses and is in range.
    """
    if rows == 0 or row_bytes == 0:
        return True
    first = start
    last = start + (rows - 1) * stride
    lo = min(first, last)
    hi = max(first, last)
    return 0 <= lo and hi + row_bytes <= pool_size


class _BlitRequest:
    """A validated, queued blit chain with its source-byte snapshot."""

    __slots__ = ("head", "entries")

    def __init__(self, head: int, entries: list) -> None:
        self.head = head
        # entries: (index, BlitDescriptorSnapshot, src_rows); src_rows[r]
        # is the immutable source bytes of logical row r at submit time.
        self.entries = entries


class BlitEngine:
    """Synchronous device model blitting 2D strided rectangles.

    Shares the ring byte pool with CopyEngine and reuses the same
    descriptor-ring ownership handoff; descriptors are staged with
    DmaRing.cpu_write_blit and committed with cpu_release.

    Request reception is submit_chain: it walks the whole chain,
    acquires each descriptor, and validates *every* generated row
    address (source and destination) against the pool.  If any row of
    any descriptor is out of bounds, nothing is written and nothing is
    queued: the offending descriptor is completed 'error', already-
    acquired predecessors are completed 'skipped', and ChainError is
    raised.  On success the engine snapshots exactly the source rows
    the chain will read (not the whole pool), so the transfer is
    interpreted against the source bytes as of the successful submit;
    overlapping source/destination therefore has a deterministic
    result.

    run() then performs the queued transfers: logical rows are written
    in ascending row order, columns ascending within a row, so when
    destination rows overlap each other the later write wins.  Pool
    bytes outside every destination row (stride padding included) keep
    their previous values.  bytes_done counts payload bytes only
    (rows * row_bytes), never stride padding.
    """

    def __init__(self, ring: DmaRing) -> None:
        if not isinstance(ring, DmaRing):
            raise TypeError(
                f"ring: expected DmaRing, got {type(ring).__name__}")
        self.ring = ring
        self._queue: deque[_BlitRequest] = deque()
        self.results: list[BlitChainResult] = []

    @property
    def queued(self) -> int:
        """Number of submitted chains not yet processed."""
        return len(self._queue)

    def submit_chain(self, head) -> None:
        """Validate and queue a blit chain, snapshotting source bytes.

        Raises OwnershipError if the head (or a chained descriptor) was
        never released to the device, and ChainError (carrying
        index/reason) on a bad next pointer, a cycle, a non-blit
        descriptor, or any generated row address outside the pool.  A
        rejected submit writes nothing to the pool and queues nothing.
        """
        slot, head = self.ring._slot(head)
        if slot.owner is not Owner.DEVICE:
            raise OwnershipError(
                f"chain head {head}: not released to device "
                f"(owner={slot.owner.value})")
        ring = self.ring
        pool_size = len(ring.pool)
        acquired: list[tuple[int, BlitDescriptorSnapshot]] = []
        seen: set[int] = set()
        idx: int | None = head
        prev: int | None = None
        problem: str | None = None
        problem_index: int | None = None
        reason: str | None = None
        while idx is not None:
            if not isinstance(idx, int) or isinstance(idx, bool) \
                    or not 0 <= idx < ring.capacity:
                problem = (f"descriptor {prev}: next={idx!r} is not a "
                           f"valid ring index")
                problem_index, reason = prev, "next-out-of-range"
                break
            if idx in seen:
                problem = f"descriptor {prev}: next={idx} forms a cycle"
                problem_index, reason = prev, "cycle"
                break
            try:
                ring.device_acquire(idx)
            except OwnershipError:
                for done_idx, _ in acquired:
                    ring.device_release(done_idx, status="skipped")
                raise
            snap = ring.device_read_blit(idx)
            if ring.inspect(idx)["kind"] != "blit":
                problem = f"descriptor {idx}: not a blit descriptor"
                problem_index, reason = idx, "wrong-kind"
                acquired.append((idx, snap))
                break
            if not _blit_range_ok(pool_size, snap.src, snap.rows,
                                  snap.row_bytes, snap.src_stride) or \
                    not _blit_range_ok(pool_size, snap.dst, snap.rows,
                                       snap.row_bytes, snap.dst_stride):
                problem = (f"descriptor {idx}: blit rectangle generates "
                           f"addresses outside pool size {pool_size}")
                problem_index, reason = idx, "buffer-out-of-range"
                acquired.append((idx, snap))
                break
            seen.add(idx)
            acquired.append((idx, snap))
            prev = idx
            idx = snap.next
        if problem is not None:
            for done_idx, _ in acquired:
                if done_idx == problem_index and reason in (
                        "wrong-kind", "buffer-out-of-range"):
                    ring.device_release(done_idx, status="error",
                                        error=problem, bytes_done=0)
                else:
                    ring.device_release(done_idx, status="skipped")
            raise ChainError(problem, index=problem_index, reason=reason)
        # Snapshot exactly the source rows, at submit time.
        entries = []
        for done_idx, snap in acquired:
            src_rows = [
                bytes(ring.pool[snap.src + r * snap.src_stride:
                                snap.src + r * snap.src_stride
                                + snap.row_bytes])
                for r in range(snap.rows)
            ]
            entries.append((done_idx, snap, src_rows))
        self._queue.append(_BlitRequest(head, entries))

    def run(self) -> None:
        """Process queued chains in submission order."""
        while self._queue:
            self._process_request(self._queue.popleft())

    # -- internals ----------------------------------------------------------

    def _process_request(self, request: _BlitRequest) -> None:
        ring = self.ring
        done: list[int] = []
        blitted = 0
        for idx, snap, src_rows in request.entries:
            for r in range(snap.rows):
                dst_addr = snap.dst + r * snap.dst_stride
                ring.pool[dst_addr:dst_addr + snap.row_bytes] = src_rows[r]
            payload = snap.rows * snap.row_bytes
            blitted += payload
            ring.device_release(idx, status="ok", bytes_done=payload)
            done.append(idx)
        self.results.append(BlitChainResult(
            head=request.head, status="ok", error=None, error_index=None,
            bytes_blitted=blitted, descriptors=tuple(done)))
