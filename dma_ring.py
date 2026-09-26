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


def periodic_segment_bytes(sequence, size) -> bytes:
    """Deterministic, monotonically increasing reference payload.

    Segment ``sequence`` of ``size`` bytes contains the byte values
    ``(sequence * size + j) & 0xFF`` for ``j`` in ``range(size)``.  This is
    the fixed content contract of the periodic capture model; an independent
    copy of the same formula is used by the tests as the oracle.
    """
    sequence = _as_int(sequence, "sequence")
    size = _as_int(size, "size")
    if sequence < 0:
        raise ValueError(f"sequence: must be >= 0, got {sequence}")
    if size < 0:
        raise ValueError(f"size: must be >= 0, got {size}")
    base = sequence * size
    return bytes((base + j) & 0xFF for j in range(size))


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
# Periodic capture ring
# ---------------------------------------------------------------------------

class OverrunError(DmaRingError):
    """Software lagged so far that the lost sequence range became ambiguous.

    Raised only when a finite counter_bits is configured and the hardware
    has advanced a full counter modulus past the unclaimed cursor, so the
    exact lost interval can no longer be derived from the counter.
    """


@dataclass(frozen=True)
class SegmentInfo:
    """One completed periodic segment handed to the software consumer.

    sequence is the infinite (never reused) segment number; lap is the
    1-based completion lap; generation counts how many times this slot has
    been completed; data is an immutable copy of the slot buffer taken at
    claim time.
    """

    sequence: int
    lap: int
    slot: int
    generation: int
    data: bytes


@dataclass(frozen=True)
class OverrunReport:
    """Explicit record of segments overwritten before the software claimed
    them.

    lost_start..lost_end is the inclusive range of infinite sequences that
    were overwritten; every sequence in the range is gone and can never be
    claimed.
    """

    lost_start: int
    lost_end: int

    @property
    def lost_count(self) -> int:
        return self.lost_end - self.lost_start + 1


@dataclass(frozen=True)
class PartialSegment:
    """State of the segment the hardware was writing when capture stopped."""

    sequence: int
    slot: int
    bytes_written: int
    segment_size: int


@dataclass(frozen=True)
class ClaimResult:
    """Outcome of one claim: the segments delivered and the losses reported."""

    segments: tuple[SegmentInfo, ...]
    lost: tuple[OverrunReport, ...]

    @property
    def lost_count(self) -> int:
        return sum(report.lost_count for report in self.lost)


class PeriodicRing:
    """Periodic capture: hardware loops over fixed buffer segments.

    The hardware writes segment sequence into slot sequence % capacity;
    each slot owns a fixed segment_size byte buffer. Completed segments
    carry a completion generation (per-slot completion count) and are
    claimed by the software in sequence order via claim(). If the software
    falls more than one lap behind, the hardware overwrites the oldest
    unclaimed segment; the overwrite is recorded explicitly and reported
    as an OverrunReport naming the exact lost sequence range. Overwritten
    data is never returned.

    Ordering rule: the model is synchronous and single-threaded, so a
    claim and a hardware completion never happen in the same instant.
    When a claim and a completion are adjacent, the hardware step runs
    first and the claim observes the new completion
    (see hw_step_and_claim()).

    Parameters:
        capacity:     number of segments (slots); a power of two in
                      [1, MAX_CAPACITY].
        segment_size: bytes per segment, >= 1.
        counter_bits: optional hardware sequence-counter width. With
                      None (default) the model uses an infinite counter
                      and every loss is reported exactly. With a finite
                      width the visible counter wraps modulo
                      2 ** counter_bits; if the software lags by a full
                      modulus the loss range becomes ambiguous and
                      OverrunError is raised instead of guessing.
    """

    def __init__(self, capacity, segment_size, *, counter_bits=None) -> None:
        self.capacity = _check_capacity(capacity)
        segment_size = _as_int(segment_size, "segment_size")
        if segment_size < 1:
            raise ValueError(
                f"segment_size: must be >= 1, got {segment_size}")
        self.segment_size = segment_size
        if counter_bits is None:
            self.counter_bits = None
            self._modulus = None
        else:
            counter_bits = _as_int(counter_bits, "counter_bits")
            if not 1 <= counter_bits <= 63:
                raise ValueError(
                    f"counter_bits: must be in [1, 63], got {counter_bits}")
            self.counter_bits = counter_bits
            self._modulus = 1 << counter_bits
        # Fixed per-slot buffers; the hardware always writes in place.
        self.buffers = [bytearray(segment_size) for _ in range(self.capacity)]
        self._running = False
        self._next_sequence = 0          # sequence of the segment in progress
        self._bytes_in_current = 0       # bytes written into that segment
        self._claimed_cursor = 0         # next sequence software may claim
        self._completed = {}            # sequence -> (lap, generation)
        self._generations = [0] * self.capacity
        self._pending_lost = []
        self.total_completed = 0
        self.total_lost = 0

    # -- lifecycle ---------------------------------------------------------

    @property
    def running(self) -> bool:
        return self._running

    def start(self) -> None:
        """(Re)start capture. Retains all previously completed segments.

        The software claim cursor is untouched: segments completed before
        a stop remain claimable after the restart, and infinite sequence
        numbering continues where it left off. A segment half written at
        stop time is discarded; its sequence is rewritten from byte 0.
        """
        if self._running:
            raise OwnershipError("periodic ring is already running")
        self._bytes_in_current = 0
        self._running = True

    def stop(self) -> PartialSegment | None:
        """Stop capture. Completed segments stay claimable.

        Returns the half-written segment state (sequence, slot, byte
        count) if the hardware was mid-segment, else None. The partial
        bytes are never delivered as a completed segment.
        """
        if not self._running:
            raise OwnershipError("periodic ring is not running")
        self._running = False
        if self._bytes_in_current == 0:
            return None
        partial = PartialSegment(
            sequence=self._next_sequence,
            slot=self._next_sequence % self.capacity,
            bytes_written=self._bytes_in_current,
            segment_size=self.segment_size,
        )
        self._bytes_in_current = 0
        return partial

    # -- hardware side -----------------------------------------------------

    def _check_overrun_ambiguity(self) -> None:
        if self._modulus is not None \
                and self._next_sequence - self._claimed_cursor >= self._modulus:
            raise OverrunError(
                f"software lag {self._next_sequence - self._claimed_cursor} "
                f"reached the {self.counter_bits}-bit counter modulus "
                f"{self._modulus}: the lost sequence range is ambiguous; "
                "claim more often or widen the counter")

    def _note_overwrite(self, sequence: int) -> None:
        self._completed.pop(sequence, None)
        self.total_lost += 1
        if self._pending_lost \
                and self._pending_lost[-1].lost_end == sequence - 1:
            last = self._pending_lost[-1]
            self._pending_lost[-1] = OverrunReport(last.lost_start, sequence)
        else:
            self._pending_lost.append(OverrunReport(sequence, sequence))

    def _write_bytes(self, data: bytes) -> None:
        for byte in data:
            if self._bytes_in_current == 0:
                self._check_overrun_ambiguity()
                # Starting a segment reuses the slot of the segment one
                # lap earlier; if that older segment is still unclaimed
                # its data is overwritten now and is lost forever.
                victim = self._next_sequence - self.capacity
                if victim in self._completed:
                    self._note_overwrite(victim)
            slot = self._next_sequence % self.capacity
            self.buffers[slot][self._bytes_in_current] = byte
            self._bytes_in_current += 1
            if self._bytes_in_current == self.segment_size:
                self._finish_segment()

    def _finish_segment(self) -> None:
        sequence = self._next_sequence
        slot = sequence % self.capacity
        self._generations[slot] += 1
        lap = sequence // self.capacity + 1
        self._completed[sequence] = (lap, self._generations[slot])
        self.total_completed += 1
        self._next_sequence = sequence + 1
        self._bytes_in_current = 0

    def hw_write_bytes(self, data) -> None:
        """Hardware writes data (bytes-like) into the current segment."""
        if not self._running:
            raise OwnershipError("periodic ring is not running")
        self._write_bytes(bytes(data))

    def hw_advance(self, segments=1) -> None:
        """Hardware completes segments whole segments in one call."""
        if not self._running:
            raise OwnershipError("periodic ring is not running")
        segments = _as_int(segments, "segments")
        if segments < 0:
            raise ValueError(f"segments: must be >= 0, got {segments}")
        for _ in range(segments):
            if self._bytes_in_current != 0:
                raise OwnershipError(
                    "hw_advance requires segment-aligned state; "
                    "finish or stop the partial segment first")
            self._write_bytes(periodic_segment_bytes(
                self._next_sequence, self.segment_size))

    def hw_step(self) -> None:
        """Hardware completes exactly one segment (aligned fast path)."""
        self.hw_advance(1)

    # -- software side -----------------------------------------------------

    def _deliver(self, sequence: int) -> SegmentInfo:
        lap, generation = self._completed.pop(sequence)
        slot = sequence % self.capacity
        return SegmentInfo(
            sequence=sequence, lap=lap, slot=slot, generation=generation,
            data=bytes(self.buffers[slot]))

    def claim(self, max_segments=None) -> ClaimResult:
        """Claim the next run of consecutive completed segments.

        Returns up to max_segments (default: all available) segments in
        sequence order, starting at the software claim cursor. Segments
        the hardware is still writing are never delivered. Sequences that
        were overwritten before this claim are reported in lost and
        skipped; their (stale, overwritten) bytes are never returned. Each
        reported loss is consumed by exactly one claim.
        """
        if max_segments is None:
            limit = None
        else:
            limit = _as_int(max_segments, "max_segments")
            if limit < 0:
                raise ValueError(
                    f"max_segments: must be >= 0, got {limit}")
        lost = tuple(
            report for report in self._pending_lost
            if report.lost_end >= self._claimed_cursor)
        self._pending_lost = [
            report for report in self._pending_lost
            if report.lost_end < self._claimed_cursor]
        out = []
        while limit is None or len(out) < limit:
            sequence = self._claimed_cursor
            if any(report.lost_start <= sequence <= report.lost_end
                   for report in lost):
                self._claimed_cursor = sequence + 1
                continue
            if sequence not in self._completed:
                break
            out.append(self._deliver(sequence))
            self._claimed_cursor = sequence + 1
        return ClaimResult(segments=tuple(out), lost=lost)

    def hw_step_and_claim(self, max_segments=None) -> ClaimResult:
        """One hardware completion immediately followed by one claim.

        Defines the ordering when a completion and a claim are adjacent:
        the hardware completes first, then the claim runs and therefore
        observes (and may deliver) the just-completed segment.
        """
        self.hw_step()
        return self.claim(max_segments)

    # -- introspection -----------------------------------------------------

    def hw_counter(self) -> int:
        """The write position visible on the hardware sequence counter.

        With finite counter_bits this is next_sequence modulo the counter
        modulus (it wraps); otherwise it is the infinite value.
        """
        if self._modulus is None:
            return self._next_sequence
        return self._next_sequence % self._modulus

    @property
    def next_sequence(self) -> int:
        """Infinite sequence number of the segment being written."""
        return self._next_sequence

    @property
    def claim_cursor(self) -> int:
        """Next sequence number the software may claim."""
        return self._claimed_cursor

    def pending_completed(self) -> list[int]:
        """Sequences completed by hardware but not yet claimed, sorted."""
        return sorted(self._completed)
