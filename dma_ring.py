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
engine halts without consuming subsequent queued chains.  Queued chains
that never started can be cancelled; a running chain can be asked to
stop, and the not-yet-executed descriptors are reclaimed only after the
device confirms it no longer accesses them (see :class:`CopyEngine`).

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


class ChainStateError(DmaRingError):
    """A chain-control operation was attempted in the wrong state.

    Raised when cancelling a chain that is not queued (already started or
    unknown), or when requesting a stop for a chain that is not currently
    executing.  Stale or duplicate *notifications* (stop confirmations)
    are not errors: they are rejected by a False return instead.
    """


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

    ``status`` is one of "ok", "error", "cancelled" or "stopped".
    ``descriptors`` lists the slots that were completed (ok or error) in
    processing order; ``skipped`` lists the slots that were handed back
    to the CPU without being executed (cancel/stop).  The final state of a
    chain therefore aggregates both the completed and the never-executed
    parts.  Slots after a bad descriptor are *not* consumed and do not
    appear in ``descriptors``.
    """

    head: int
    status: str            # "ok", "error", "cancelled" or "stopped"
    error: str | None
    error_index: int | None
    bytes_copied: int
    descriptors: tuple[int, ...]
    skipped: tuple[int, ...] = ()


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

class _ChainRecord:
    """Internal per-submission chain state tracked by :class:`CopyEngine`."""

    __slots__ = ("token", "head", "idx", "seen", "done", "copied")

    def __init__(self, *, token: int, head: int) -> None:
        self.token = token
        self.head = head
        self.idx: int | None = head   # next descriptor to process
        self.seen: set[int] = set()
        self.done: list[int] = []
        self.copied = 0

class CopyEngine:
    """A synchronous device model that copies chain bytes into an output.

    Chains are submitted by head index and processed in submission order.
    :meth:`run` processes everything to completion; :meth:`step` processes
    exactly one descriptor, which lets a caller observe (and stop) a chain
    while it is executing.  Each descriptor is acquired, validated (buffer
    range and ``next`` pointer), copied from the ring pool into the engine
    output, and released back to the CPU with status ``ok``.  On the first
    bad descriptor the engine releases *that* descriptor with status
    ``error``, completes the chain as an error, and halts: the rest of the
    broken chain and all subsequently queued chains are left unconsumed
    until :meth:`reset_halt` is called.  Descriptors completed before the
    bad one keep their ``ok`` completions and copied bytes (documented
    partial completion semantics).

    Chain control:

    * :meth:`cancel_chain` revokes a chain that is still queued and was
      never started.  The device never touched its descriptors, so they
      are handed back to the CPU immediately with status ``skipped``.
    * :meth:`request_stop` asks to stop the chain that is currently
      executing.  The parts already handed to the device cannot be
      reclaimed right away: the not-yet-executed descriptors stay
      device-owned until the device confirms it no longer accesses them
      via :meth:`confirm_stop`.  Only then are they released back to
      the CPU with status ``skipped`` and the chain finalised with
      status ``stopped``.
    * Every submitted chain carries a unique token (returned by
      :meth:`submit_chain` and :meth:`request_stop`).  A stop
      confirmation must name the token; stale confirmations (slot/head
      already reused by a newer chain) and duplicate confirmations are
      rejected with a ``False`` return and release nothing twice.
    """

    def __init__(self, ring: DmaRing) -> None:
        if not isinstance(ring, DmaRing):
            raise TypeError(
                f"ring: expected DmaRing, got {type(ring).__name__}")
        self.ring = ring
        self._output = bytearray()
        self._queue: deque[_ChainRecord] = deque()
        self._active: _ChainRecord | None = None
        self._stopping: list[_ChainRecord] = []
        self._next_token = 0
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

    @property
    def active_head(self) -> int | None:
        """Head index of the chain currently executing, or ``None``."""
        return self._active.head if self._active is not None else None

    @property
    def stopping_heads(self) -> tuple[int, ...]:
        """Heads of chains awaiting a stop confirmation."""
        return tuple(record.head for record in self._stopping)

    def submit_chain(self, head) -> int:
        """Queue a chain by head index.  The head must be device-owned.

        Returns the chain token, which uniquely identifies this submission
        and is required to confirm a later stop request.
        """
        slot, head = self.ring._slot(head)
        if slot.owner is not Owner.DEVICE:
            raise OwnershipError(
                f"chain head {head}: not released to device "
                f"(owner={slot.owner.value})")
        self._next_token += 1
        token = self._next_token
        self._queue.append(_ChainRecord(token=token, head=head))
        return token

    def run(self) -> None:
        """Process queued chains in order until the queue drains or a chain
        error halts the engine.

        Chains with a pending stop request are not finished here; they wait
        for :meth:`confirm_stop` and do not block later queued chains.
        """
        while self.step():
            pass

    def step(self) -> bool:
        """Process exactly one descriptor of the active/next queued chain.

        Returns ``True`` if a descriptor was processed (or a chain
        finalised), ``False`` when halted or when no queued chain remains.
        """
        if self.halted:
            return False
        if self._active is None:
            if not self._queue:
                return False
            self._active = self._queue.popleft()
        self._process_one(self._active)
        return True

    def reset_halt(self) -> None:
        """Clear the halt flag so queued chains can be processed again."""
        self.halted = False

    # -- chain control ------------------------------------------------------

    def cancel_chain(self, head) -> ChainResult:
        """Revoke a queued chain that was never started.

        Every descriptor of the chain is handed back to the CPU with
        status ``skipped`` (the device never acquired any of them, so
        immediate reclaim is safe) and a ``cancelled`` :class:`ChainResult`
        is recorded and returned.

        Raises :class:`ChainStateError` if no queued chain with this head
        exists (already started, already cancelled, or never submitted).
        """
        _, head = self.ring._slot(head)
        for record in self._queue:
            if record.head == head:
                self._queue.remove(record)
                skipped = self._reclaim(record.idx, record.seen)
                result = ChainResult(
                    head=head, status="cancelled", error=None,
                    error_index=None, bytes_copied=0, descriptors=(),
                    skipped=tuple(skipped))
                self.results.append(result)
                return result
        raise ChainStateError(
            f"chain head {head}: cannot cancel; not queued "
            f"(already started, already cancelled, or never submitted)")

    def request_stop(self, head) -> int:
        """Ask to stop the chain that is currently executing.

        The already-completed prefix keeps its ``ok`` completions.  The
        not-yet-executed descriptors stay device-owned — they were handed
        to the device and cannot be reclaimed until the device confirms it
        no longer accesses them via :meth:`confirm_stop`.  Returns the
        chain token needed for that confirmation.

        Raises :class:`ChainStateError` if this head is not the currently
        executing chain (use :meth:`cancel_chain` for queued chains).
        """
        _, head = self.ring._slot(head)
        record = self._active
        if record is None or record.head != head:
            raise ChainStateError(
                f"chain head {head}: cannot stop; not currently executing")
        self._active = None
        self._stopping.append(record)
        return record.token

    def confirm_stop(self, head, token) -> bool:
        """Device confirmation that it no longer accesses a stopped chain.

        On acceptance the not-yet-executed descriptors are released back
        to the CPU with status ``skipped``, a ``stopped``
        :class:`ChainResult`` aggregating the completed and skipped parts
        is recorded, and ``True`` is returned.

        Stale confirmations (the token belongs to an old chain whose
        head/slots were reused by a newer chain) and duplicate
        confirmations return ``False`` and change nothing: resources are
        never released twice and new chains are never affected by old
        notifications.
        """
        _, head = self.ring._slot(head)
        token = _as_int(token, "token")
        for record in self._stopping:
            if record.head == head and record.token == token:
                self._stopping.remove(record)
                skipped = self._reclaim(record.idx, record.seen)
                self.results.append(ChainResult(
                    head=record.head, status="stopped", error=None,
                    error_index=None, bytes_copied=record.copied,
                    descriptors=tuple(record.done),
                    skipped=tuple(skipped)))
                return True
        return False

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

    def _process_one(self, record: _ChainRecord) -> None:
        """Process the next descriptor of the active chain record."""
        ring = self.ring
        idx = record.idx
        try:
            snap = ring.device_acquire(idx)
        except OwnershipError as exc:
            # e.g. a descriptor in the chain was never cpu_release'd
            self._fail_chain(record, idx, f"descriptor {idx}: {exc}")
            return
        record.seen.add(idx)
        problem = self._validate_descriptor(idx, snap, record.seen)
        if problem is not None:
            ring.device_release(idx, status="error", error=problem,
                                bytes_done=0)
            record.done.append(idx)
            self._fail_chain(record, idx, problem)
            return
        data = bytes(ring.pool[snap.offset: snap.offset + snap.length])
        self._output += data
        record.copied += len(data)
        ring.device_release(idx, status="ok", bytes_done=snap.length)
        record.done.append(idx)
        record.idx = snap.next
        if record.idx is None:
            self.results.append(ChainResult(
                head=record.head, status="ok", error=None, error_index=None,
                bytes_copied=record.copied, descriptors=tuple(record.done)))
            self._active = None

    def _fail_chain(self, record: _ChainRecord, idx: int,
                    problem: str) -> None:
        self.results.append(ChainResult(
            head=record.head, status="error", error=problem, error_index=idx,
            bytes_copied=record.copied, descriptors=tuple(record.done)))
        self.halted = True
        self._active = None

    def _walk_from(self, start: int | None, seen: set[int]) -> list[int]:
        """Defensively walk committed ``next`` pointers from ``start``.

        Used only for cancel/stop reclaim, not for chain validation: the
        walk stops silently at an out-of-range index or a cycle instead of
        raising, so a malformed chain can still be torn down.
        """
        order: list[int] = []
        visited = set(seen)
        idx = start
        while idx is not None:
            if not isinstance(idx, int) or isinstance(idx, bool) \
                    or not 0 <= idx < self.ring.capacity:
                break
            if idx in visited:
                break
            visited.add(idx)
            order.append(idx)
            idx = self.ring._slots[idx].next
        return order

    def _reclaim(self, start: int | None, seen: set[int]) -> list[int]:
        """Hand not-yet-executed chain descriptors back to the CPU.

        Every device-owned descriptor reachable from ``start`` is
        acquired and released with status ``skipped`` (the device
        confirms it no longer accesses them).  Returns the walked indices
        in chain order, whether or not they needed reclaiming.
        """
        order = self._walk_from(start, seen)
        for idx in order:
            slot = self.ring._slots[idx]
            if slot.owner is Owner.DEVICE and not slot.device_held:
                self.ring.device_acquire(idx)
                self.ring.device_release(idx, status="skipped", bytes_done=0)
        return order
