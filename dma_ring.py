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

Notification coalescing: an optional :class:`NotificationCoalescer` may be
attached to a ring to merge the per-descriptor completion notifications into
batched events (count threshold, wait timeout on the oldest unnotified
record, or explicit flush).  Coalescing only affects *notification*; the
publication of completion records and the ownership hand-back still happen
in ``device_release`` exactly as without a coalescer, so data visibility is
never delayed.

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
        self._coalescers: list = []  # attached NotificationCoalescers

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
        # The completion record is published and ownership returned above;
        # attached coalescers only batch the *notification* of this fact.
        for coalescer in self._coalescers:
            coalescer._on_completion(
                i, slot.generation, status, error, bytes_done)

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
# Notification coalescer
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CompletionRecord:
    """One completion as reported inside a coalesced notification.

    The record identity is ``(index, generation)``: when a ring slot wraps
    and is recycled, the new completion carries a fresh generation, so
    records published before and after the reuse stay distinguishable.

    ``time`` is the coalescer's logical clock value at the moment the
    device released the descriptor.
    """

    index: int
    generation: int
    status: str            # one of VALID_STATUSES
    error: str | None
    bytes_done: int
    time: int


@dataclass(frozen=True)
class Notification:
    """One coalesced notification event.

    ``records`` holds every completion since the previous notification, in
    the order the device actually completed them (any status, including
    ``"error"``).  ``reason`` is one of ``"threshold"``, ``"timeout"`` or
    ``"flush"``; ``time`` is the logical time the notification fired (the
    triggering record's arrival time for ``threshold``, the deadline for
    ``timeout``, the current time for ``flush``).
    """

    time: int
    reason: str
    records: tuple[CompletionRecord, ...]


class NotificationCoalescer:
    """Merges per-descriptor completion notifications into batched events.

    Attach to a :class:`DmaRing`; every successful ``device_release`` (any
    status, errors included) is recorded at the coalescer's current logical
    time.  A single :class:`Notification` is emitted when

    * the number of unnotified records reaches ``threshold`` (immediately,
      at the completing record's arrival time), or
    * the oldest unnotified record has waited ``timeout`` ticks, i.e. the
      logical clock reaches ``first_record_time + timeout`` — boundary
      equality triggers, checked by :meth:`advance`, or
    * the caller explicitly :meth:`flush`es.

    If several conditions would hold at the same instant only one
    notification is emitted: each record belongs to exactly one batch, so
    one trigger can never produce two events over the same records.

    Delivery: notifications fired implicitly inside ``device_release``
    (threshold reached) are queued and consumed with :meth:`read` /
    :meth:`drain`; notifications fired by the caller's own explicit
    triggers (:meth:`advance` timeout, :meth:`flush`) are returned
    directly by those calls and are not also queued, so every notification
    is delivered exactly once.  Reading is destructive: a consumed
    notification is never returned again.

    Publication of completion records and the ownership hand-back are
    unchanged: they happen in ``device_release``/``cpu_acquire`` exactly as
    without a coalescer.  Coalescing only delays the *notification*, never
    the visibility of the data.

    Logical time is driven solely by :meth:`advance`; nothing here sleeps,
    spawns threads, or reacts to interrupts.
    """

    REASONS = ("threshold", "timeout", "flush")

    def __init__(self, ring: DmaRing, *, threshold, timeout) -> None:
        if not isinstance(ring, DmaRing):
            raise TypeError(
                f"ring: expected DmaRing, got {type(ring).__name__}")
        self.ring = ring
        self.threshold = self._positive_int(threshold, "threshold")
        self.timeout = self._positive_int(timeout, "timeout")
        self._now = 0
        self._batch: list[CompletionRecord] = []
        self._deadline: int | None = None
        self._events: deque[Notification] = deque()
        ring._coalescers.append(self)

    @staticmethod
    def _positive_int(value, name: str) -> int:
        value = _as_int(value, name)
        if value < 1:
            raise ValueError(f"{name}: must be >= 1, got {value}")
        return value

    # -- introspection ------------------------------------------------------

    @property
    def now(self) -> int:
        """Current logical time."""
        return self._now

    @property
    def pending_count(self) -> int:
        """Records completed but not yet covered by any notification."""
        return len(self._batch)

    @property
    def next_deadline(self) -> int | None:
        """Timeout deadline of the current batch, or ``None`` if empty."""
        return self._deadline

    @property
    def events_pending(self) -> int:
        """Implicitly fired (threshold) notifications not yet consumed."""
        return len(self._events)

    # -- time and triggers ----------------------------------------------------

    def advance(self, ticks=1) -> list[Notification]:
        """Advance the logical clock; fire the timeout if it is reached.

        Boundary equality triggers: if the oldest unnotified record's
        deadline falls on or before the new time, exactly one ``timeout``
        notification is emitted (stamped at the deadline).  Returns the
        notifications fired by this call (empty or a single element); they
        are delivered to the caller directly, not queued.
        """
        ticks = _as_int(ticks, "ticks")
        if ticks < 0:
            raise ValueError(f"ticks: must be >= 0, got {ticks}")
        target = self._now + ticks
        fired: list[Notification] = []
        if self._batch and self._deadline <= target:
            fired.append(self._emit("timeout", self._deadline))
        self._now = target
        return fired

    def flush(self) -> Notification | None:
        """Emit a notification covering all pending records right now.

        Returns the emitted notification (delivered directly, not queued).
        An empty flush is a no-op and produces no event (returns ``None``).
        """
        if not self._batch:
            return None
        return self._emit("flush", self._now)

    # -- consuming notifications ----------------------------------------------

    def read(self) -> Notification | None:
        """Consume and return the oldest pending notification, or ``None``.

        Reading is destructive: a consumed notification is never returned
        again, and reading an empty queue simply returns ``None``.
        """
        if self._events:
            return self._events.popleft()
        return None

    def drain(self) -> list[Notification]:
        """Consume and return all pending notifications, oldest first."""
        out = list(self._events)
        self._events.clear()
        return out

    def detach(self) -> None:
        """Stop receiving completions from the ring (idempotent)."""
        try:
            self.ring._coalescers.remove(self)
        except ValueError:
            pass

    # -- internals --------------------------------------------------------------

    def _emit(self, reason: str, time: int) -> Notification:
        """Build the notification for the current batch and clear it.

        Does not deliver: the caller either queues the event (implicit
        threshold/timeout triggers) or returns it (explicit triggers).
        """
        note = Notification(time=time, reason=reason,
                            records=tuple(self._batch))
        self._batch.clear()
        self._deadline = None
        return note

    def _on_completion(self, index: int, generation: int, status: str,
                       error: str | None, bytes_done: int) -> None:
        # Called by DmaRing.device_release after the completion record has
        # been published and ownership returned to the CPU.
        if self._batch and self._now >= self._deadline:
            # The previous batch's timeout expired at this same instant;
            # emit it first so the new record starts a fresh batch and no
            # record is ever covered by two notifications.
            self._events.append(self._emit("timeout", self._deadline))
        self._batch.append(CompletionRecord(
            index, generation, status, error, bytes_done, self._now))
        if len(self._batch) == 1:
            self._deadline = self._now + self.timeout
        if len(self._batch) >= self.threshold:
            self._events.append(self._emit("threshold", self._now))
