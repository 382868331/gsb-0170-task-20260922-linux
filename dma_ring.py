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

Two device models share the ring protocol:

* ``CopyEngine`` — the original synchronous model: ``submit_chain`` +
  ``run()`` processes whole chains to completion, halting on the first
  descriptor error.
* ``SteppedCopyEngine`` — a step-driven, cancellable model.  ``submit``
  validates and freezes a chain's slots and generations and returns a
  ``CopyRequest`` handle; each ``step()`` processes at most one descriptor
  (or confirms one pending cancel).  ``CopyRequest.cancel()`` only
  *registers* intent: the device confirms it on the next step, ending the
  request and handing the not-yet-processed descriptors back to the CPU
  through the normal release/acquire rules (status ``skipped``).  The
  already-copied prefix is kept; the terminal ``CopyRequestResult``
  distinguishes ``ok`` / ``error`` / ``cancelled`` and reports the actual
  copied byte count.  All events are serial; no threads are involved.

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


class SlotBusyError(DmaRingError):
    """A submit tried to occupy a slot held by another in-flight request.

    Attributes:
        index:      ring index of the contended slot.
        request_id: id of the in-flight request currently occupying it.
    """

    def __init__(self, message: str, *, index: int | None = None,
                 request_id: int | None = None) -> None:
        super().__init__(message)
        self.index = index
        self.request_id = request_id


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
class CopyRequestResult:
    """Terminal outcome of one :class:`SteppedCopyEngine` request.

    ``status`` is one of:

    * ``"ok"``        — the whole chain was copied;
    * ``"error"``     — a descriptor failed device-side validation;
    * ``"cancelled"`` — a registered cancel was confirmed by a step.

    ``bytes_copied`` counts only bytes actually copied into the engine
    output before the request ended; unprocessed descriptors contribute
    nothing and are never counted as success.  ``descriptors`` is the
    completed prefix (``ok`` completions, plus the failing descriptor for
    ``status == "error"``); ``skipped`` lists the slots that were handed
    back to the CPU unprocessed (completion status ``skipped``).
    """

    request_id: int
    head: int
    status: str            # "ok", "error" or "cancelled"
    error: str | None
    error_index: int | None
    bytes_copied: int
    descriptors: tuple[int, ...]
    skipped: tuple[int, ...]


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

def _validate_copy_descriptor(ring: DmaRing, idx: int,
                              snap: DescriptorSnapshot,
                              seen: set[int]) -> str | None:
    """Device-side descriptor check shared by both copy engines.

    Returns an error string if the descriptor is bad (buffer range outside
    the pool, invalid ``next`` pointer, or a cycle against ``seen``), else
    ``None``.  ``seen`` must already contain the descriptors acquired
    earlier in the same chain, including ``idx``.
    """
    end = snap.offset + snap.length
    if end > len(ring.pool):
        return (f"descriptor {idx}: buffer range "
                f"[{snap.offset}, {end}) exceeds pool size "
                f"{len(ring.pool)}")
    nxt = snap.next
    if nxt is not None:
        if not isinstance(nxt, int) or isinstance(nxt, bool) \
                or not 0 <= nxt < ring.capacity:
            return (f"descriptor {idx}: next={nxt!r} is not a valid "
                    f"ring index")
        if nxt in seen:
            return f"descriptor {idx}: next={nxt} forms a cycle"
    return None


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
        return _validate_copy_descriptor(self.ring, idx, snap, seen)

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
# Stepped copy engine (cancellable, one descriptor per step)
# ---------------------------------------------------------------------------

class CopyRequest:
    """Handle for one stepped copy request returned by
    :meth:`SteppedCopyEngine.submit`.

    The handle stays meaningful after the request ends — and after its
    slots are recycled by a later generation: :meth:`result` always reports
    *this* request's own terminal state, while :meth:`cancel` can only
    register intent on *this* request and is a no-op once it has finished,
    so a stale handle can never cancel a newer request reusing the slots.
    """

    __slots__ = (
        "request_id", "head",
        "_chain", "_generations", "_position", "_copied",
        "_descriptors", "_seen",
        "_cancel_requested", "_done", "_result",
    )

    def __init__(self, request_id: int, head: int, chain: tuple[int, ...],
                 generations: tuple[int, ...]) -> None:
        self.request_id = request_id
        self.head = head
        self._chain = chain
        self._generations = generations
        self._position = 0
        self._copied = 0
        self._descriptors: list[int] = []
        self._seen: set[int] = set()
        self._cancel_requested = False
        self._done = False
        self._result: CopyRequestResult | None = None

    def __repr__(self) -> str:
        state = "done" if self._done else "in-flight"
        return (f"CopyRequest(request_id={self.request_id}, "
                f"head={self.head}, {state})")

    @property
    def done(self) -> bool:
        """Whether the request has reached its terminal state."""
        return self._done

    @property
    def cancel_requested(self) -> bool:
        """Whether cancel intent has been registered (not yet confirmed)."""
        return self._cancel_requested

    def cancel(self) -> bool:
        """Register cancel intent; the device confirms it on the next step.

        Returns ``True`` if intent was newly registered.  Returns ``False``
        (changing nothing) when intent was already registered or the
        request has already finished — completion is final and cannot be
        rewritten by a later cancel.
        """
        if self._done or self._cancel_requested:
            return False
        self._cancel_requested = True
        return True

    def result(self) -> CopyRequestResult | None:
        """The terminal result, or ``None`` while the request is in flight.

        Repeated reads return the same object and change nothing.
        """
        return self._result


class SteppedCopyEngine:
    """A step-driven, cancellable device model sharing the ring protocol.

    Unlike :class:`CopyEngine` (which processes whole chains inside one
    ``run()`` call and halts on error), this engine exposes the device as a
    serial sequence of single-descriptor steps so a caller can cancel an
    in-flight request:

    * :meth:`submit` validates the chain's structure, requires every slot
      to be released to the device, freezes each slot's generation, and
      returns a :class:`CopyRequest`.  Two in-flight requests may not
      occupy the same slot (:class:`SlotBusyError`).
    * :meth:`step` processes at most one event: it confirms the oldest
      in-flight request's pending cancel, or copies that request's next
      descriptor.  Events are serial; there are no threads.
    * A registered cancel is confirmed by the *next* step: the request
      ends as ``cancelled`` and its not-yet-processed descriptors are
      handed back to the CPU through the normal release/acquire rules with
      completion status ``skipped``.  The already-copied prefix is kept.
    * If the last descriptor completes first, the request is ``ok`` and a
      later :meth:`CopyRequest.cancel` changes nothing.
    * A descriptor error ends only that request (status ``error``),
      releasing its remaining slots as ``skipped``; disjoint in-flight
      requests are unaffected, and this engine has no halt flag.

    Every request publishes exactly one terminal :class:`CopyRequestResult`
    (appended to :attr:`results`); repeated cancels and repeated result
    reads never add completions.
    """

    def __init__(self, ring: DmaRing) -> None:
        if not isinstance(ring, DmaRing):
            raise TypeError(
                f"ring: expected DmaRing, got {type(ring).__name__}")
        self.ring = ring
        self._output = bytearray()
        self._requests: deque[CopyRequest] = deque()
        self._occupied: dict[int, CopyRequest] = {}
        self._next_request_id = 1
        self.results: list[CopyRequestResult] = []

    @property
    def output(self) -> bytes:
        """All bytes copied so far, in request/descriptor order."""
        return bytes(self._output)

    @property
    def active(self) -> int:
        """Number of in-flight (not yet finished) requests."""
        return len(self._requests)

    def submit(self, head) -> CopyRequest:
        """Validate and freeze a chain, returning its request handle.

        The whole chain is walked from ``head`` following committed ``next``
        pointers: every slot must be released to the device and not already
        occupied by another in-flight request.  On success each slot's
        current generation is frozen into the request; later steps acquire
        the slots with those exact generations, so slot reuse while the
        request is in flight is impossible.  All validation happens before
        any state changes, so a rejected submit leaves the engine
        untouched.  Buffer-range checks remain device-side (per step), as
        with :class:`CopyEngine`.
        """
        _, head = self.ring._slot(head)
        chain: list[int] = []
        seen: set[int] = set()
        idx: int | None = head
        prev: int | None = None
        while idx is not None:
            if not isinstance(idx, int) or isinstance(idx, bool) \
                    or not 0 <= idx < self.ring.capacity:
                raise ChainError(
                    f"descriptor {prev}: next={idx!r} is not a valid ring "
                    f"index", index=prev, reason="next-out-of-range")
            if idx in seen:
                raise ChainError(
                    f"descriptor {prev}: next={idx} forms a cycle",
                    index=prev, reason="cycle")
            slot = self.ring._slots[idx]
            if slot.owner is not Owner.DEVICE:
                raise OwnershipError(
                    f"descriptor {idx}: not released to device "
                    f"(owner={slot.owner.value})")
            holder = self._occupied.get(idx)
            if holder is not None:
                raise SlotBusyError(
                    f"descriptor {idx}: already occupied by in-flight "
                    f"request {holder.request_id}",
                    index=idx, request_id=holder.request_id)
            seen.add(idx)
            chain.append(idx)
            prev = idx
            idx = slot.next
        generations = tuple(self.ring._slots[i].generation for i in chain)
        request = CopyRequest(self._next_request_id, head,
                              tuple(chain), generations)
        self._next_request_id += 1
        for i in chain:
            self._occupied[i] = request
        self._requests.append(request)
        return request

    def step(self) -> bool:
        """Process one event for the oldest in-flight request.

        Exactly one of: confirm that request's pending cancel (ending it
        and releasing its remaining descriptors), or copy/complete its next
        descriptor.  Returns ``True`` if an event was processed, ``False``
        if no request is in flight.
        """
        if not self._requests:
            return False
        request = self._requests[0]
        if request._cancel_requested:
            skipped = self._release_remaining(request, request._position)
            self._finish(request, "cancelled", skipped=skipped)
            return True
        pos = request._position
        idx = request._chain[pos]
        generation = request._generations[pos]
        try:
            snap = self.ring.device_acquire(idx, expected_generation=generation)
        except DmaRingError as exc:
            # Defensive: submit() froze ownership and generations, so this
            # cannot happen while the invariants hold.  If it ever does,
            # end only this request and release what we still hold.
            problem = f"descriptor {idx}: {exc}"
            skipped = self._release_remaining(request, pos + 1)
            request._descriptors.append(idx)
            self._finish(request, "error", error=problem, error_index=idx,
                         skipped=skipped)
            return True
        request._seen.add(idx)
        problem = _validate_copy_descriptor(self.ring, idx, snap,
                                            request._seen)
        if problem is not None:
            self.ring.device_release(idx, status="error", error=problem,
                                     bytes_done=0)
            request._descriptors.append(idx)
            skipped = self._release_remaining(request, pos + 1)
            self._finish(request, "error", error=problem, error_index=idx,
                         skipped=skipped)
            return True
        data = bytes(self.ring.pool[snap.offset: snap.offset + snap.length])
        self._output += data
        request._copied += len(data)
        self.ring.device_release(idx, status="ok", bytes_done=snap.length)
        request._descriptors.append(idx)
        request._position += 1
        if request._position == len(request._chain):
            self._finish(request, "ok")
        return True

    def drain(self) -> None:
        """Step until no request is in flight."""
        while self.step():
            pass

    # -- internals ----------------------------------------------------------

    def _release_remaining(self, request: CopyRequest, from_pos: int
                           ) -> tuple[int, ...]:
        """Hand the request's unprocessed descriptors back to the CPU.

        Uses the normal device release/acquire rules: each slot is acquired
        at its frozen generation and released with completion status
        ``skipped`` and ``bytes_done=0``, so the CPU observes a real
        completion record when it re-acquires the slot.
        """
        skipped: list[int] = []
        for pos in range(from_pos, len(request._chain)):
            idx = request._chain[pos]
            self.ring.device_acquire(
                idx, expected_generation=request._generations[pos])
            self.ring.device_release(idx, status="skipped", bytes_done=0)
            skipped.append(idx)
        return tuple(skipped)

    def _finish(self, request: CopyRequest, status: str, *,
                error: str | None = None, error_index: int | None = None,
                skipped: tuple[int, ...] = ()) -> None:
        """Publish the request's single terminal result and free its slots."""
        result = CopyRequestResult(
            request_id=request.request_id,
            head=request.head,
            status=status,
            error=error,
            error_index=error_index,
            bytes_copied=request._copied,
            descriptors=tuple(request._descriptors),
            skipped=skipped,
        )
        request._result = result
        request._done = True
        self.results.append(result)
        self._requests.popleft()
        for idx in request._chain:
            del self._occupied[idx]
