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

Requests: a request groups several explicitly listed descriptors.  The
device may complete them in any order; each per-descriptor completion
returns that slot to the CPU immediately, but the aggregate
:class:`RequestResult` is published only once *all* members have completed.
Membership records each slot's generation, so a replayed completion for a
recycled slot is rejected as stale, and completing the same
(request, slot, generation) twice is rejected as a duplicate.  An error
completion finalises the whole request: every still-pending member is
released back to the CPU with status ``skipped``, so an error request never
leaves occupied slots behind (the ring can wrap immediately).

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


class DuplicateCompletionError(DmaRingError):
    """A descriptor was completed twice for the same (request, generation).

    Attributes:
        request_id: the request the completion was submitted to.
        index:      ring index of the descriptor.
        generation: the generation recorded in the request membership.
    """

    def __init__(self, message: str, *, request_id, index, generation) -> None:
        super().__init__(message)
        self.request_id = request_id
        self.index = index
        self.generation = generation


class RequestError(DmaRingError):
    """A request is malformed or its lifecycle was violated.

    Attributes:
        request_id: id of the offending request, or ``None`` when the request
                    was rejected at registration time.
        index:      ring index of the offending member, if any.
        reason:     short machine-stable reason string.
    """

    def __init__(self, message: str, *, request_id=None, index=None,
                 reason: str | None = None) -> None:
        super().__init__(message)
        self.request_id = request_id
        self.index = index
        self.reason = reason


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
class RequestCompletion:
    """One member descriptor's outcome inside a :class:`RequestResult`.

    Members may complete out of order; records inside a ``RequestResult`` are
    always presented in request (membership) order, not completion order.
    """

    index: int
    generation: int
    status: str            # "ok", "error" or "skipped"
    error: str | None
    bytes_done: int


@dataclass(frozen=True)
class RequestResult:
    """Aggregate outcome of a multi-descriptor request.

    Published only after every member has completed.  ``status`` is ``ok``
    exactly when every member completed ``ok``; otherwise it is ``error`` and
    ``error_index`` names the first descriptor that completed with an error.
    Members still outstanding when the request is finalised on an error are
    force-released with status ``skipped``; their slots return to the CPU so
    the ring can wrap, but no bytes are copied for them.
    """

    request_id: int
    status: str            # "ok" or "error"
    error: str | None
    error_index: int | None
    bytes_copied: int
    completions: tuple[RequestCompletion, ...]


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
        "request_id",                        # active request owning this slot
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
        self.request_id: int | None = None


# ---------------------------------------------------------------------------
# In-flight request (internal)
# ---------------------------------------------------------------------------

class _Request:
    """A registered multi-descriptor request and its completion bookkeeping.

    ``members`` preserves request (membership) order: the
    :class:`RequestResult` is presented in that order regardless of the
    actual completion order.  ``pending`` counts descriptors not yet
    accounted for (a normal completion or the first error each accounts for
    exactly one); the request publishes as soon as it reaches zero.
    """

    __slots__ = (
        "request_id", "members", "gens", "pending", "results",
        "finished", "status", "error", "error_index",
    )

    def __init__(self, request_id: int, members: list[tuple[int, int]]) -> None:
        self.request_id = request_id
        self.members = members                    # [(index, generation)] in order
        self.gens = {idx: gen for idx, gen in members}
        self.pending = len(members)
        # index -> RequestCompletion, keyed while the request is in flight
        self.results: dict[int, RequestCompletion] = {}
        self.finished = False
        self.status = "ok"
        self.error: str | None = None
        self.error_index: int | None = None


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
        # request_id -> _Request for in-flight (not yet finalised) requests
        self._requests: dict[int, _Request] = {}
        # request_id -> RequestResult awaiting cpu request_acquire
        self._request_results: dict[int, RequestResult] = {}
        self._next_request_id = 1

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

    def device_release(self, index, *, status, bytes_done=0, error=None,
                       request_id=None) -> None:
        """Publish a completion and hand the descriptor back to the CPU.

        Without ``request_id`` this is the plain per-descriptor release
        (unchanged semantics).  With ``request_id`` the completion is
        accounted against that request's membership:

        * the slot must belong to the request (else :class:`RequestError`),
        * the generation recorded at registration must still match the
          slot's committed generation — a mismatch means the slot was
          recycled and this completion is *stale* (:class:`GenerationError`),
        * the same (request, slot, generation) may complete only once —
          a replayed completion raises :class:`DuplicateCompletionError`.

        Request-scoped replay/stale checks run *before* the ownership check
        so a duplicate or stale completion is reported as such even when the
        slot has meanwhile been handed back to the CPU.
        """
        slot, i = self._slot(index)
        req: _Request | None = None
        if request_id is not None:
            rid = _as_int(request_id, "request_id")
            req = self._requests.get(rid)
            if req is None:
                finished = self._request_results.get(rid)
                if finished is not None:
                    for comp in finished.completions:
                        if comp.index == i:
                            if slot.generation != comp.generation:
                                raise GenerationError(
                                    f"descriptor {i}: request {rid} "
                                    f"finished at generation "
                                    f"{comp.generation}; slot has been "
                                    f"recycled to generation "
                                    f"{slot.generation} (stale completion)")
                            raise DuplicateCompletionError(
                                f"descriptor {i}: request {rid} already "
                                f"recorded this completion "
                                f"(generation {comp.generation})",
                                request_id=rid, index=i,
                                generation=comp.generation)
                raise RequestError(
                    f"descriptor {i}: request {rid} is not in flight",
                    request_id=rid, index=i, reason="unknown-request")
            if i not in req.gens:
                raise RequestError(
                    f"descriptor {i}: not a member of request {rid}",
                    request_id=rid, index=i, reason="not-a-member")
            if req.gens[i] != slot.generation:
                raise GenerationError(
                    f"descriptor {i}: request {rid} recorded generation "
                    f"{req.gens[i]}, committed generation is "
                    f"{slot.generation} (stale completion)")
            if i in req.results:
                raise DuplicateCompletionError(
                    f"descriptor {i}: request {rid} already completed this "
                    f"(slot, generation {req.gens[i]})",
                    request_id=rid, index=i, generation=req.gens[i])
        elif slot.request_id is not None:
            raise RequestError(
                f"descriptor {i}: belongs to request {slot.request_id}; "
                f"release it with request_id={slot.request_id}",
                request_id=slot.request_id, index=i,
                reason="request-owned-slot")
        if slot.owner is not Owner.DEVICE or not slot.device_held:
            raise OwnershipError(
                f"descriptor {i}: device cannot release "
                f"(owner={slot.owner.value}, device_held={slot.device_held})")
        if not isinstance(status, str) or status not in VALID_STATUSES:
            raise ValueError(
                f"descriptor {i}: status must be one of {VALID_STATUSES}, "
                f"got {status!r}")
        if req is not None and status == "skipped":
            raise ValueError(
                f"descriptor {i}: status 'skipped' is reserved for the "
                f"ring's force-release when a request fails; members are "
                f"completed with 'ok' or 'error'")
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
        slot.request_id = None
        if req is not None:
            req.results[i] = RequestCompletion(
                i, req.gens[i], status, error, bytes_done)
            req.pending -= 1
            if status == "error":
                self._finish_request_error(req, i, error)
            elif req.pending == 0:
                self._publish_request(req)

    # -- requests ------------------------------------------------------------

    def register_request(self, indices) -> tuple[int, tuple[int, ...]]:
        """Register a multi-descriptor request over already-released slots.

        ``indices`` is a list/tuple of ring indices, in request order; every
        slot must currently be owned by the device (i.e. the CPU has
        called ``cpu_release`` on it).  Each slot's committed generation is
        recorded now; any later completion carrying a different generation is
        rejected as stale.

        Returns ``(request_id, generations)`` where ``generations`` parallels
        ``indices``.  All validation happens before any state changes, so a
        rejected call leaves the ring untouched.
        """
        if not isinstance(indices, (list, tuple)):
            raise TypeError(
                f"indices: expected list or tuple of ring indices, got "
                f"{type(indices).__name__}")
        if not indices:
            raise ValueError("indices: a request needs at least one member")
        parsed: list[int] = []
        for raw in indices:
            slot, idx = self._slot(raw)   # validates type and range
            parsed.append(idx)
        if len(set(parsed)) != len(parsed):
            dup = next(i for i in parsed if parsed.count(i) > 1)
            raise RequestError(
                f"request: index {dup} listed more than once",
                index=dup, reason="duplicate-member")
        for idx in parsed:
            slot = self._slots[idx]
            if slot.owner is not Owner.DEVICE:
                raise RequestError(
                    f"descriptor {idx}: not released to device "
                    f"(owner={slot.owner.value})",
                    index=idx, reason="not-device-owned")
            if slot.device_held:
                raise RequestError(
                    f"descriptor {idx}: device already holds this slot",
                    index=idx, reason="device-held")
            if slot.request_id is not None:
                raise RequestError(
                    f"descriptor {idx}: already belongs to request "
                    f"{slot.request_id}",
                    request_id=slot.request_id, index=idx,
                    reason="already-in-request")
        rid = self._next_request_id
        self._next_request_id += 1
        members = [(idx, self._slots[idx].generation) for idx in parsed]
        self._requests[rid] = _Request(rid, members)
        for idx in parsed:
            self._slots[idx].request_id = rid
        return rid, tuple(gen for _, gen in members)

    def request_is_done(self, request_id) -> bool:
        """True once the request's aggregate result has been published."""
        rid = _as_int(request_id, "request_id")
        if rid in self._requests:
            return False
        if rid in self._request_results:
            return True
        raise RequestError(
            f"request {rid}: unknown request id",
            request_id=rid, reason="unknown-request")

    def request_acquire(self, request_id) -> RequestResult:
        """Read a finished request's aggregate result.

        Raises :class:`RequestError` while the request is still in flight
        (result not yet published) or if the id is unknown.  The result stays
        readable afterwards (idempotent).
        """
        rid = _as_int(request_id, "request_id")
        if rid in self._requests:
            raise RequestError(
                f"request {rid}: still in flight "
                f"({self._requests[rid].pending} descriptor(s) pending)",
                request_id=rid, reason="in-flight")
        try:
            return self._request_results[rid]
        except KeyError:
            raise RequestError(
                f"request {rid}: unknown request id",
                request_id=rid, reason="unknown-request") from None

    def _publish_request(self, req: _Request) -> None:
        """Assemble and publish the aggregate result of a finished request."""
        req.finished = True
        completions = tuple(req.results[idx] for idx, _ in req.members)
        bytes_copied = sum(c.bytes_done for c in completions
                           if c.status == "ok")
        result = RequestResult(
            request_id=req.request_id, status=req.status, error=req.error,
            error_index=req.error_index, bytes_copied=bytes_copied,
            completions=completions)
        self._request_results[req.request_id] = result
        del self._requests[req.request_id]

    def _finish_request_error(self, req: _Request, index: int,
                              error: str | None) -> None:
        """Finalise a request on an error completion.

        Members not yet completed are force-released back to the CPU with
        status ``skipped`` (bytes_done=0): their slots are freed so the ring
        can wrap and be reused, and the aggregate result is published at
        once.  Only the first error finalises the request.
        """
        if req.finished:
            return
        req.finished = True
        req.status = "error"
        req.error = error
        req.error_index = index
        for idx, gen in req.members:
            if idx in req.results:
                continue
            slot = self._slots[idx]
            # Normal protocol invariant: a pending member is still owned by
            # this request at the recorded generation (the CPU cannot
            # recycle a device-owned slot).  If a caller violated ownership
            # out-of-band and the slot has moved on, do not clobber it: it is
            # still accounted as skipped in the aggregate result.
            if slot.request_id != req.request_id or slot.generation != gen:
                req.results[idx] = RequestCompletion(
                    idx, gen, "skipped", None, 0)
                req.pending -= 1
                continue
            slot.completion = ("skipped", None, 0)
            slot.completions += 1
            slot.device_held = False
            slot.owner = Owner.CPU
            slot.request_id = None
            req.results[idx] = RequestCompletion(idx, gen, "skipped", None, 0)
            req.pending -= 1
        self._publish_request(req)

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
            "request_id": slot.request_id,
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

    Requests: :meth:`submit_request` registers a multi-descriptor request
    (see :meth:`DmaRing.register_request`) and queues it.  Members are
    processed in ``order`` (default: request order) — the device may
    complete them out of order; the aggregate :class:`RequestResult` is
    published by the ring only when all members are accounted for, and is
    mirrored into :attr:`request_results`.  A bad member is released with
    status ``error``; the ring then force-releases the remaining members as
    ``skipped`` and the engine halts exactly as for a broken chain.
    """

    def __init__(self, ring: DmaRing) -> None:
        if not isinstance(ring, DmaRing):
            raise TypeError(
                f"ring: expected DmaRing, got {type(ring).__name__}")
        self.ring = ring
        self._output = bytearray()
        # queue entries: ("chain", head) or ("request", request_id, order)
        self._queue: deque[tuple] = deque()
        self.results: list[ChainResult] = []
        self.request_results: list[RequestResult] = []
        self.halted = False

    @property
    def output(self) -> bytes:
        """All bytes copied so far, in chain/descriptor order."""
        return bytes(self._output)

    @property
    def queued(self) -> int:
        """Number of submitted chains/requests not yet processed."""
        return len(self._queue)

    def submit_chain(self, head) -> None:
        """Queue a chain by head index.  The head must be device-owned."""
        slot, head = self.ring._slot(head)
        if slot.owner is not Owner.DEVICE:
            raise OwnershipError(
                f"chain head {head}: not released to device "
                f"(owner={slot.owner.value})")
        if slot.request_id is not None:
            raise RequestError(
                f"chain head {head}: already belongs to request "
                f"{slot.request_id}",
                request_id=slot.request_id, index=head,
                reason="already-in-request")
        self._queue.append(("chain", head))

    def submit_request(self, indices, *, order=None) -> int:
        """Register and queue a multi-descriptor request; return its id.

        ``indices`` lists the member slots in request order (each must have
        been released by ``cpu_release``).  ``order``, if given, is a
        permutation of ``indices`` fixing the device's completion order —
        out-of-order completion is legal.  All validation happens before
        anything is queued, so a rejected call changes nothing.
        """
        if not isinstance(indices, (list, tuple)):
            raise TypeError(
                f"indices: expected list or tuple of ring indices, got "
                f"{type(indices).__name__}")
        members = [self.ring._slot(raw)[1] for raw in indices]
        proc_order = members
        if order is not None:
            if not isinstance(order, (list, tuple)):
                raise TypeError(
                    f"order: expected list or tuple of ring indices, got "
                    f"{type(order).__name__}")
            proc_order = [self.ring._slot(raw)[1] for raw in order]
            if sorted(proc_order) != sorted(members):
                raise RequestError(
                    f"order {list(order)!r} is not a permutation of the "
                    f"request members {list(indices)!r}",
                    reason="bad-order")
        # register_request performs the duplicate/ownership checks atomically;
        # only after it succeeds do we mutate the queue.
        rid, _gens = self.ring.register_request(members)
        self._queue.append(("request", rid, proc_order))
        return rid

    def run(self) -> None:
        """Process queued chains/requests in order until the queue drains or
        an error halts the engine."""
        while self._queue and not self.halted:
            entry = self._queue.popleft()
            if entry[0] == "chain":
                self._process_chain(entry[1])
            else:
                self._process_request(entry[1], entry[2])

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
            # idx is always in range here: the head was range-checked at
            # submit, and each subsequent idx was range-checked by the
            # previous descriptor's validation before traversal.
            if ring._slots[idx].request_id is not None:
                self._fail_chain(
                    head, done, copied, idx,
                    f"descriptor {idx}: belongs to request "
                    f"{ring._slots[idx].request_id}")
                return
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

    def _process_request(self, rid: int, order: list[int]) -> None:
        """Process one request, completing members in ``order``.

        A request lists its members explicitly, so ``next`` pointers are not
        traversed; only each member's buffer range is validated.  Bytes
        copied by members completed ``ok`` before the first error remain in
        the engine output; the ring accounts the rest as ``skipped``.
        """
        ring = self.ring
        for idx in order:
            snap = ring.device_acquire(idx)
            end = snap.offset + snap.length
            if end > len(ring.pool):
                problem = (f"descriptor {idx}: buffer range "
                           f"[{snap.offset}, {end}) exceeds pool size "
                           f"{len(ring.pool)}")
                # The ring finalises the whole request here: remaining
                # members come back as "skipped" and their slots are freed.
                ring.device_release(idx, status="error", error=problem,
                                    bytes_done=0, request_id=rid)
                self.request_results.append(ring.request_acquire(rid))
                self.halted = True
                return
            data = bytes(ring.pool[snap.offset: snap.offset + snap.length])
            self._output += data
            ring.device_release(idx, status="ok", bytes_done=snap.length,
                                request_id=rid)
        self.request_results.append(ring.request_acquire(rid))
