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

Cancellation: ``CopyEngine.request_cancel`` revokes a chain that has not
started or asks to stop the running one.  Descriptors already handed to the
device are not reclaimed immediately; the device confirms at a work
boundary by releasing the unexecuted remainder with status ``"skipped"``,
and the chain completes as ``"cancelled"`` aggregating the completed
prefix and the skipped rest.  Final states emit token/generation-tagged
notifications so stale or duplicated notifications can never affect a new
chain that recycled the same slots, and ``reclaim`` is idempotent.

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
    not appear here.  For a cancelled chain, ``descriptors`` holds the
    prefix that completed before the stop took effect and ``skipped``
    holds the not-yet-executed descriptors the device confirmed it will no
    longer access, so the final state accounts for both the completed and
    the never-executed parts of the chain.
    """

    head: int
    status: str            # "ok", "error" or "cancelled"
    error: str | None
    error_index: int | None
    bytes_copied: int
    descriptors: tuple[int, ...]
    skipped: tuple[int, ...] = ()
    token: int = 0


@dataclass(frozen=True)
class ChainNotification:
    """Device->CPU notification that a chain reached a final state.

    Notifications are tagged with the chain ``token`` (unique per
    submission, never reused) and the ``generation`` the head slot had
    at submit time, so a stale or duplicated notification — e.g. one
    belonging to a chain whose slots have since been reclaimed and reused
    for a new chain — can never be mistaken for a notification about the
    new chain.
    """

    token: int
    head: int
    generation: int
    status: str            # "ok", "error" or "cancelled"


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
    :meth:`run` (or one descriptor at a time via :meth:`step`).  Each
    descriptor is acquired, validated (buffer range and ``next`` pointer),
    copied from the ring pool into the engine output, and released back to
    the CPU with status ``ok``.  On the first bad descriptor the engine
    releases *that* descriptor with status ``error``, completes the chain
    as an error, and halts: the rest of the broken chain and all
    subsequently queued chains are left unconsumed until :meth:`reset_halt`
    is called.  Descriptors completed before the bad one keep their ``ok``
    completions and copied bytes (documented partial completion semantics).

    Cancellation / stop:

    * :meth:`submit_chain` returns a unique chain token.
    * :meth:`request_cancel` asks to revoke a chain that has not started
      yet, or to stop the chain currently executing.  The request only
      takes effect at a device work boundary (the next :meth:`step`):
      descriptors already handed to the device are *not* reclaimed
      immediately — the device first confirms it no longer accesses them
      by releasing each remaining descriptor with status ``"skipped"``.
    * The chain then reaches its final state ``"cancelled"``, whose
      :class:`ChainResult` aggregates the completed prefix
      (``descriptors``, ``bytes_copied``) and the never-executed rest
      (``skipped``).
    * Every final state (ok / error / cancelled) emits exactly one
      :class:`ChainNotification` into :attr:`notifications`, tagged with
      the chain token and the head-slot generation at submit time.  Tokens
      are never reused, so a stale or duplicated notification can never be
      attributed to a later chain that recycled the same ring slots.
    * :meth:`reclaim` is the CPU-side consumption of that notification:
      it raises :class:`OwnershipError` while the device has not confirmed
      the final state, and is idempotent afterwards — repeated calls and
      duplicated notifications release nothing twice.
    """

    def __init__(self, ring: DmaRing) -> None:
        if not isinstance(ring, DmaRing):
            raise TypeError(
                f"ring: expected DmaRing, got {type(ring).__name__}")
        self.ring = ring
        self._output = bytearray()
        self._queue: deque[int] = deque()      # tokens, submission order
        self._chains: dict[int, _ChainState] = {}
        self._current: _ChainState | None = None
        self._next_token = 1
        self.results: list[ChainResult] = []
        self.notifications: deque[ChainNotification] = deque()
        self.halted = False

    @property
    def output(self) -> bytes:
        """All bytes copied so far, in chain/descriptor order."""
        return bytes(self._output)

    @property
    def queued(self) -> int:
        """Number of submitted chains not yet processed."""
        return len(self._queue)

    def submit_chain(self, head) -> int:
        """Queue a chain by head index.  The head must be device-owned.

        Returns the chain token: a unique, never-reused handle used by
        :meth:`request_cancel` and :meth:`reclaim`.
        """
        slot, head = self.ring._slot(head)
        if slot.owner is not Owner.DEVICE:
            raise OwnershipError(
                f"chain head {head}: not released to device "
                f"(owner={slot.owner.value})")
        token = self._next_token
        self._next_token += 1
        self._chains[token] = _ChainState(
            token=token, head=head, head_generation=slot.generation)
        self._queue.append(token)
        return token

    def request_cancel(self, token) -> None:
        """Ask the device to revoke or stop a chain.

        For a queued chain this revokes it before it starts; for the chain
        currently executing it is a stop request that takes effect at the
        next descriptor boundary.  The request is only recorded here — the
        device confirms it (releasing the not-yet-executed descriptors with
        status ``"skipped"``) during :meth:`step`/:meth:`run`.
        Requesting twice while pending is a harmless no-op; requesting for
        an unknown token or an already-final chain raises
        :class:`DmaRingError`.
        """
        state = self._chain_state(token)
        if state.result is not None:
            raise DmaRingError(
                f"chain token {state.token}: already in final state "
                f"{state.result.status!r}; cannot cancel")
        state.cancel_requested = True

    def reclaim(self, token) -> ChainResult:
        """Consume the final-state notification for a chain (CPU side).

        Raises :class:`OwnershipError` if the device has not confirmed the
        chain's final state yet — descriptors already handed to the device
        cannot be reclaimed until then.  Once final, the call discards any
        queued notifications for this token (including stale duplicates)
        and returns the recorded :class:`ChainResult`.  The operation is
        idempotent: repeating it releases no resource twice.
        """
        state = self._chain_state(token)
        if state.result is None:
            raise OwnershipError(
                f"chain token {state.token}: device has not confirmed the "
                f"final state; its descriptors are still device-owned")
        if self.notifications:
            kept = [n for n in self.notifications
                    if not self._notification_matches(state, n)]
            self.notifications.clear()
            self.notifications.extend(kept)
        return state.result

    def run(self) -> None:
        """Process queued chains in order until the queue drains or a chain
        error halts the engine.  Pending cancel requests are confirmed at
        the same work boundaries."""
        while self.step():
            pass

    def step(self) -> bool:
        """Advance the device by one work unit; return True if it did work.

        One work unit is either the confirmation of a requested
        cancellation, or the processing of a single descriptor of the
        current chain.  Interleaving :meth:`step` with
        :meth:`request_cancel` models a stop request landing while a
        chain is executing.  Cancellation confirmations are honoured even
        while the engine is halted — the device can always confirm it will
        not touch a descriptor — but ordinary chain processing still waits
        for :meth:`reset_halt`.
        """
        current = self._current
        if current is not None:
            if current.cancel_requested:
                self._confirm_cancel(current)
                self._current = None
                return True
            if self.halted:
                return False
            self._step_descriptor(current)
            return True
        if not self._queue:
            return False
        state = self._chains[self._queue[0]]
        if state.cancel_requested:
            self._queue.popleft()
            self._confirm_cancel(state)
            return True
        if self.halted:
            return False
        self._queue.popleft()
        self._current = state
        return True

    def reset_halt(self) -> None:
        """Clear the halt flag so queued chains can be processed again."""
        self.halted = False

    # -- internals ----------------------------------------------------------

    def _chain_state(self, token) -> "_ChainState":
        token = _as_int(token, "token")
        state = self._chains.get(token)
        if state is None:
            raise DmaRingError(f"unknown chain token {token}")
        return state

    @staticmethod
    def _notification_matches(state: "_ChainState",
                              note: ChainNotification) -> bool:
        """A notification belongs to `state` only if both the token and the
        submit-time head generation match; anything else is stale (e.g. the
        slot was reclaimed and reused by a newer chain) and must not affect
        the new chain."""
        return (note.token == state.token
                and note.generation == state.head_generation)

    def _finalize(self, state: "_ChainState", result: ChainResult) -> None:
        state.result = result
        self.results.append(result)
        self.notifications.append(ChainNotification(
            token=state.token, head=state.head,
            generation=state.head_generation, status=result.status))

    def _confirm_cancel(self, state: "_ChainState") -> None:
        """Device confirms a cancel/stop: every not-yet-executed descriptor
        is released back to the CPU with status ``"skipped"``, and the
        chain completes as ``"cancelled"`` aggregating the completed
        prefix and the skipped rest."""
        ring = self.ring
        idx = state.idx
        while idx is not None and 0 <= idx < ring.capacity \
                and idx not in state.seen:
            state.seen.add(idx)
            try:
                snap = ring.device_acquire(idx)
            except OwnershipError:
                # Never handed to the device (missing cpu_release); the CPU
                # still owns it, so there is nothing to confirm for it.
                break
            ring.device_release(idx, status="skipped", bytes_done=0)
            state.skipped.append(idx)
            idx = snap.next
        self._finalize(state, ChainResult(
            head=state.head, status="cancelled", error=None,
            error_index=None, bytes_copied=state.copied,
            descriptors=tuple(state.done), skipped=tuple(state.skipped),
            token=state.token))

    def _step_descriptor(self, state: "_ChainState") -> None:
        ring = self.ring
        idx = state.idx
        try:
            snap = ring.device_acquire(idx)
        except OwnershipError as exc:
            # e.g. a descriptor in the chain was never cpu_release'd
            self._fail_chain(state, idx, f"descriptor {idx}: {exc}")
            return
        state.seen.add(idx)
        problem = self._validate_descriptor(idx, snap, state.seen)
        if problem is not None:
            ring.device_release(idx, status="error", error=problem,
                                bytes_done=0)
            state.done.append(idx)
            self._fail_chain(state, idx, problem)
            return
        data = bytes(ring.pool[snap.offset: snap.offset + snap.length])
        self._output += data
        state.copied += len(data)
        ring.device_release(idx, status="ok", bytes_done=snap.length)
        state.done.append(idx)
        state.idx = snap.next
        if state.idx is None:
            self._finalize(state, ChainResult(
                head=state.head, status="ok", error=None, error_index=None,
                bytes_copied=state.copied, descriptors=tuple(state.done),
                token=state.token))
            self._current = None

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

    def _fail_chain(self, state: "_ChainState", idx: int,
                    problem: str) -> None:
        self._finalize(state, ChainResult(
            head=state.head, status="error", error=problem, error_index=idx,
            bytes_copied=state.copied, descriptors=tuple(state.done),
            token=state.token))
        self.halted = True
        self._current = None


class _ChainState:
    """Per-submission bookkeeping for one chain (internal)."""

    __slots__ = (
        "token", "head", "head_generation", "idx", "seen",
        "done", "skipped", "copied", "cancel_requested", "result",
    )

    def __init__(self, *, token: int, head: int, head_generation: int
                 ) -> None:
        self.token = token
        self.head = head
        self.head_generation = head_generation
        self.idx: int | None = head
        self.seen: set[int] = set()
        self.done: list[int] = []
        self.skipped: list[int] = []
        self.copied = 0
        self.cancel_requested = False
        self.result: ChainResult | None = None
