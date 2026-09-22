"""Core synthetic DMA ring with explicit ownership handoff.

Model
-----
The module simulates -- with no real hardware -- the ownership protocol of a
DMA descriptor ring:

- A :class:`DmaRing` holds ``capacity`` descriptors (power of two, <= 64).
- Each descriptor carries ``generation``, a buffer range (``offset`` /
  ``length`` into a :class:`CoherentMemory`), a ``next`` chain pointer and an
  ``owner`` (``"cpu"`` or ``"device"``).
- The CPU writes fields only while it owns the descriptor, then
  :meth:`DmaRing.cpu_release` publishes a snapshot (the flush event) and
  hands ownership to the device.
- The device must :meth:`DmaRing.device_acquire` before it may read the
  published fields; the acquire also guards against re-consuming a stale
  generation (ABA after ring wrap-around).
- After copying, the device calls :meth:`DmaRing.device_release`, which
  writes the completion fields and hands ownership back.
- The CPU may not read completion fields until :meth:`DmaRing.cpu_acquire`
  for the current cycle; reading earlier raises :class:`VisibilityError`.

Memory coherence is explicit: :class:`CoherentMemory` keeps a CPU view and a
device view; only :meth:`CoherentMemory.flush` publishes CPU writes to the
device, and :meth:`CoherentMemory.invalidate` publishes device writes back.

Numeric contract: all numeric inputs are integers.  ``bool`` is rejected for
integer parameters, and floats (including NaN and Infinity) are rejected as
non-integers.  Nothing is silently coerced.
"""

from __future__ import annotations

from typing import NamedTuple, Optional, Tuple

from .errors import (
    CapacityError,
    ChainError,
    DeviceHaltedError,
    DmaRingError,
    OwnershipError,
    StaleGenerationError,
    ValidationError,
    VisibilityError,
)

OWNER_CPU = "cpu"
OWNER_DEVICE = "device"

STATUS_OK = "ok"
STATUS_ERROR = "error"
_VALID_STATUS = (STATUS_OK, STATUS_ERROR)

#: Chain terminator for the ``next`` field.
END = None

#: Maximum ring capacity (must be a power of two).
MAX_CAPACITY = 64

#: Maximum coherent memory size in bytes (1 MiB).
MAX_MEMORY_SIZE = 1 << 20

ChainPointer = Optional[Tuple[int, int]]  # (ring_id, slot) or None (= END)


# ---------------------------------------------------------------------------
# validation helpers
# ---------------------------------------------------------------------------

def _require_int(name: str, value, *, minimum=None, maximum=None) -> int:
    """Validate that ``value`` is an int (never bool, never float)."""
    if isinstance(value, bool):
        raise ValidationError(
            f"{name}: bool is not accepted as an integer (got {value!r})"
        )
    if not isinstance(value, int):
        raise ValidationError(
            f"{name}: expected int, got {type(value).__name__} ({value!r}); "
            "floats including NaN/Infinity are rejected"
        )
    if minimum is not None and value < minimum:
        raise ValidationError(f"{name}: {value} < minimum {minimum}")
    if maximum is not None and value > maximum:
        raise ValidationError(f"{name}: {value} > maximum {maximum}")
    return value


def _require_pointer(name: str, value) -> ChainPointer:
    """Validate a ``next`` chain pointer: None or (ring_id, slot) of ints."""
    if value is None:
        return None
    if isinstance(value, tuple) and len(value) == 2:
        ring_id = _require_int(f"{name}[0] (ring_id)", value[0], minimum=0)
        slot = _require_int(f"{name}[1] (slot)", value[1], minimum=0)
        return (ring_id, slot)
    raise ValidationError(
        f"{name}: expected None or a (ring_id, slot) tuple, got {value!r}"
    )


# ---------------------------------------------------------------------------
# public immutable views
# ---------------------------------------------------------------------------

class DescriptorSnapshot(NamedTuple):
    """What the device sees after a successful ``device_acquire``."""

    ring_id: int
    slot: int
    generation: int
    offset: int
    length: int
    next: ChainPointer


class Completion(NamedTuple):
    """Completion record written by the device, read by the CPU."""

    ring_id: int
    slot: int
    generation: int
    status: str  # STATUS_OK or STATUS_ERROR
    bytes_copied: int


class ChainResult(NamedTuple):
    """Outcome of processing one scatter/gather chain."""

    status: str  # STATUS_OK or STATUS_ERROR
    items: Tuple[Completion, ...]  # one per consumed descriptor, in order
    bytes_copied: int
    error: Optional[str]  # human-readable cause when status == STATUS_ERROR


# ---------------------------------------------------------------------------
# coherent memory
# ---------------------------------------------------------------------------

class CoherentMemory:
    """Byte storage with an explicit CPU/device visibility boundary.

    ``cpu_write`` / ``cpu_read`` operate on the CPU view; ``device_read``
    operates on the device view.  The views diverge until ``flush()``
    (CPU -> device) or ``invalidate()`` (device -> CPU) is called.
    """

    def __init__(self, size: int):
        _require_int("size", size, minimum=1, maximum=MAX_MEMORY_SIZE)
        self.size = size
        self._cpu_view = bytearray(size)
        self._device_view = bytearray(size)

    def _check_range(self, name: str, offset: int, length: int) -> None:
        _require_int(f"{name}.offset", offset, minimum=0)
        _require_int(f"{name}.length", length, minimum=0)
        if offset + length > self.size:
            raise ValidationError(
                f"{name}: range [{offset}, {offset + length}) exceeds "
                f"memory size {self.size}"
            )

    def cpu_write(self, offset: int, data) -> None:
        """Write bytes into the CPU view (not visible to the device yet)."""
        if isinstance(data, (bytes, bytearray, memoryview)):
            payload = bytes(data)
        else:
            raise ValidationError(
                f"cpu_write.data: expected bytes-like, got "
                f"{type(data).__name__} ({data!r})"
            )
        self._check_range("cpu_write", offset, len(payload))
        self._cpu_view[offset:offset + len(payload)] = payload

    def cpu_read(self, offset: int, length: int) -> bytes:
        self._check_range("cpu_read", offset, length)
        return bytes(self._cpu_view[offset:offset + length])

    def device_read(self, offset: int, length: int) -> bytes:
        self._check_range("device_read", offset, length)
        return bytes(self._device_view[offset:offset + length])

    def flush(self) -> None:
        """Publish the CPU view to the device (write-back)."""
        self._device_view[:] = self._cpu_view

    def invalidate(self) -> None:
        """Publish the device view to the CPU (write-invalidate)."""
        self._cpu_view[:] = self._device_view


# ---------------------------------------------------------------------------
# descriptor ring
# ---------------------------------------------------------------------------

class _Descriptor:
    __slots__ = ("generation", "offset", "length", "next", "owner",
                 "status", "bytes_copied")

    def __init__(self) -> None:
        self.generation = 0
        self.offset = 0
        self.length = 0
        self.next: ChainPointer = None
        self.owner = OWNER_CPU
        self.status = STATUS_OK
        self.bytes_copied = 0


class DmaRing:
    """A ring of DMA descriptors with CPU/device ownership handoff."""

    def __init__(self, capacity: int, *, ring_id: int = 0):
        _require_int("capacity", capacity, minimum=1, maximum=MAX_CAPACITY)
        if capacity & (capacity - 1):
            raise CapacityError(
                f"capacity: {capacity} is not a power of two"
            )
        _require_int("ring_id", ring_id, minimum=0)
        self.capacity = capacity
        self.ring_id = ring_id
        self._desc = [_Descriptor() for _ in range(capacity)]
        # Snapshot published to the device by cpu_release (flush event).
        self._device_view = [None] * capacity  # list[Optional[DescriptorSnapshot]]
        # Completion published by device_release, pending cpu_acquire.
        self._pending = [None] * capacity  # list[Optional[Completion]]
        # Completion the CPU has acquired in the current cycle.
        self._acquired = [None] * capacity  # list[Optional[Completion]]
        # Last generation the device acquired per slot (ABA guard).
        self._last_device_gen = [None] * capacity  # list[Optional[int]]

    # -- internal helpers --------------------------------------------------

    def _slot(self, slot) -> int:
        return _require_int("slot", slot, minimum=0, maximum=self.capacity - 1)

    def _require_owner(self, slot: int, expected: str, op: str) -> _Descriptor:
        desc = self._desc[slot]
        if desc.owner != expected:
            raise OwnershipError(
                f"{op}: slot {slot} of ring {self.ring_id} is owned by "
                f"{desc.owner!r}, not {expected!r}"
            )
        return desc

    # -- CPU side ------------------------------------------------------------

    def cpu_write(self, slot: int, *, offset: int, length: int,
                  next: ChainPointer = None) -> None:
        """Write descriptor fields.  Requires CPU ownership of the slot.

        ``offset``/``length`` are validated as non-negative integers; whether
        the range fits the target memory is a *device-side* chain-validation
        concern (a bad range produces an error completion, not a write-time
        failure), mirroring how hardware discovers it at doorbell time.
        """
        slot = self._slot(slot)
        offset = _require_int("offset", offset, minimum=0)
        length = _require_int("length", length, minimum=0)
        nxt = _require_pointer("next", next)
        desc = self._require_owner(slot, OWNER_CPU, "cpu_write")
        desc.offset = offset
        desc.length = length
        desc.next = nxt

    def cpu_write_many(self, writes) -> None:
        """Atomically apply a batch of ``cpu_write`` operations.

        ``writes`` is an iterable of dicts with keys ``slot``, ``offset``,
        ``length`` and optional ``next``.  Every entry is validated first;
        if any entry is invalid, nothing is applied and the first
        :class:`ValidationError` (or :class:`OwnershipError`) is raised.
        """
        if not isinstance(writes, (list, tuple)):
            writes = list(writes)
        prepared = []
        for i, entry in enumerate(writes):
            if not isinstance(entry, dict):
                raise ValidationError(
                    f"cpu_write_many[{i}]: expected dict, got "
                    f"{type(entry).__name__} ({entry!r})"
                )
            unknown = set(entry) - {"slot", "offset", "length", "next"}
            if unknown:
                raise ValidationError(
                    f"cpu_write_many[{i}]: unknown keys {sorted(unknown)}"
                )
            for key in ("slot", "offset", "length"):
                if key not in entry:
                    raise ValidationError(
                        f"cpu_write_many[{i}]: missing key {key!r}"
                    )
            slot = self._slot(entry["slot"])
            offset = _require_int(
                f"cpu_write_many[{i}].offset", entry["offset"], minimum=0)
            length = _require_int(
                f"cpu_write_many[{i}].length", entry["length"], minimum=0)
            nxt = _require_pointer(
                f"cpu_write_many[{i}].next", entry.get("next"))
            # Ownership checked up-front so the batch stays atomic.
            self._require_owner(slot, OWNER_CPU, f"cpu_write_many[{i}]")
            prepared.append((slot, offset, length, nxt))
        for slot, offset, length, nxt in prepared:
            desc = self._desc[slot]
            desc.offset = offset
            desc.length = length
            desc.next = nxt

    def cpu_release(self, slot: int) -> int:
        """Hand the descriptor to the device; returns the new generation.

        Bumps the slot's generation, publishes a snapshot to the device
        view (the flush event) and flips ownership to the device.  Any
        completion the CPU acquired in the previous cycle is retired.
        """
        slot = self._slot(slot)
        desc = self._require_owner(slot, OWNER_CPU, "cpu_release")
        desc.generation += 1
        self._acquired[slot] = None
        self._device_view[slot] = DescriptorSnapshot(
            ring_id=self.ring_id,
            slot=slot,
            generation=desc.generation,
            offset=desc.offset,
            length=desc.length,
            next=desc.next,
        )
        desc.owner = OWNER_DEVICE
        return desc.generation

    def cpu_acquire(self, slot: int) -> Completion:
        """Acquire the completion the device published for this cycle."""
        slot = self._slot(slot)
        self._require_owner(slot, OWNER_CPU, "cpu_acquire")
        comp = self._pending[slot]
        if comp is None:
            raise VisibilityError(
                f"cpu_acquire: no completion pending for slot {slot} of "
                f"ring {self.ring_id} (device has not released this cycle)"
            )
        self._pending[slot] = None
        self._acquired[slot] = comp
        return comp

    def completion(self, slot: int) -> Completion:
        """Read the completion fields of the current cycle.

        Raises :class:`VisibilityError` unless the CPU has performed
        ``cpu_acquire`` for the current ownership cycle.
        """
        slot = self._slot(slot)
        comp = self._acquired[slot]
        if comp is None:
            if self._pending[slot] is not None:
                raise VisibilityError(
                    f"completion: slot {slot} of ring {self.ring_id} has a "
                    "device-released completion that the CPU has not "
                    "acquired; call cpu_acquire first"
                )
            raise VisibilityError(
                f"completion: no completion available for slot {slot} of "
                f"ring {self.ring_id}"
            )
        return comp

    # -- device side ---------------------------------------------------------

    def device_acquire(self, slot: int) -> DescriptorSnapshot:
        """Acquire the published descriptor snapshot (device side).

        Requires device ownership.  Rejects re-acquiring a generation the
        device already consumed -- after ring wrap-around the slot may hold
        identical field values, but the generation differs, so a stale
        acquire is an ABA bug and raises :class:`StaleGenerationError`.
        """
        slot = self._slot(slot)
        self._require_owner(slot, OWNER_DEVICE, "device_acquire")
        snap = self._device_view[slot]
        if snap is None:  # defensive; ownership implies a published snapshot
            raise VisibilityError(
                f"device_acquire: slot {slot} of ring {self.ring_id} has no "
                "published snapshot (missing cpu_release)"
            )
        if self._last_device_gen[slot] == snap.generation:
            raise StaleGenerationError(
                f"device_acquire: slot {slot} of ring {self.ring_id} still "
                f"carries generation {snap.generation}, already consumed; "
                "stale re-acquire (ABA) rejected"
            )
        self._last_device_gen[slot] = snap.generation
        return snap

    def device_release(self, slot: int, *, status: str,
                       bytes_copied: int) -> Completion:
        """Write completion fields and hand the descriptor back to the CPU."""
        slot = self._slot(slot)
        desc = self._require_owner(slot, OWNER_DEVICE, "device_release")
        if status not in _VALID_STATUS:
            raise ValidationError(
                f"status: expected one of {_VALID_STATUS}, got {status!r}"
            )
        bytes_copied = _require_int("bytes_copied", bytes_copied, minimum=0)
        desc.status = status
        desc.bytes_copied = bytes_copied
        comp = Completion(
            ring_id=self.ring_id,
            slot=slot,
            generation=desc.generation,
            status=status,
            bytes_copied=bytes_copied,
        )
        self._pending[slot] = comp
        desc.owner = OWNER_CPU
        return comp

    # -- introspection ---------------------------------------------------------

    def owner_of(self, slot: int) -> str:
        """Return the current owner (``"cpu"`` or ``"device"``) of a slot."""
        slot = self._slot(slot)
        return self._desc[slot].owner

    def generation_of(self, slot: int) -> int:
        """Return the current generation counter of a slot."""
        slot = self._slot(slot)
        return self._desc[slot].generation


# ---------------------------------------------------------------------------
# synthetic device
# ---------------------------------------------------------------------------

class SyntheticDevice:
    """Simulated DMA engine: walks scatter/gather chains and copies bytes.

    For each descriptor in a chain the device acquires it, validates the
    buffer range against the memory size, copies the bytes from the device
    view of memory into its output stream, and releases the descriptor with
    an ok completion.  A bad descriptor (acquire failure, out-of-bounds
    range, chain loop, dangling ring pointer) completes the chain with an
    error: already-consumed items keep their completions, the offending
    descriptor gets an error completion when the device owns it, the rest of
    the chain is not consumed, and the device halts -- subsequent chains are
    rejected with :class:`DeviceHaltedError` until :meth:`reset`.
    """

    def __init__(self, memory: CoherentMemory):
        if not isinstance(memory, CoherentMemory):
            raise ValidationError(
                f"memory: expected CoherentMemory, got "
                f"{type(memory).__name__}"
            )
        self.memory = memory
        self.output = bytearray()
        #: (ring_id, slot) -> number of completions issued for that descriptor.
        self.completion_counts = {}
        self.halted = False
        self.halt_reason: Optional[str] = None

    @staticmethod
    def _normalize_rings(rings) -> dict:
        if isinstance(rings, DmaRing):
            rings = [rings]
        if isinstance(rings, dict):
            items = list(rings.items())
        elif isinstance(rings, (list, tuple)):
            items = []
            for ring in rings:
                if not isinstance(ring, DmaRing):
                    raise ValidationError(
                        f"rings: expected DmaRing entries, got "
                        f"{type(ring).__name__} ({ring!r})"
                    )
                items.append((ring.ring_id, ring))
        else:
            raise ValidationError(
                f"rings: expected DmaRing, sequence or dict, got "
                f"{type(rings).__name__}"
            )
        ring_map = {}
        for key, ring in items:
            if not isinstance(ring, DmaRing):
                raise ValidationError(
                    f"rings: expected DmaRing entries, got "
                    f"{type(ring).__name__} ({ring!r})"
                )
            ring_map[key] = ring
        return ring_map

    def _count(self, ring_id: int, slot: int) -> None:
        key = (ring_id, slot)
        self.completion_counts[key] = self.completion_counts.get(key, 0) + 1

    def completion_count(self, ring_id: int, slot: int) -> int:
        """Number of completions issued for one descriptor so far."""
        _require_int("ring_id", ring_id, minimum=0)
        _require_int("slot", slot, minimum=0)
        return self.completion_counts.get((ring_id, slot), 0)

    def process_chain(self, rings, head: ChainPointer) -> ChainResult:
        """Validate and consume one scatter/gather chain starting at ``head``.

        ``rings`` maps ring ids to :class:`DmaRing` objects (a single ring,
        a sequence of rings, or a ``{ring_id: ring}`` dict).  ``head`` is a
        ``(ring_id, slot)`` pointer or ``None`` (an empty chain, which
        completes ok with no items).
        """
        if self.halted:
            raise DeviceHaltedError(
                f"device halted after previous chain error: "
                f"{self.halt_reason}; call reset() before submitting again"
            )
        ring_map = self._normalize_rings(rings)
        head = _require_pointer("head", head)

        items = []
        total = 0
        visited = set()
        cur = head
        error = None
        while cur is not None:
            ring_id, slot = cur
            if cur in visited:
                error = (f"chain loop detected at (ring={ring_id}, "
                         f"slot={slot})")
                break
            visited.add(cur)
            ring = ring_map.get(ring_id)
            if ring is None:
                error = f"chain references unknown ring_id {ring_id}"
                break
            try:
                snap = ring.device_acquire(slot)
            except DmaRingError as exc:
                error = (f"acquire failed at (ring={ring_id}, slot={slot}): "
                         f"{exc}")
                break
            end = snap.offset + snap.length
            if end > self.memory.size:
                comp = ring.device_release(
                    slot, status=STATUS_ERROR, bytes_copied=0)
                self._count(ring_id, slot)
                items.append(comp)
                error = (f"buffer range [{snap.offset}, {end}) exceeds "
                         f"memory size {self.memory.size} at "
                         f"(ring={ring_id}, slot={slot})")
                break
            data = self.memory.device_read(snap.offset, snap.length)
            self.output += data
            comp = ring.device_release(
                slot, status=STATUS_OK, bytes_copied=snap.length)
            self._count(ring_id, slot)
            items.append(comp)
            total += snap.length
            cur = snap.next

        status = STATUS_OK if error is None else STATUS_ERROR
        if error is not None:
            self.halted = True
            self.halt_reason = error
        return ChainResult(
            status=status,
            items=tuple(items),
            bytes_copied=total,
            error=error,
        )

    def reset(self) -> None:
        """Clear the halt state (completions and output are preserved)."""
        self.halted = False
        self.halt_reason = None


__all__ = [
    "OWNER_CPU",
    "OWNER_DEVICE",
    "STATUS_OK",
    "STATUS_ERROR",
    "END",
    "MAX_CAPACITY",
    "MAX_MEMORY_SIZE",
    "ChainPointer",
    "DescriptorSnapshot",
    "Completion",
    "ChainResult",
    "CoherentMemory",
    "DmaRing",
    "SyntheticDevice",
    "ChainError",
]
