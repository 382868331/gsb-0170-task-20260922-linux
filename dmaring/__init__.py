"""dmaring -- synthetic DMA ring descriptor ownership handoff.

A reusable, hardware-free model of DMA descriptor ring ownership:
CPU writes fields and releases to the device; the device acquires, copies,
and releases back; the CPU must acquire before reading completion fields.
Visibility is modeled with explicit flush/acquire events.

Public API is re-exported here; see :mod:`dmaring.core`,
:mod:`dmaring.errors` and :mod:`dmaring.reference`.
"""

from .core import (
    END,
    MAX_CAPACITY,
    MAX_MEMORY_SIZE,
    OWNER_CPU,
    OWNER_DEVICE,
    STATUS_ERROR,
    STATUS_OK,
    ChainPointer,
    ChainResult,
    CoherentMemory,
    Completion,
    DescriptorSnapshot,
    DmaRing,
    SyntheticDevice,
)
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
from .reference import reference_gather_copy

__version__ = "1.0.0"

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
    "reference_gather_copy",
    "DmaRingError",
    "ValidationError",
    "CapacityError",
    "OwnershipError",
    "VisibilityError",
    "StaleGenerationError",
    "ChainError",
    "DeviceHaltedError",
    "__version__",
]
