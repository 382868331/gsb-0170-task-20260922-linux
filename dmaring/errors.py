"""Error taxonomy for the dmaring package.

Every error raised by the package derives from :class:`DmaRingError`, so
callers can catch one base class.  More specific subclasses let tests and
drivers pinpoint the failure class:

- :class:`ValidationError`     -- malformed input values (type/range).
- :class:`CapacityError`       -- ring capacity violates the power-of-two
                                  and <= 64 contract.
- :class:`OwnershipError`      -- an operation was attempted while the
                                  descriptor was owned by the other side.
- :class:`VisibilityError`     -- a field was read without the required
                                  acquire/flush event.
- :class:`StaleGenerationError`-- device tried to acquire a generation it
                                  already consumed (ABA guard).
- :class:`ChainError`          -- a scatter/gather chain is malformed
                                  (loop, unknown ring, ...).  Reported inside
                                  ChainResult, and raised only for misuse of
                                  the device API itself.
- :class:`DeviceHaltedError`   -- process_chain called after a chain error
                                  halted the device.
"""


class DmaRingError(Exception):
    """Base class for all errors raised by dmaring."""


class ValidationError(DmaRingError, ValueError):
    """An input value failed validation (wrong type, out of range, ...).

    The message always names the offending parameter so the call site can
    be located.  The package never silently corrects bad input.
    """


class CapacityError(ValidationError):
    """Ring capacity is not a power of two within [1, 64]."""


class OwnershipError(DmaRingError):
    """Operation requires the descriptor to be owned by the caller's side."""


class VisibilityError(DmaRingError):
    """Read of a field whose snapshot was not made visible to the reader.

    Raised when the CPU reads completion fields without a preceding
    ``cpu_acquire`` for the current ownership cycle, or when no completion
    exists at all.
    """


class StaleGenerationError(DmaRingError):
    """Device attempted to acquire a descriptor generation it already saw.

    This is the ABA guard: after ring wrap-around a slot is reused with the
    same field values, but the generation counter differs, so a stale
    re-acquire of an old generation is rejected.
    """


class ChainError(DmaRingError):
    """A scatter/gather chain is malformed (loop, dangling pointer, ...)."""


class DeviceHaltedError(DmaRingError):
    """The device halted after an error chain and refuses further chains."""
