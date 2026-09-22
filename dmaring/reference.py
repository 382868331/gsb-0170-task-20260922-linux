"""Independent reference implementation: plain synchronous gather copy.

This module deliberately does NOT import or call the dmaring core.  It
computes the expected byte output of a set of scatter/gather chains by
walking plain ``(offset, length)`` specs over a plain ``bytes`` buffer, in
the legal handoff order (chain by chain, descriptor by descriptor).  Tests
compare the synthetic device's output against this reference; agreement is a
round-trip consistency check, not a self-fulfilling one.
"""

from __future__ import annotations


def reference_gather_copy(source: bytes, chains) -> bytes:
    """Copy the bytes described by ``chains`` out of ``source``, in order.

    ``chains`` is an iterable of chains; each chain is an iterable of
    ``(offset, length)`` integer pairs.  Returns the concatenation of
    ``source[offset:offset + length]`` for every item, chain by chain.

    Raises ``ValueError`` for malformed input (bool/float numbers, negative
    values, ranges outside ``source``).  The reference validates strictly
    because it models the *legal* path only.
    """
    if not isinstance(source, (bytes, bytearray, memoryview)):
        raise ValueError(
            f"source: expected bytes-like, got {type(source).__name__}"
        )
    src = bytes(source)
    out = bytearray()
    for ci, chain in enumerate(chains):
        for di, item in enumerate(chain):
            where = f"chains[{ci}][{di}]"
            if not (isinstance(item, tuple) and len(item) == 2):
                raise ValueError(
                    f"{where}: expected (offset, length) tuple, got {item!r}"
                )
            offset, length = item
            for name, value in (("offset", offset), ("length", length)):
                if isinstance(value, bool) or not isinstance(value, int):
                    raise ValueError(
                        f"{where}.{name}: expected int, got "
                        f"{type(value).__name__} ({value!r})"
                    )
                if value < 0:
                    raise ValueError(f"{where}.{name}: negative value {value}")
            if offset + length > len(src):
                raise ValueError(
                    f"{where}: range [{offset}, {offset + length}) exceeds "
                    f"source size {len(src)}"
                )
            out += src[offset:offset + length]
    return bytes(out)
