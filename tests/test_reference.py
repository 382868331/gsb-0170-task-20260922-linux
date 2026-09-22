"""Reference implementation tests and core/reference round-trip checks.

The reference (dmaring.reference) is standalone: it never calls the core.
These tests pin the reference's own behavior on hand-computed expectations,
then check that the synthetic device agrees with it on generated inputs.
"""

import unittest

from dmaring import (
    CoherentMemory,
    DmaRing,
    SyntheticDevice,
    reference_gather_copy,
)


def make_source(size=256):
    return bytes(((i * 31 + 7) ^ (i >> 1)) & 0xFF for i in range(size))


class TestReferenceStandalone(unittest.TestCase):
    def test_hand_computed(self):
        src = bytes(range(16))
        out = reference_gather_copy(src, [[(0, 4), (8, 2)], [(3, 3)]])
        self.assertEqual(out, bytes([0, 1, 2, 3, 8, 9, 3, 4, 5]))

    def test_empty(self):
        self.assertEqual(reference_gather_copy(b"abc", []), b"")
        self.assertEqual(reference_gather_copy(b"abc", [[]]), b"")
        self.assertEqual(reference_gather_copy(b"abc", [[(1, 0)]]), b"")

    def test_strict_validation(self):
        src = b"abcd"
        with self.assertRaises(ValueError):
            reference_gather_copy(src, [[(0, 5)]])       # out of range
        with self.assertRaises(ValueError):
            reference_gather_copy(src, [[(-1, 2)]])      # negative
        with self.assertRaises(ValueError):
            reference_gather_copy(src, [[(True, 1)]])    # bool
        with self.assertRaises(ValueError):
            reference_gather_copy(src, [[(0, 1.5)]])     # float
        with self.assertRaises(ValueError):
            reference_gather_copy(src, [[(0,)]])         # malformed item
        with self.assertRaises(ValueError):
            reference_gather_copy("abcd", [[(0, 1)]])    # non-bytes source


class TestRoundTrip(unittest.TestCase):
    """Core device output must equal the independent reference on the
    same deterministic inputs (round-trip consistency, supplementary to
    the hand-computed pins above)."""

    def _run_core(self, src, chains_specs):
        """Drive the core through the legal handoff order.

        chains_specs: list of chains; each chain is a list of
        (offset, length) pairs.  Descriptors are laid out across two rings
        (alternating) so cross-ring links are exercised.
        """
        mem = CoherentMemory(len(src))
        mem.cpu_write(0, src)
        mem.flush()
        rings = {0: DmaRing(2, ring_id=0), 1: DmaRing(2, ring_id=1)}
        dev = SyntheticDevice(mem)
        results = []
        n = 0  # global descriptor counter so slots wrap across chains
        for chain in chains_specs:
            # Assign slots alternating between the two rings, wrapping
            # around each ring's capacity of 2.
            ptrs = [((n + i) % 2, ((n + i) // 2) % 2)
                    for i in range(len(chain))]
            n += len(chain)
            for (rid, slot), (offset, length), nxt in zip(
                    ptrs, chain, ptrs[1:] + [None]):
                rings[rid].cpu_write(slot, offset=offset, length=length,
                                     next=nxt)
            for rid, slot in ptrs:
                rings[rid].cpu_release(slot)
            result = dev.process_chain(rings, ptrs[0] if ptrs else None)
            results.append(result)
            for rid, slot in ptrs:
                rings[rid].cpu_acquire(slot)
        return bytes(dev.output), results, dev

    def test_single_chain(self):
        src = make_source(256)
        chains = [[(0, 16), (32, 8), (100, 12), (200, 1)]]
        out, results, dev = self._run_core(src, chains)
        self.assertEqual(out, reference_gather_copy(src, chains))
        self.assertTrue(all(r.status == "ok" for r in results))

    def test_multiple_chains_and_wraparound(self):
        src = make_source(256)
        # 12 single-descriptor chains over two capacity-2 rings: slots wrap
        # around, so each of the 4 slots is reused 3 times.
        chains = [[(i * 16 % 256, 7)] for i in range(12)]
        out, results, dev = self._run_core(src, chains)
        self.assertEqual(out, reference_gather_copy(src, chains))
        for ring_id in (0, 1):
            for slot in (0, 1):
                self.assertEqual(dev.completion_count(ring_id, slot), 3)

    def test_zero_length_descriptors(self):
        src = make_source(64)
        chains = [[(0, 0), (10, 5), (63, 1), (0, 0)]]
        out, results, _ = self._run_core(src, chains)
        self.assertEqual(out, reference_gather_copy(src, chains))
        self.assertEqual(out, src[10:15] + src[63:64])


if __name__ == "__main__":
    unittest.main()
