"""Final placement-only terminal rewards."""

from __future__ import annotations

PLACEMENT_REWARD = (1.0, 1.0 / 3.0, -1.0 / 3.0, -1.0)


def rewards(ranks, *, terminal=True, valid=True):
    if not terminal or not valid: return (0.0, 0.0, 0.0, 0.0)
    if sorted(map(int, ranks)) != [0, 1, 2, 3]: raise ValueError("resolved ranks must be a permutation")
    return tuple(PLACEMENT_REWARD[int(rank)] for rank in ranks)

