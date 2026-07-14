"""Shanten-first, public-ukeire discard regret."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DiscardScore:
    shanten: int
    ukeire: int


def public_remaining(actor_counts, improving_mask: int, public_physical_tiles=()):
    visible = list(map(int, actor_counts))
    seen = set()
    for physical in public_physical_tiles:
        physical = int(physical)
        if 0 <= physical < 136 and physical not in seen:
            visible[physical // 4] += 1; seen.add(physical)
    return sum(max(0, 4 - visible[tile]) for tile in range(34) if improving_mask & (1 << tile))


def regret(scores, chosen: int) -> float:
    if not 0 <= chosen < len(scores) or not scores: raise ValueError("invalid chosen discard")
    best_shanten = min(score.shanten for score in scores)
    selected = scores[chosen]
    shanten_regret = selected.shanten - best_shanten
    ukeire_regret = 0
    if shanten_regret == 0:
        ukeire_regret = max(score.ukeire for score in scores if score.shanten == best_shanten) - selected.ukeire
    return -float(shanten_regret) - float(ukeire_regret) / 137.0

