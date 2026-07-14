"""Rules-derived per-seat kyoku score deltas."""

from __future__ import annotations


def rewards(before_scores, after_scores):
    if len(before_scores) != 4 or len(after_scores) != 4: raise ValueError("kyoku scores require four seats")
    return tuple((int(after) - int(before)) / 1000.0 for before, after in zip(before_scores, after_scores))


def deduplicate_settlements(events):
    seen, result = set(), []
    for event in events:
        delta = tuple(event.get("settlement", ()))
        if delta and delta not in seen:
            seen.add(delta); result.append(delta)
    return result

