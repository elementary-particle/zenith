"""Update-based convex reward curriculum."""

from __future__ import annotations

from ..types import CurriculumSnapshot


def _blend(progress, start, end):
    if end <= start: return float(progress >= end)
    return min(1.0, max(0.0, (progress - start) / (end - start)))


class Curriculum:
    def __init__(self, config): self.config = dict(config)
    def snapshot(self, update: int, policy_version: int) -> CurriculumSnapshot:
        total = max(1, int(self.config["total_updates"]))
        progress = min(max(update / total, 0.0), 1.0)
        a, b, c, d = (self.config[key] for key in ("discard_only_end",
            "discard_kyoku_blend_end", "kyoku_only_end", "kyoku_rank_blend_end"))
        if progress <= a: weights = (1.0, 0.0, 0.0)
        elif progress < b:
            incoming = _blend(progress, a, b); weights = (1 - incoming, incoming, 0.0)
        elif progress <= c: weights = (0.0, 1.0, 0.0)
        elif progress < d:
            incoming = _blend(progress, c, d); weights = (0.0, 1 - incoming, incoming)
        else: weights = (0.0, 0.0, 1.0)
        return CurriculumSnapshot(update, progress, weights,
            "match" if weights[2] > 0 else "kyoku", policy_version)
