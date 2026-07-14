"""Held-out cyclic-seat evaluation and convergence summaries."""

from __future__ import annotations

from hashlib import sha256
import json


def cyclic_lineups(checkpoint_ids):
    ids = tuple(checkpoint_ids)
    if len(ids) != 4 or len(set(ids)) != 4: raise ValueError("evaluation requires four distinct checkpoints")
    return tuple(ids[offset:] + ids[:offset] for offset in range(4))


def run_series(checkpoint_ids, seeds, play_game, *, series_id="series-0"):
    outcomes = []
    for block, seed in zip(cyclic_lineups(checkpoint_ids), seeds, strict=True):
        try:
            result = play_game(
                block, int(seed), ordinary_view=True, rank_only=True, gradients=False
            )
            ranks = tuple(result.get("ranks", ()))
            scores = tuple(result.get("scores", ()))
            valid = (
                bool(result.get("valid", False))
                and len(ranks) == 4
                and len(scores) == 4
                and sorted(ranks) == [0, 1, 2, 3]
            )
            error = None if valid else "invalid terminal outcome"
        except Exception as exc:
            result, ranks, scores, valid = {}, (), (), False
            error = f"{type(exc).__name__}: {exc}"
        payload = json.dumps(result, sort_keys=True, default=list).encode()
        outcomes.append({"series_id": series_id, "game_id": len(outcomes), "checkpoint_ids": block,
            "ranks": ranks, "scores": scores, "valid": valid, "error": error,
            "trajectory_digest": sha256(payload).hexdigest()})
    return outcomes


def convergence(curves, threshold):
    import numpy as np
    crossings, areas = {}, {}
    for name, seeds in curves.items():
        points = [next((x for x, value in curve if value >= threshold), float("inf")) for curve in seeds]
        crossings[name] = float(np.median(points))
        areas[name] = [float(np.trapezoid([value for _, value in curve], [x for x, _ in curve])) for curve in seeds]
    return {"median_threshold_crossing": crossings, "areas": areas}
