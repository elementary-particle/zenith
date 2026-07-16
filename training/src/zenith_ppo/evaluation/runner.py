"""Held-out cyclic-seat evaluation and convergence summaries."""

from __future__ import annotations

from hashlib import sha256
import json


def cyclic_lineups(checkpoint_ids):
    ids = tuple(checkpoint_ids)
    if len(ids) != 4:
        raise ValueError("evaluation requires four policy identities")
    return tuple(ids[offset:] + ids[:offset] for offset in range(4))


def run_series(checkpoint_ids, seeds, play_game, *, series_id="series-0"):
    lineups = cyclic_lineups(checkpoint_ids)
    outcomes = []
    for seed in seeds:
        for rotation, block in enumerate(lineups):
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
            outcomes.append({"series_id": series_id, "game_id": len(outcomes),
                "checkpoint_ids": block, "ranks": ranks, "scores": scores,
                "seed": int(seed), "rotation": rotation,
                "valid": valid, "error": error,
                "trajectory_digest": sha256(payload).hexdigest()})
    return outcomes


def paired_bootstrap(outcomes, candidate, reference, *, confidence=.95,
                     resamples=10_000, seed=0):
    """Paired, seed-block bootstrap for score and placement differences."""
    import numpy as np

    if not 0 < float(confidence) < 1 or int(resamples) < 1:
        raise ValueError("bootstrap confidence and resample count are invalid")
    grouped = {}
    for outcome in outcomes:
        if not outcome.get("valid", False):
            continue
        ids = tuple(outcome["checkpoint_ids"])
        candidate_seats = [index for index, value in enumerate(ids) if value == candidate]
        reference_seats = [index for index, value in enumerate(ids) if value == reference]
        if not candidate_seats or not reference_seats:
            continue
        scores, ranks = outcome["scores"], outcome["ranks"]
        score = np.mean([scores[index] for index in candidate_seats]) - np.mean(
            [scores[index] for index in reference_seats]
        )
        placement = np.mean([ranks[index] for index in candidate_seats]) - np.mean(
            [ranks[index] for index in reference_seats]
        )
        grouped.setdefault(int(outcome.get("seed", outcome["game_id"])), []).append(
            (float(score), float(placement))
        )
    if not grouped:
        raise ValueError("no valid paired outcomes contain both policies")
    pairs = np.asarray([
        np.mean(rows, axis=0) for _, rows in sorted(grouped.items())
    ], dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    draws = rng.integers(0, len(pairs), size=(int(resamples), len(pairs)))
    estimates = pairs[draws].mean(axis=1)
    tail = (1.0 - float(confidence)) / 2.0

    def summary(column, scale=1.0):
        values = estimates[:, column] / scale
        return {
            "mean": float(pairs[:, column].mean() / scale),
            "lower": float(np.quantile(values, tail)),
            "upper": float(np.quantile(values, 1.0 - tail)),
            "confidence": float(confidence),
            "paired_seeds": int(len(pairs)),
        }
    return {
        "score_difference": summary(0, 1000.0),
        "placement_difference": summary(1),
    }


def full_design_selected(full, outcome_only, paired_interval):
    """Apply the documented short-ablation promotion gates."""
    def reduction(name):
        baseline = float(outcome_only[name])
        return baseline > 0 and float(full[name]) <= .75 * baseline

    return all((
        reduction("discard_worse_shanten_rate"),
        reduction("discard_mean_cost"),
        float(full["calls_per_kyoku"]) <= .70 * float(outcome_only["calls_per_kyoku"]),
        float(full["deal_ins_after_opponent_riichi_per_kyoku"]) <=
            float(outcome_only["deal_ins_after_opponent_riichi_per_kyoku"]) + .02,
        float(full["riichi_conversion_rate"]) > 0,
        float(full["riichi_conversion_rate"]) > float(outcome_only["riichi_conversion_rate"]),
        float(paired_interval["placement_difference"]["lower"]) <= .10,
    ))


def convergence(curves, threshold):
    import numpy as np
    crossings, areas = {}, {}
    for name, seeds in curves.items():
        points = [next((x for x, value in curve if value >= threshold), float("inf")) for curve in seeds]
        crossings[name] = float(np.median(points))
        areas[name] = [float(np.trapezoid([value for _, value in curve], [x for x, _ in curve])) for curve in seeds]
    return {"median_threshold_crossing": crossings, "areas": areas}
