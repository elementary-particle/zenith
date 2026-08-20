"""Held-out cyclic-seat evaluation and convergence summaries."""

from __future__ import annotations

from hashlib import sha256
from itertools import permutations
import json

from ..seeds import derive_seed


def cyclic_lineups(checkpoint_ids):
    ids = tuple(checkpoint_ids)
    if len(ids) != 4:
        raise ValueError("evaluation requires four policy identities")
    return tuple(ids[offset:] + ids[:offset] for offset in range(4))


def seat_balanced_lineups(checkpoint_ids):
    """Cover unique seat allocations without replaying repeated identities."""
    ids = tuple(checkpoint_ids)
    if len(ids) != 4:
        raise ValueError("evaluation requires four policy identities")
    if len(set(ids)) == 4:
        return cyclic_lineups(ids)
    return tuple(dict.fromkeys(permutations(ids)))


def run_series(checkpoint_ids, seeds, play_game, *, series_id="series-0"):
    def play_batch(requests):
        return tuple(
            play_game(
                request["checkpoint_ids"], request["seed"],
                ordinary_view=True,
                rank_only=True,
                gradients=False,
                action_seed=request["action_seed"],
            )
            for request in requests
        )

    return run_series_batched(
        checkpoint_ids, seeds, play_batch,
        batch_size=1, series_id=series_id,
    )


def series_requests(checkpoint_ids, seeds):
    """Materialize the stable seed/seat schedule used by every evaluator."""
    lineups = seat_balanced_lineups(checkpoint_ids)
    return tuple(
        {
            "game_id": game_id,
            "checkpoint_ids": block,
            "seed": int(seed),
            "rotation": rotation,
            "action_seed": derive_seed(
                int(seed), f"rank_evaluation_lineup_{rotation}"
            ),
        }
        for game_id, (seed, rotation, block) in enumerate(
            (seed, rotation, block)
            for seed in seeds
            for rotation, block in enumerate(lineups)
        )
    )


def _series_outcome(request, result, *, series_id):
    if isinstance(result, BaseException):
        ranks, scores, valid = (), (), False
        error = f"{type(result).__name__}: {result}"
        payload_result = {}
    else:
        payload_result = result
        ranks = tuple(result.get("ranks", ()))
        scores = tuple(result.get("scores", ()))
        valid = (
            bool(result.get("valid", False))
            and len(ranks) == 4
            and len(scores) == 4
            and sorted(ranks) == [0, 1, 2, 3]
        )
        error = None if valid else "invalid terminal outcome"
    payload = json.dumps(
        payload_result, sort_keys=True, default=list
    ).encode()
    outcome = {
        "series_id": series_id,
        "game_id": request["game_id"],
        "checkpoint_ids": request["checkpoint_ids"],
        "ranks": ranks,
        "scores": scores,
        "seed": request["seed"],
        "rotation": request["rotation"],
        "action_seed": request["action_seed"],
        "valid": valid,
        "error": error,
        "trajectory_digest": sha256(payload).hexdigest(),
    }
    if valid and "gameplay_counts" in payload_result:
        outcome["gameplay_counts"] = payload_result["gameplay_counts"]
    return outcome


def run_series_batched(
    checkpoint_ids, seeds, play_games, *, batch_size=32, series_id="series-0",
):
    """Evaluate seed/seat blocks while batching independent game inference.

    ``play_games`` receives immutable request dictionaries and must return one
    result (or exception) in the same order. Environment and action seeds stay
    per-game, so changing ``batch_size`` changes throughput rather than the
    statistical schedule.
    """
    batch_size = int(batch_size)
    if batch_size < 1:
        raise ValueError("evaluation batch size must be positive")
    requests = series_requests(checkpoint_ids, seeds)
    outcomes = []
    for start in range(0, len(requests), batch_size):
        batch = requests[start:start + batch_size]
        try:
            results = tuple(play_games(batch))
            if len(results) != len(batch):
                raise ValueError(
                    "batched evaluator returned the wrong result count"
                )
        except Exception as exc:
            results = (exc,) * len(batch)
        outcomes.extend(
            _series_outcome(request, result, series_id=series_id)
            for request, result in zip(batch, results, strict=True)
        )
    return outcomes


def paired_bootstrap(outcomes, candidate, reference, *, confidence=.95,
                     resamples=10_000, seed=0):
    """Paired, seed-block bootstrap for score and placement differences."""
    import numpy as np

    if not 0 < float(confidence) < 1 or int(resamples) < 1:
        raise ValueError("bootstrap confidence and resample count are invalid")
    blocks = {}
    for outcome in outcomes:
        blocks.setdefault(
            int(outcome.get("seed", outcome["game_id"])), []
        ).append(outcome)
    rotations = {
        int(outcome["rotation"])
        for outcome in outcomes if "rotation" in outcome
    }
    expected_rotations = set(range(max(rotations) + 1)) if rotations else None
    grouped = {}
    for block_seed, block in blocks.items():
        if not all(outcome.get("valid", False) for outcome in block):
            continue
        if expected_rotations is not None and (
            len(block) != len(expected_rotations)
            or {int(outcome["rotation"]) for outcome in block}
            != expected_rotations
        ):
            continue
        for outcome in block:
            ids = tuple(outcome["checkpoint_ids"])
            candidate_seats = [
                index for index, value in enumerate(ids) if value == candidate
            ]
            reference_seats = [
                index for index, value in enumerate(ids) if value == reference
            ]
            if not candidate_seats or not reference_seats:
                continue
            scores, ranks = outcome["scores"], outcome["ranks"]
            score = np.mean([
                scores[index] for index in candidate_seats
            ]) - np.mean([
                scores[index] for index in reference_seats
            ])
            placement = np.mean([
                ranks[index] for index in candidate_seats
            ]) - np.mean([
                ranks[index] for index in reference_seats
            ])
            pairwise = np.mean([
                float(ranks[candidate_seat] < ranks[reference_seat])
                + 0.5 * float(
                    ranks[candidate_seat] == ranks[reference_seat]
                )
                for candidate_seat in candidate_seats
                for reference_seat in reference_seats
            ])
            grouped.setdefault(block_seed, []).append(
                (float(score), float(placement), float(pairwise))
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
    placement_difference = summary(1)
    rank_advantage = {
        "mean": -placement_difference["mean"],
        "lower": -placement_difference["upper"],
        "upper": -placement_difference["lower"],
        "confidence": placement_difference["confidence"],
        "paired_seeds": placement_difference["paired_seeds"],
    }
    return {
        "rank_advantage": rank_advantage,
        "score_difference": summary(0, 1000.0),
        # Retain candidate-minus-reference placement for artifact compatibility.
        # Rank advantage is its sign-reversed, positive-is-better form.
        "placement_difference": placement_difference,
        "pairwise_win_rate": summary(2),
    }


def convergence(curves, threshold):
    import numpy as np
    crossings, areas = {}, {}
    for name, seeds in curves.items():
        points = [next((x for x, value in curve if value >= threshold), float("inf")) for curve in seeds]
        crossings[name] = float(np.median(points))
        areas[name] = [float(np.trapezoid([value for _, value in curve], [x for x, _ in curve])) for curve in seeds]
    return {"median_threshold_crossing": crossings, "areas": areas}
