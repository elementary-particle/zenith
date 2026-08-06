"""Rank-utility constants and compact critic diagnostics."""

import numpy as np


RANK_UTILITIES = np.asarray((1.0, 1 / 3, -1 / 3, -1.0), dtype=np.float32)


def explained_variance(predictions, targets) -> float:
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if predictions.shape != targets.shape:
        raise ValueError("predictions and targets must have the same shape")
    if not predictions.size:
        return 0.0
    target_variance = float(np.var(targets))
    if target_variance < 1e-12:
        return 0.0
    return float(1.0 - np.var(targets - predictions) / target_variance)


def rank_explained_variance(predictions, placements) -> float:
    placements = np.asarray(placements, dtype=np.int64)
    if bool(((placements < 0) | (placements > 3)).any()):
        raise ValueError("rank explained variance requires placements in 0..3")
    return explained_variance(predictions, RANK_UTILITIES[placements])
