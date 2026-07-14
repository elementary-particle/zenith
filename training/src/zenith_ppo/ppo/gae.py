"""Explicit-link GAE for sparse per-seat trajectories."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class AdvantageBatch:
    indices: np.ndarray
    advantages: np.ndarray
    normalized: np.ndarray
    returns: np.ndarray


def compute(samples, *, gamma: float, gae_lambda: float) -> AdvantageBatch:
    eligible = [index for index, sample in enumerate(samples) if sample.ppo_eligible]
    raw = np.zeros(len(samples), dtype=np.float32)
    returns = np.zeros(len(samples), dtype=np.float32)
    for index in reversed(eligible):
        sample = samples[index]
        boundary = sample.match_boundary if sample.reward.boundary_mode == "match" else sample.kyoku_boundary
        if sample.terminal or boundary:
            bootstrap = trace = 0.0
        elif sample.successor is not None:
            successor = samples[sample.successor]
            bootstrap, trace = float(successor.old_value), float(raw[sample.successor])
        elif sample.truncated:
            bootstrap, trace = float(sample.bootstrap_value), 0.0
        else:
            bootstrap = trace = 0.0
        delta = sample.reward.total + gamma * bootstrap - sample.old_value
        raw[index] = delta + gamma * gae_lambda * trace
        returns[index] = raw[index] + sample.old_value
    values = raw[eligible]
    if len(values) == 0:
        normalized = values.copy()
    else:
        std = values.std(dtype=np.float64)
        normalized = np.zeros_like(values) if std < 1e-8 else (values - values.mean(dtype=np.float64)) / (std + 1e-8)
    return AdvantageBatch(np.asarray(eligible, dtype=np.int64), values, normalized.astype(np.float32),
                          returns[eligible])

