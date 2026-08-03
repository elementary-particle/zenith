"""Lambda-one Q-boosting over trajectories that stop at a kyoku boundary."""

from __future__ import annotations

import numpy as np


def lambda_one_advantages(
    endpoint_values: np.ndarray,
    selected_q_values: np.ndarray,
    expected_q_values: np.ndarray,
    segments: tuple[tuple[int, ...], ...],
) -> np.ndarray:
    """Return the Expected-SARSA(1) estimator for independent kyoku segments.

    ``endpoint_values`` contains the realized next-boundary rank potential G.
    Q predicts that absolute potential, and E[Q] uses the current policy over
    every legal action.  The reverse sum stops at each supplied segment, so no
    control variate can leak into the next kyoku.
    """
    endpoint_values = np.asarray(endpoint_values, dtype=np.float64)
    selected_q_values = np.asarray(selected_q_values, dtype=np.float64)
    expected_q_values = np.asarray(expected_q_values, dtype=np.float64)
    if not (
        endpoint_values.shape
        == selected_q_values.shape
        == expected_q_values.shape
    ):
        raise ValueError("endpoint, selected-Q, and expected-Q shapes differ")
    if endpoint_values.ndim != 1:
        raise ValueError("Q-boosting inputs must be one-dimensional")
    if not all(
        np.isfinite(values).all()
        for values in (endpoint_values, selected_q_values, expected_q_values)
    ):
        raise FloatingPointError("Q-boosting inputs must be finite")

    result = np.full(endpoint_values.shape, np.nan, dtype=np.float64)
    seen = np.zeros(endpoint_values.shape, dtype=np.bool_)
    for segment in segments:
        future_centered_q = 0.0
        for raw_index in reversed(segment):
            index = int(raw_index)
            if not 0 <= index < len(result) or seen[index]:
                raise ValueError("Q-boosting segments overlap or leave the domain")
            result[index] = (
                endpoint_values[index]
                - expected_q_values[index]
                - future_centered_q
            )
            future_centered_q += (
                selected_q_values[index] - expected_q_values[index]
            )
            seen[index] = True
    if len(result) and not bool(seen.all()):
        raise ValueError("Q-boosting segments do not cover every input row")
    return result.astype(np.float32)


def kyoku_segments(samples, eligible_indices) -> tuple[tuple[int, ...], ...]:
    """Map collector rows to dense, seat-local segments ending at each kyoku."""
    eligible_indices = tuple(map(int, eligible_indices))
    dense_by_sample = {
        sample_index: dense_index
        for dense_index, sample_index in enumerate(eligible_indices)
    }
    trajectories: dict[tuple[int, int, int, str], list[int]] = {}
    for sample_index in eligible_indices:
        sample = samples[sample_index]
        binding = sample.binding
        key = (
            int(binding.environment_id),
            int(binding.episode_generation),
            int(binding.seat),
            str(sample.checkpoint_id),
        )
        trajectories.setdefault(key, []).append(sample_index)

    result = []
    for indices in trajectories.values():
        current = []
        for sample_index in indices:
            current.append(dense_by_sample[sample_index])
            if bool(samples[sample_index].kyoku_boundary):
                result.append(tuple(current))
                current = []
        if current:
            raise ValueError("eligible trajectory ended without a kyoku boundary")
    return tuple(result)
