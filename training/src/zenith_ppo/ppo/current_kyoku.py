"""Dense current-kyoku rank-potential advantages."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


RANK_UTILITIES = np.asarray((1.0, 1 / 3, -1 / 3, -1.0), dtype=np.float32)


@dataclass(frozen=True)
class AdvantageBatch:
    indices: np.ndarray
    advantages: np.ndarray
    normalized: np.ndarray
    start_values: np.ndarray
    end_values: np.ndarray


def _standardize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if not len(values):
        return values
    standard_deviation = float(values.std(dtype=np.float64))
    if standard_deviation < 1e-8:
        return np.zeros_like(values)
    return np.asarray(
        (values - values.mean(dtype=np.float64))
        / (standard_deviation + 1e-8),
        dtype=np.float32,
    )


def _trajectory_key(row):
    binding = row.binding
    return (
        int(binding.environment_id),
        int(binding.episode_generation),
        int(binding.seat),
        str(row.checkpoint_id),
    )


def _boundary_identity(frame):
    features = np.asarray(frame.encoded.rank_boundary_features, dtype=np.float32)
    if features.shape != (28,):
        raise ValueError("current-kyoku frame has invalid boundary features")
    return features.tobytes()


def _frame_potentials(frames):
    """Map every nonterminal public frame to its kyoku start/end potential."""
    trajectories = {}
    for index, frame in enumerate(frames):
        if frame.ppo_eligible:
            trajectories.setdefault(_trajectory_key(frame), []).append(index)

    potentials = {}
    for trajectory, indices in trajectories.items():
        placements = {
            int(frames[index].terminal_placement)
            for index in indices
            if 0 <= int(frames[index].terminal_placement) < 4
        }
        if len(placements) != 1:
            raise ValueError(
                "current-kyoku trajectory lacks one terminal placement"
            )
        terminal_value = float(RANK_UTILITIES[placements.pop()])
        segments = []
        for index in indices:
            frame = frames[index]
            identity = _boundary_identity(frame)
            if not segments or segments[-1][0] != identity:
                segments.append((identity, [index]))
            else:
                segments[-1][1].append(index)
        for segment_index, (_, frame_indices) in enumerate(segments):
            values = np.asarray([
                float(frames[index].old_boundary_rank_value)
                for index in frame_indices
            ], dtype=np.float64)
            if not np.isfinite(values).all():
                raise FloatingPointError(
                    "current-kyoku boundary value is non-finite"
                )
            if float(np.ptp(values)) > 1e-6:
                raise ValueError(
                    "boundary V changed within one kyoku trajectory"
                )
            start = float(values[0])
            if segment_index + 1 < len(segments):
                next_frame = segments[segment_index + 1][1][0]
                end = float(frames[next_frame].old_boundary_rank_value)
            else:
                end = terminal_value
            if not np.isfinite(end):
                raise FloatingPointError(
                    "current-kyoku end potential is non-finite"
                )
            for index in frame_indices:
                potentials[index] = (start, end)
    return potentials


def compute(samples, frames) -> AdvantageBatch:
    """Assign every action the expected rank-utility change of its kyoku."""
    samples, frames = tuple(samples), tuple(frames)
    indices = np.asarray(
        [index for index, sample in enumerate(samples) if sample.ppo_eligible],
        dtype=np.int64,
    )
    if not len(indices):
        empty = np.zeros(0, dtype=np.float32)
        return AdvantageBatch(indices, empty, empty, empty, empty)
    potentials = _frame_potentials(frames)
    starts, ends = [], []
    for sample_index in map(int, indices):
        sample = samples[sample_index]
        frame_index = int(sample.frame_index)
        if not 0 <= frame_index < len(frames):
            raise ValueError("current-kyoku action lacks a public frame")
        frame = frames[frame_index]
        if _trajectory_key(sample) != _trajectory_key(frame):
            raise ValueError("current-kyoku action/frame trajectory mismatch")
        try:
            start, end = potentials[frame_index]
        except KeyError as error:
            raise ValueError(
                "current-kyoku action is bound to a terminal frame"
            ) from error
        starts.append(start)
        ends.append(end)
    start_values = np.asarray(starts, dtype=np.float32)
    end_values = np.asarray(ends, dtype=np.float32)
    advantages = np.asarray(end_values - start_values, dtype=np.float32)
    return AdvantageBatch(
        indices,
        advantages,
        _standardize(advantages),
        start_values,
        end_values,
    )


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


def action_family_statistics(samples, advantages) -> dict[str, float]:
    """Return additive action-family counts without outcome conditioning."""
    counts = {"call": 0, "pass": 0, "riichi": 0, "dama": 0}
    call_kinds = {3, 4, 5}
    for sample_index in map(int, advantages.indices):
        sample = samples[sample_index]
        factors = np.asarray(sample.encoded.action_factors)
        if factors.ndim != 2 or not len(factors):
            continue
        kinds = set(map(int, factors[:, 0]))
        selected = int(factors[int(sample.selected_group), 0])
        if 0 in kinds and kinds & call_kinds:
            if selected in call_kinds:
                counts["call"] += 1
            elif selected == 0:
                counts["pass"] += 1
        if 1 in kinds and 2 in kinds:
            if selected == 2:
                counts["riichi"] += 1
            elif selected == 1:
                counts["dama"] += 1
    return {
        f"{name}_count": float(count) for name, count in counts.items()
    }


def action_family_diagnostics(samples, advantages) -> dict[str, float]:
    statistics = action_family_statistics(samples, advantages)

    def rate(left, right):
        total = statistics[f"{left}_count"] + statistics[f"{right}_count"]
        return statistics[f"{left}_count"] / total if total else 0.0

    return {
        "rollout/call_opportunity_selected_call_rate": rate("call", "pass"),
        "rollout/riichi_opportunity_selected_riichi_rate": rate(
            "riichi", "dama"
        ),
    }
