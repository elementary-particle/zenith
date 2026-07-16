"""Explicit-link GAE for sparse per-seat trajectories."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class AdvantageBatch:
    indices: np.ndarray
    score_advantages: np.ndarray
    rank_advantages: np.ndarray
    advantages: np.ndarray
    normalized: np.ndarray
    score_returns: np.ndarray
    rank_returns: np.ndarray


def compute(
    samples,
    *,
    gamma: float,
    score_gae_lambda: float,
    rank_gae_lambda: float,
) -> AdvantageBatch:
    if float(gamma) != 1.0:
        raise ValueError("choice-only complete-match trajectories require gamma = 1")
    if not 0 <= float(score_gae_lambda) <= 1:
        raise ValueError("score GAE lambda must be in [0,1]")
    if float(rank_gae_lambda) != 1.0:
        raise ValueError("rank GAE lambda must equal 1 for exact match-long returns")
    eligible = [index for index, sample in enumerate(samples) if sample.ppo_eligible]
    score_raw = np.zeros(len(samples), dtype=np.float32)
    rank_raw = np.zeros(len(samples), dtype=np.float32)
    score_returns = np.zeros(len(samples), dtype=np.float32)
    rank_returns = np.zeros(len(samples), dtype=np.float32)
    weights = None
    for index in reversed(eligible):
        sample = samples[index]
        if not 0 <= int(sample.terminal_placement) < 4:
            raise ValueError("learner rollout sample lacks a resolved placement class")
        if not all(np.isfinite(value) for value in (
            sample.old_score_value, sample.old_rank_value,
            sample.reward.kyoku_delta, sample.reward.rank_reward,
        )):
            raise FloatingPointError("non-finite rollout value or reward component")
        sample_weights = tuple(float(value) for value in sample.reward.weights)
        if weights is None:
            weights = sample_weights
        elif sample_weights != weights:
            raise ValueError("one GAE batch cannot mix curriculum weights")
        if (sample.terminal or sample.match_boundary) and sample.successor is not None:
            raise ValueError("terminal rollout sample cannot have a successor")
        if sample.truncated and sample.successor is not None:
            raise ValueError("truncated rollout sample cannot have a successor")
        if sample.terminal or sample.match_boundary:
            score_bootstrap = rank_bootstrap = 0.0
            score_trace = rank_trace = 0.0
        elif sample.successor is not None:
            successor_index = int(sample.successor)
            if not index < successor_index < len(samples):
                raise ValueError("rollout successor must be a later sample")
            successor = samples[sample.successor]
            if not successor.ppo_eligible:
                raise ValueError("learner rollout successor must be PPO-eligible")
            current_key = (
                sample.binding.environment_id,
                sample.binding.episode_generation,
                sample.binding.seat,
            )
            successor_key = (
                successor.binding.environment_id,
                successor.binding.episode_generation,
                successor.binding.seat,
            )
            if successor_key != current_key:
                raise ValueError("rollout successor crosses a seat trajectory")
            score_bootstrap = (
                0.0 if sample.kyoku_boundary else float(successor.old_score_value)
            )
            rank_bootstrap = float(successor.old_rank_value)
            score_trace = (
                0.0 if sample.kyoku_boundary else float(score_raw[sample.successor])
            )
            rank_trace = float(rank_raw[sample.successor])
        elif sample.truncated:
            raise ValueError("complete-match trajectories cannot be truncated or bootstrapped")
        else:
            raise ValueError("nonterminal rollout tail is neither linked nor truncated")
        score_delta = (
            sample.reward.kyoku_delta + gamma * score_bootstrap - sample.old_score_value
        )
        rank_delta = (
            sample.reward.rank_reward + gamma * rank_bootstrap - sample.old_rank_value
        )
        score_raw[index] = score_delta + gamma * score_gae_lambda * score_trace
        rank_raw[index] = rank_delta + gamma * rank_gae_lambda * rank_trace
        score_returns[index] = score_raw[index] + sample.old_score_value
        rank_returns[index] = rank_raw[index] + sample.old_rank_value
        if not all(np.isfinite(value) for value in (
            score_raw[index], rank_raw[index], score_returns[index], rank_returns[index]
        )):
            raise FloatingPointError("non-finite GAE target")
    weights = weights or (1.0, 0.0)
    score_advantages = score_raw[eligible]
    rank_advantages = rank_raw[eligible]
    values = weights[0] * score_advantages + weights[1] * rank_advantages
    if len(values) == 0:
        normalized = values.copy()
    else:
        std = values.std(dtype=np.float64)
        normalized = np.zeros_like(values) if std < 1e-8 else (values - values.mean(dtype=np.float64)) / (std + 1e-8)
    return AdvantageBatch(
        np.asarray(eligible, dtype=np.int64),
        score_advantages,
        rank_advantages,
        values,
        normalized.astype(np.float32),
        score_returns[eligible],
        rank_returns[eligible],
    )


def explained_variance(predictions, targets) -> float:
    """Fraction of target variance explained by rollout-time value predictions."""
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if predictions.shape != targets.shape:
        raise ValueError("value predictions and targets must have the same shape")
    if not predictions.size:
        return 0.0
    target_variance = float(np.var(targets))
    if target_variance < 1e-12:
        return 0.0
    return float(1.0 - np.var(targets - predictions) / target_variance)
