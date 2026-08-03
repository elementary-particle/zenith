"""Exact finite-game audit of GAE and Q-boosting bias/variance claims.

The game is deliberately Mahjong-shaped rather than a claim of Mahjong skill:
four stochastic seats act over a finite hand, transitions are deterministic once
an action is selected, and the terminal payoff uses Zenith's rank utilities.
Dynamic programming supplies exact V, Q, and the exact root policy gradient.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


RANK_UTILITIES = np.asarray((-1.0, -1 / 3, 1 / 3, 1.0), dtype=np.float64)


class FiniteRankGame:
    """A small stochastic self-play hand with an exactly enumerable critic."""

    def __init__(self, horizon: int = 12):
        self.horizon = int(horizon)
        if self.horizon < 2:
            raise ValueError("horizon must contain a root and a future action")
        self.positive = np.asarray(
            [2 if step % 4 == 0 else 1 for step in range(horizon)],
            dtype=np.int64,
        )
        self.negative = np.asarray(
            [-1 if step % 3 else -2 for step in range(horizon)],
            dtype=np.int64,
        )
        self.radius = int(
            np.maximum(np.abs(self.positive), np.abs(self.negative)).sum() + 4
        )
        self.scores = np.arange(-self.radius, self.radius + 1, dtype=np.int64)
        self.offset = self.radius
        self.probabilities = np.stack([
            self._policy_probability(step, self.scores)
            for step in range(horizon)
        ])
        self.q = np.zeros((horizon, len(self.scores), 2), dtype=np.float64)
        self.v = np.zeros((horizon + 1, len(self.scores)), dtype=np.float64)
        self._solve()

    @staticmethod
    def _sigmoid(value):
        return 1.0 / (1.0 + np.exp(-value))

    def _policy_probability(self, step, score):
        # All four seats mix. The root probability is constant so its scalar
        # logit gradient has a closed-form exact reference.
        if step == 0:
            return np.full_like(np.asarray(score, dtype=np.float64), 0.43)
        seat_offsets = (-0.20, 0.15, -0.10, 0.05)
        logit = seat_offsets[step % 4] + 0.11 * np.clip(score, -6, 6)
        logit += 0.18 * math.sin(0.7 * step)
        return self._sigmoid(logit)

    @staticmethod
    def _utility(score):
        rank = np.select(
            [score <= -3, score <= 0, score <= 3],
            [0, 1, 2],
            default=3,
        )
        return RANK_UTILITIES[rank]

    def _score_index(self, score):
        index = np.asarray(score, dtype=np.int64) + self.offset
        if bool(((index < 0) | (index >= len(self.scores))).any()):
            raise IndexError("score escaped the exact dynamic-programming table")
        return index

    def _solve(self):
        for step in reversed(range(self.horizon)):
            for action, increment in enumerate(
                (self.negative[step], self.positive[step])
            ):
                next_score = self.scores + int(increment)
                valid = (next_score >= -self.radius) & (next_score <= self.radius)
                if step == self.horizon - 1:
                    self.q[step, valid, action] = self._utility(next_score[valid])
                else:
                    next_index = self._score_index(next_score[valid])
                    self.q[step, valid, action] = self.v[step + 1, next_index]
            probability = self.probabilities[step]
            self.v[step] = (
                (1.0 - probability) * self.q[step, :, 0]
                + probability * self.q[step, :, 1]
            )

    def approximate_q(self, maximum_error: float):
        """Return a deterministic critic with a controlled sup-norm error."""
        step = np.arange(self.horizon, dtype=np.float64)[:, None, None]
        score = self.scores.astype(np.float64)[None, :, None]
        action = np.arange(2, dtype=np.float64)[None, None, :]
        pattern = np.sin(0.71 * (step + 1) + 0.37 * score + 1.13 * action)
        pattern /= np.max(np.abs(pattern))
        return self.q + float(maximum_error) * pattern

    def state_values_from_q(self, q_values):
        probability = self.probabilities
        values = np.zeros_like(self.v)
        values[:-1] = (
            (1.0 - probability) * q_values[:, :, 0]
            + probability * q_values[:, :, 1]
        )
        return values

    def sample(self, count: int, seed: int):
        rng = np.random.default_rng(seed)
        scores = np.zeros((count, self.horizon + 1), dtype=np.int16)
        actions = np.zeros((count, self.horizon), dtype=np.int8)
        for step in range(self.horizon):
            index = self._score_index(scores[:, step])
            probability = self.probabilities[step, index]
            actions[:, step] = rng.random(count) < probability
            increment = np.where(
                actions[:, step] == 1,
                self.positive[step],
                self.negative[step],
            )
            scores[:, step + 1] = scores[:, step] + increment
        returns = self._utility(scores[:, -1]).astype(np.float64)
        return scores, actions, returns


def _trace(values, trace_lambda):
    powers = np.power(float(trace_lambda), np.arange(values.shape[1]))
    return values @ powers


def _summarize(name, advantages, actions, true_advantages, exact_gradient):
    root_probability = 0.43
    score = actions[:, 0].astype(np.float64) - root_probability
    gradients = score * advantages
    mean = float(gradients.mean(dtype=np.float64))
    variance = float(gradients.var(dtype=np.float64))
    standard_error = math.sqrt(variance / len(gradients))
    bias = mean - exact_gradient
    conditional_bias = []
    conditional_variance = []
    for action in (0, 1):
        selected = actions[:, 0] == action
        conditional_bias.append(
            float(advantages[selected].mean(dtype=np.float64) - true_advantages[action])
        )
        conditional_variance.append(
            float(advantages[selected].var(dtype=np.float64))
        )
    truth_snr = abs(exact_gradient) / math.sqrt(variance) if variance else math.inf
    return {
        "estimator": name,
        "samples": int(len(gradients)),
        "advantage_variance": float(advantages.var(dtype=np.float64)),
        "conditional_advantage_bias": conditional_bias,
        "conditional_advantage_variance": conditional_variance,
        "gradient_exact": float(exact_gradient),
        "gradient_mean": mean,
        "gradient_bias": float(bias),
        "gradient_bias_standard_errors": (
            float(bias / standard_error) if standard_error else 0.0
        ),
        "gradient_variance": variance,
        "gradient_standard_error": standard_error,
        "gradient_truth_snr_per_trajectory": float(truth_snr),
        "critical_batch_size_for_snr_one": (
            float(variance / exact_gradient**2) if exact_gradient else math.inf
        ),
    }


def run_experiment(samples: int, seed: int):
    game = FiniteRankGame()
    scores, actions, returns = game.sample(samples, seed)
    score_indices = game._score_index(scores)
    root_index = game.offset
    root_probability = float(game.probabilities[0, root_index])
    root_q = game.q[0, root_index]
    root_v = float(game.v[0, root_index])
    true_advantages = root_q - root_v
    exact_gradient = root_probability * (1 - root_probability) * (
        root_q[1] - root_q[0]
    )

    estimators = []
    current = returns - root_v
    estimators.append(_summarize(
        "current_kyoku_terminal_minus_boundary_v",
        current,
        actions,
        true_advantages,
        exact_gradient,
    ))

    # Exact-V GAE is included to isolate future sampled-action noise.
    exact_v_residuals = np.zeros_like(actions, dtype=np.float64)
    for step in range(game.horizon):
        current_v = game.v[step, score_indices[:, step]]
        if step == game.horizon - 1:
            reward = returns
            next_v = 0.0
        else:
            reward = 0.0
            next_v = game.v[step + 1, score_indices[:, step + 1]]
        exact_v_residuals[:, step] = reward + next_v - current_v
    for trace_lambda in (0.95, 1.0):
        estimate = _trace(exact_v_residuals, trace_lambda)
        estimators.append(_summarize(
            f"exact_v_gae_lambda_{trace_lambda:g}",
            estimate,
            actions,
            true_advantages,
            exact_gradient,
        ))

    for maximum_error in (0.0, 0.05, 0.15, 0.30):
        q_bar = game.approximate_q(maximum_error)
        v_bar = game.state_values_from_q(q_bar)
        selected_q = np.zeros_like(actions, dtype=np.float64)
        residuals = np.zeros_like(actions, dtype=np.float64)
        for step in range(game.horizon):
            current_index = score_indices[:, step]
            selected_q[:, step] = q_bar[
                step, current_index, actions[:, step]
            ]
            if step == game.horizon - 1:
                reward = returns
                next_v = 0.0
            else:
                reward = 0.0
                next_v = v_bar[step + 1, score_indices[:, step + 1]]
            residuals[:, step] = reward + next_v - selected_q[:, step]
        root_q_bar = selected_q[:, 0]
        root_v_bar = float(v_bar[0, root_index])
        for trace_lambda in (0.95, 1.0):
            estimate = (
                root_q_bar - root_v_bar
                + _trace(residuals, trace_lambda)
            )
            estimators.append(_summarize(
                f"q_boost_lambda_{trace_lambda:g}_q_sup_error_{maximum_error:g}",
                estimate,
                actions,
                true_advantages,
                exact_gradient,
            ))

    baseline = next(
        row for row in estimators
        if row["estimator"] == "current_kyoku_terminal_minus_boundary_v"
    )
    for row in estimators:
        row["gradient_variance_ratio_vs_current_kyoku"] = (
            row["gradient_variance"] / baseline["gradient_variance"]
        )
        row["critical_batch_ratio_vs_current_kyoku"] = (
            row["critical_batch_size_for_snr_one"]
            / baseline["critical_batch_size_for_snr_one"]
        )
        row["gradient_bias_detected_at_three_sigma"] = (
            abs(row["gradient_bias_standard_errors"]) > 3.0
        )

    return {
        "protocol": {
            "game": "four-seat finite stochastic rank-utility hand",
            "horizon": game.horizon,
            "samples": int(samples),
            "seed": int(seed),
            "gamma": 1.0,
            "root_action_probability": root_probability,
            "rank_utilities": RANK_UTILITIES.tolist(),
            "gradient_snr": "abs(exact gradient) / per-trajectory gradient std",
            "bias_test": "absolute Monte Carlo bias greater than 3 standard errors",
        },
        "exact": {
            "root_q": root_q.tolist(),
            "root_v": root_v,
            "root_advantages": true_advantages.tolist(),
            "root_policy_gradient": float(exact_gradient),
        },
        "estimators": estimators,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.samples < 10_000:
        parser.error("--samples must be at least 10000")
    result = run_experiment(args.samples, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
