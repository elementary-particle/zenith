import numpy as np

from experiments.gae_falls_short.exact_bias_variance import (
    FiniteRankGame,
    run_experiment,
)


def _by_name(result):
    return {row["estimator"]: row for row in result["estimators"]}


def test_dynamic_programming_bellman_identity():
    game = FiniteRankGame()
    for step in range(game.horizon):
        probability = game.probabilities[step]
        expected = (
            (1 - probability) * game.q[step, :, 0]
            + probability * game.q[step, :, 1]
        )
        assert np.allclose(game.v[step], expected)


def test_exact_q_boost_is_pathwise_true_advantage():
    result = run_experiment(50_000, 19)
    rows = _by_name(result)
    for trace_lambda in ("0.95", "1"):
        row = rows[f"q_boost_lambda_{trace_lambda}_q_sup_error_0"]
        assert max(map(abs, row["conditional_advantage_bias"])) < 1e-12
        assert max(row["conditional_advantage_variance"]) < 1e-20


def test_lambda_one_has_no_detectable_gradient_bias_with_inaccurate_q():
    result = run_experiment(200_000, 23)
    rows = _by_name(result)
    for error in ("0.05", "0.15", "0.3"):
        row = rows[f"q_boost_lambda_1_q_sup_error_{error}"]
        assert not row["gradient_bias_detected_at_three_sigma"]


def test_approximate_q_boost_reduces_variance_in_controlled_game():
    result = run_experiment(50_000, 29)
    rows = _by_name(result)
    row = rows["q_boost_lambda_0.95_q_sup_error_0.15"]
    assert row["gradient_variance_ratio_vs_current_kyoku"] < 1.0
