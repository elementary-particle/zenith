"""Stable metric tag registry."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MetricDefinition:
    axis: str
    unit: str
    window: str
    reduction: str


REGISTRY = {
    "ppo/policy_loss": MetricDefinition("match", "scalar", "batch", "mean"),
    "ppo/critic_loss": MetricDefinition("match", "scalar", "batch", "mean"),
    "ppo/score_value_loss": MetricDefinition("match", "scalar", "batch", "mean"),
    "ppo/rank_value_loss": MetricDefinition("match", "scalar", "batch", "mean"),
    "ppo/score_normalized_mse": MetricDefinition("match", "scalar", "batch", "mean"),
    "ppo/score_raw_mae": MetricDefinition("match", "thousand_points", "batch", "mean"),
    "ppo/score_raw_rmse": MetricDefinition("match", "thousand_points", "batch", "mean"),
    "ppo/score_critic_explained_variance": MetricDefinition("match", "ratio", "batch", "mean"),
    "ppo/rank_cross_entropy": MetricDefinition("match", "nats", "batch", "mean"),
    "ppo/rank_accuracy": MetricDefinition("match", "ratio", "batch", "mean"),
    "ppo/rank_brier": MetricDefinition("match", "scalar", "batch", "mean"),
    "ppo/rank_utility_explained_variance": MetricDefinition("match", "ratio", "batch", "mean"),
    "ppo/score_explained_variance": MetricDefinition("match", "ratio", "batch", "mean"),
    "ppo/rank_explained_variance": MetricDefinition("match", "ratio", "batch", "mean"),
    "ppo/entropy": MetricDefinition("match", "nats", "batch", "mean"),
    "ppo/total_loss": MetricDefinition("match", "scalar", "batch", "mean"),
    "ppo/approximate_kl": MetricDefinition("match", "ratio", "batch", "mean"),
    "ppo/kl_stop_value": MetricDefinition("match", "ratio", "batch", "last"),
    "ppo/kl_early_stop": MetricDefinition("match", "ratio", "batch", "last"),
    "ppo/optimization_fraction": MetricDefinition("match", "ratio", "batch", "last"),
    "ppo/actor_optimization_fraction": MetricDefinition("match", "ratio", "batch", "last"),
    "ppo/critic_optimization_fraction": MetricDefinition("match", "ratio", "batch", "last"),
    "ppo/clip_fraction": MetricDefinition("match", "ratio", "batch", "mean"),
    "ppo/score_value_clip_fraction": MetricDefinition("match", "ratio", "batch", "mean"),
    "ppo/gradient_norm": MetricDefinition("match", "scalar", "batch", "mean"),
    "ppo/actor_gradient_norm": MetricDefinition("match", "scalar", "batch", "mean"),
    "ppo/critic_gradient_norm": MetricDefinition("match", "scalar", "batch", "mean"),
    "ppo/gradient_clip_fraction": MetricDefinition("match", "ratio", "batch", "mean"),
    "belief/total_loss": MetricDefinition("match", "scalar", "batch", "mean"),
    "belief/count_loss": MetricDefinition("match", "nats", "batch", "mean"),
    "belief/tenpai_loss": MetricDefinition("match", "nats", "batch", "mean"),
    "belief/count_accuracy": MetricDefinition("match", "ratio", "batch", "mean"),
    "belief/tenpai_accuracy": MetricDefinition("match", "ratio", "batch", "mean"),
    "teacher/auxiliary_loss": MetricDefinition("match", "scalar", "batch", "mean"),
    "teacher/discard_loss": MetricDefinition("match", "nats", "batch", "mean"),
    "teacher/discard_agreement": MetricDefinition("match", "ratio", "batch", "mean"),
    "teacher/discard_applicable_rows": MetricDefinition("match", "rows", "batch", "sum"),
    "teacher/discard_mean_shanten_regret": MetricDefinition("match", "shanten", "batch", "mean"),
    "teacher/discard_mean_ukeire_regret": MetricDefinition("match", "tiles", "batch", "mean"),
    "teacher/discard_mean_cost": MetricDefinition("match", "scalar", "batch", "mean"),
    "teacher/discard_worse_shanten_rate": MetricDefinition("match", "ratio", "batch", "mean"),
    "teacher/reaction_loss": MetricDefinition("match", "nats", "batch", "mean"),
    "teacher/reaction_applicable_rows": MetricDefinition("match", "rows", "batch", "sum"),
    "teacher/reaction_entropy": MetricDefinition("match", "nats", "batch", "mean"),
    "teacher/reaction_pass_probability": MetricDefinition("match", "ratio", "batch", "mean"),
    "teacher/reaction_call_probability": MetricDefinition("match", "ratio", "batch", "mean"),
    "teacher/reaction_calls_per_kyoku": MetricDefinition(
        "match", "calls/kyoku", "batch", "mean"
    ),
    "teacher/reaction_improving_calls_per_kyoku": MetricDefinition(
        "match", "calls/kyoku", "batch", "mean"
    ),
    "teacher/reaction_improving_call_rate": MetricDefinition("match", "ratio", "batch", "mean"),
    "teacher/riichi_loss": MetricDefinition("match", "nats", "batch", "mean"),
    "teacher/riichi_applicable_rows": MetricDefinition("match", "rows", "batch", "sum"),
    "teacher/riichi_legal_opportunities_per_kyoku": MetricDefinition(
        "match", "opportunities/kyoku", "batch", "mean"
    ),
    "teacher/riichi_declarations_per_kyoku": MetricDefinition(
        "match", "declarations/kyoku", "batch", "mean"
    ),
    "teacher/riichi_conversion_rate": MetricDefinition("match", "ratio", "batch", "mean"),
    "teacher/discard_coefficient": MetricDefinition("match", "scalar", "instant", "last"),
    "teacher/reaction_coefficient": MetricDefinition("match", "scalar", "instant", "last"),
    "teacher/riichi_coefficient": MetricDefinition("match", "scalar", "instant", "last"),
    "teacher/reaction_entropy_coefficient": MetricDefinition("match", "scalar", "instant", "last"),
    "rollout/reward_mean": MetricDefinition("match", "scalar", "batch", "mean"),
    "rollout/kyoku_reward_mean": MetricDefinition("match", "thousand_points", "batch", "mean"),
    "rollout/rank_reward_mean": MetricDefinition("match", "scalar", "batch", "mean"),
    "rollout/score_return_mean": MetricDefinition("match", "scalar", "batch", "mean"),
    "rollout/rank_return_mean": MetricDefinition("match", "scalar", "batch", "mean"),
    "rollout/score_advantage_mean": MetricDefinition("match", "scalar", "batch", "mean"),
    "rollout/rank_advantage_mean": MetricDefinition("match", "scalar", "batch", "mean"),
    "rollout/policy_advantage_mean": MetricDefinition("match", "scalar", "batch", "mean"),
    "rollout/kyoku_completions": MetricDefinition("match", "count", "batch", "sum"),
    "rollout/match_completions": MetricDefinition("match", "count", "batch", "sum"),
    "game/first_place_score_mean": MetricDefinition(
        "match", "thousand_points", "batch", "mean"
    ),
    "game/fourth_place_score_mean": MetricDefinition(
        "match", "thousand_points", "batch", "mean"
    ),
    "game/kyoku_per_match_mean": MetricDefinition(
        "match", "kyoku/match", "batch", "mean"
    ),
    "rollout_rating/win_rate_vs_conservative_bot": MetricDefinition(
        "match", "ratio", "cumulative", "last"
    ),
    "rollout_rating/lower_win_rate_vs_conservative_bot": MetricDefinition(
        "match", "ratio", "cumulative", "last"
    ),
    "rollout_rating/upper_win_rate_vs_conservative_bot": MetricDefinition(
        "match", "ratio", "cumulative", "last"
    ),
    "rollout_rating/bot_matches": MetricDefinition("match", "count", "cumulative", "last"),
    "rollout_rating/head_to_head_comparisons": MetricDefinition(
        "match", "count", "cumulative", "last"
    ),
    "curriculum/progress": MetricDefinition("match", "ratio", "instant", "last"),
    "curriculum/kyoku_weight": MetricDefinition("match", "ratio", "instant", "last"),
    "curriculum/rank_weight": MetricDefinition("match", "ratio", "instant", "last"),
    "curriculum/guidance_scale": MetricDefinition("match", "ratio", "instant", "last"),
    "curriculum/competence_streak": MetricDefinition("match", "count", "instant", "last"),
    "curriculum/regression_streak": MetricDefinition("match", "count", "instant", "last"),
    "curriculum/taper_progress": MetricDefinition("match", "ratio", "instant", "last"),
    "curriculum/last_valid_worse_shanten_rate": MetricDefinition("match", "ratio", "instant", "last"),
    "population/target_bot_fraction": MetricDefinition("match", "ratio", "instant", "last"),
    "performance/model_queries_per_second": MetricDefinition("match", "queries/s", "batch", "mean"),
    "performance/rust_resolved_decisions": MetricDefinition("match", "count", "batch", "sum"),
    "performance/automatic_resolution_fraction": MetricDefinition("match", "ratio", "batch", "mean"),
    "population/pool_size": MetricDefinition("match", "count", "instant", "last"),
    "population/conservative_bot_match_fraction": MetricDefinition(
        "match", "ratio", "batch", "mean"
    ),
    "game/open_wins_per_kyoku": MetricDefinition(
        "match", "wins/kyoku", "batch", "mean"
    ),
    "game/closed_wins_per_kyoku": MetricDefinition(
        "match", "wins/kyoku", "batch", "mean"
    ),
    "game/deal_ins_after_opponent_riichi_per_kyoku": MetricDefinition(
        "match", "deal-ins/kyoku", "batch", "mean"
    ),
    "game/exhaustive_ryukyoku_rate": MetricDefinition(
        "match", "ratio", "batch", "mean"
    ),
    "game/exhaustive_ryukyoku_tenpai_score_per_kyoku": MetricDefinition(
        "match", "thousand_points/kyoku", "batch", "mean"
    ),
    "system/learning_rate": MetricDefinition("match", "scalar", "instant", "last"),
}


def validate(name, axis, unit, window, reduction):
    definition = REGISTRY.get(name)
    if definition is None: raise KeyError(f"unregistered metric {name!r}")
    actual = (axis, unit, window, reduction)
    expected = (definition.axis, definition.unit, definition.window, definition.reduction)
    if actual != expected: raise ValueError(f"metric definition mismatch for {name}: {actual} != {expected}")
