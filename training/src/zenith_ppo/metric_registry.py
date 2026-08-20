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
    "critic/boundary_order_cross_entropy": MetricDefinition(
        "match", "nats", "batch", "mean"
    ),
    "critic/boundary_order_accuracy": MetricDefinition(
        "match", "ratio", "batch", "mean"
    ),
    "critic/boundary_rank_brier": MetricDefinition(
        "match", "scalar", "batch", "mean"
    ),
    "critic/match_rank_explained_variance": MetricDefinition(
        "match", "ratio", "batch", "mean"
    ),
    "ppo/entropy": MetricDefinition("match", "nats", "batch", "mean"),
    "ppo/entropy_efficiency": MetricDefinition("match", "ratio", "batch", "mean"),
    "ppo/entropy_applicable_rows": MetricDefinition("match", "rows", "batch", "sum"),
    "ppo/total_loss": MetricDefinition("match", "scalar", "batch", "mean"),
    "ppo/approximate_kl": MetricDefinition("match", "ratio", "batch", "mean"),
    "ppo/pre_update_approximate_kl": MetricDefinition(
        "match", "ratio", "batch", "last"
    ),
    "ppo/post_update_approximate_kl": MetricDefinition(
        "match", "ratio", "batch", "mean"
    ),
    "ppo/kl_loss": MetricDefinition("match", "scalar", "batch", "mean"),
    "ppo/kl_coefficient": MetricDefinition("match", "scalar", "instant", "last"),
    "ppo/magnet_kl": MetricDefinition("match", "nats", "batch", "mean"),
    "ppo/magnet_loss": MetricDefinition("match", "scalar", "batch", "mean"),
    "ppo/magnet_kl_coefficient": MetricDefinition(
        "match", "scalar", "instant", "last"
    ),
    "ppo/magnet_ema_tau": MetricDefinition(
        "match", "ratio", "instant", "last"
    ),
    "ppo/magnet_parameter_rms_distance": MetricDefinition(
        "match", "parameter", "instant", "last"
    ),
    "ppo/magnet_relative_parameter_rms_distance": MetricDefinition(
        "match", "ratio", "instant", "last"
    ),
    "ppo/entropy_coefficient": MetricDefinition(
        "match", "scalar", "instant", "last"
    ),
    "ppo/actor_optimization_fraction": MetricDefinition("match", "ratio", "batch", "last"),
    "ppo/critic_optimization_fraction": MetricDefinition("match", "ratio", "batch", "last"),
    "ppo/clip_fraction": MetricDefinition("match", "ratio", "batch", "mean"),
    "ppo/actor_gradient_norm": MetricDefinition("match", "scalar", "batch", "mean"),
    "ppo/critic_gradient_norm": MetricDefinition("match", "scalar", "batch", "mean"),
    "ppo/actor_gradient_clip_fraction": MetricDefinition(
        "match", "ratio", "batch", "mean"
    ),
    "ppo/critic_gradient_clip_fraction": MetricDefinition(
        "match", "ratio", "batch", "mean"
    ),
    "critic/state_value_loss": MetricDefinition(
        "match", "rank_utility", "batch", "mean"
    ),
    "critic/state_value_prediction_mean": MetricDefinition(
        "match", "rank_utility", "batch", "mean"
    ),
    "critic/state_value_target_mean": MetricDefinition(
        "match", "rank_utility", "batch", "mean"
    ),
    "critic/state_value_rmse": MetricDefinition(
        "match", "rank_utility", "batch", "mean"
    ),
    "critic/state_value_explained_variance": MetricDefinition(
        "match", "ratio", "batch", "mean"
    ),
    "critic/state_value_rollout_explained_variance": MetricDefinition(
        "match", "ratio", "batch", "mean"
    ),
    "rollout/state_value_advantage_mean": MetricDefinition(
        "match", "rank_utility", "batch", "mean"
    ),
    "rollout/state_value_advantage_std": MetricDefinition(
        "match", "rank_utility", "batch", "mean"
    ),
    "rollout/policy_advantage_mean": MetricDefinition("match", "scalar", "batch", "mean"),
    "rollout/policy_advantage_std": MetricDefinition("match", "scalar", "batch", "mean"),
    "rollout/call_opportunity_selected_call_rate": MetricDefinition(
        "match", "ratio", "batch", "mean"
    ),
    "rollout/riichi_opportunity_selected_riichi_rate": MetricDefinition(
        "match", "ratio", "batch", "mean"
    ),
    "rollout/kyoku_completions": MetricDefinition("match", "count", "batch", "sum"),
    "rollout/match_completions": MetricDefinition("match", "count", "batch", "sum"),
    "curriculum/progress": MetricDefinition("match", "ratio", "instant", "last"),
    "curriculum/actor_learning_rate": MetricDefinition(
        "match", "scalar", "instant", "last"
    ),
    "curriculum/critic_learning_rate": MetricDefinition(
        "match", "scalar", "instant", "last"
    ),
    "league/arena_size": MetricDefinition("match", "count", "instant", "last"),
    "league/observed_pairs": MetricDefinition("match", "count", "instant", "last"),
    "league/pairwise_games": MetricDefinition("match", "count", "instant", "last"),
    "league/maximum_payoff_gap": MetricDefinition("match", "scalar", "instant", "last"),
    "performance/model_queries_per_second": MetricDefinition("match", "queries/s", "batch", "mean"),
    "performance/rust_resolved_decisions": MetricDefinition("match", "count", "batch", "sum"),
    "performance/automatic_resolution_fraction": MetricDefinition("match", "ratio", "batch", "mean"),
    "population/pool_size": MetricDefinition("match", "count", "instant", "last"),
    "game/player_win_rate": MetricDefinition("match", "ratio", "batch", "mean"),
    "game/player_deal_in_rate": MetricDefinition("match", "ratio", "batch", "mean"),
    "game/player_riichi_rate": MetricDefinition("match", "ratio", "batch", "mean"),
    "game/player_calling_rate": MetricDefinition("match", "ratio", "batch", "mean"),
    "game/player_average_winning_points": MetricDefinition(
        "match", "points", "batch", "mean"
    ),
    "game/player_average_deal_in_points": MetricDefinition(
        "match", "points", "batch", "mean"
    ),
    "game/exhaustive_ryukyoku_rate": MetricDefinition(
        "match", "ratio", "batch", "mean"
    ),
    "game/player_bankrupt_rate": MetricDefinition("match", "ratio", "batch", "mean"),
    "game/player_tsumo_rate": MetricDefinition("match", "ratio", "batch", "mean"),
    "game/player_dama_rate": MetricDefinition("match", "ratio", "batch", "mean"),
    "game/player_average_turns_before_winning": MetricDefinition(
        "match", "turns", "batch", "mean"
    ),
}


def validate(name, axis, unit, window, reduction):
    definition = REGISTRY.get(name)
    if definition is None:
        raise KeyError(f"unregistered metric {name!r}")
    actual = (axis, unit, window, reduction)
    expected = (definition.axis, definition.unit, definition.window, definition.reduction)
    if actual != expected:
        raise ValueError(
            f"metric definition mismatch for {name}: {actual} != {expected}"
        )
