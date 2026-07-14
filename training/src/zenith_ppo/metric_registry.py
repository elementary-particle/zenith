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
    "ppo/policy_loss": MetricDefinition("update", "scalar", "update", "mean"),
    "ppo/value_loss": MetricDefinition("update", "scalar", "update", "mean"),
    "ppo/entropy": MetricDefinition("update", "nats", "update", "mean"),
    "ppo/total_loss": MetricDefinition("update", "scalar", "update", "mean"),
    "ppo/approximate_kl": MetricDefinition("update", "ratio", "update", "mean"),
    "ppo/clip_fraction": MetricDefinition("update", "ratio", "update", "mean"),
    "ppo/gradient_norm": MetricDefinition("update", "scalar", "update", "mean"),
    "belief/total_loss": MetricDefinition("update", "scalar", "update", "mean"),
    "belief/count_loss": MetricDefinition("update", "nats", "update", "mean"),
    "belief/tenpai_loss": MetricDefinition("update", "nats", "update", "mean"),
    "belief/count_accuracy": MetricDefinition("update", "ratio", "update", "mean"),
    "belief/tenpai_accuracy": MetricDefinition("update", "ratio", "update", "mean"),
    "rollout/decisions": MetricDefinition("update", "count", "update", "sum"),
    "rollout/reward_mean": MetricDefinition("update", "scalar", "update", "mean"),
    "rollout/return_mean": MetricDefinition("update", "scalar", "update", "mean"),
    "rollout/advantage_mean": MetricDefinition("update", "scalar", "update", "mean"),
    "rollout/ppo_eligible": MetricDefinition("update", "count", "update", "sum"),
    "rollout/kyoku_completions": MetricDefinition("update", "count", "update", "sum"),
    "rollout/match_completions": MetricDefinition("update", "count", "update", "sum"),
    "rollout/kyoku_environment_coverage": MetricDefinition(
        "update", "ratio", "update", "last"
    ),
    "rollout/boundary_aligned": MetricDefinition("update", "ratio", "update", "last"),
    "curriculum/progress": MetricDefinition("update", "ratio", "instant", "last"),
    "curriculum/discard_weight": MetricDefinition("update", "ratio", "instant", "last"),
    "curriculum/kyoku_weight": MetricDefinition("update", "ratio", "instant", "last"),
    "curriculum/rank_weight": MetricDefinition("update", "ratio", "instant", "last"),
    "population/pool_size": MetricDefinition("update", "count", "instant", "last"),
    "population/historical_fraction": MetricDefinition("update", "ratio", "update", "mean"),
    "population/checkpoint_cohort_size": MetricDefinition(
        "update", "count", "instant", "last"
    ),
    "population/inference_model_count": MetricDefinition(
        "update", "count", "update", "last"
    ),
    "evaluation/games": MetricDefinition("evaluation_series", "count", "series", "sum"),
    "encoding/mean_token_length": MetricDefinition(
        "update", "tokens/decision", "update", "mean"
    ),
    "encoding/padding_fraction": MetricDefinition("update", "ratio", "update", "last"),
    "performance/decisions_per_second": MetricDefinition("update", "decisions/s", "update", "mean"),
    "system/learning_rate": MetricDefinition("update", "scalar", "instant", "last"),
    "system/writer_failures": MetricDefinition("update", "count", "run", "sum"),
}


def validate(name, axis, unit, window, reduction):
    definition = REGISTRY.get(name)
    if definition is None: raise KeyError(f"unregistered metric {name!r}")
    actual = (axis, unit, window, reduction)
    expected = (definition.axis, definition.unit, definition.window, definition.reduction)
    if actual != expected: raise ValueError(f"metric schema mismatch for {name}: {actual} != {expected}")


def checkpoint_metric_tag(checkpoint_id: str, metric: str) -> str:
    """Bound dynamic checkpoint cardinality to a stable digest prefix."""
    from hashlib import sha256

    if not checkpoint_id or "/" in metric:
        raise ValueError("invalid checkpoint metric identity")
    identity = sha256(checkpoint_id.encode()).hexdigest()[:12]
    return f"population/checkpoint_{identity}/{metric}"
