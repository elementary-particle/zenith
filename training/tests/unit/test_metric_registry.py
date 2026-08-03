import pytest

from zenith_ppo.metric_registry import validate
from zenith_ppo.metrics import tensorboard_records


def test_registry_rejects_invalid_metric_contracts():
    validate("ppo/policy_loss", "match", "scalar", "batch", "mean")
    with pytest.raises(ValueError):
        validate("ppo/policy_loss", "step", "scalar", "batch", "mean")
    with pytest.raises(KeyError):
        validate("unknown/x", "match", "count", "instant", "last")


def test_tensorboard_projection_keeps_production_and_gameplay_signals():
    names = (
        "ppo/policy_loss",
        "critic/boundary_order_cross_entropy",
        "critic/boundary_rank_brier",
        "game/player_average_winning_points",
        "game/player_average_deal_in_points",
        "game/exhaustive_ryukyoku_rate",
        "game/player_bankrupt_rate",
        "game/player_average_turns_before_winning",
    )
    records = tuple(
        {"name": name, "value": float(index), "step": 1}
        for index, name in enumerate(names)
    ) + (
        {"name": "debug/unused", "value": 0.0, "step": 1},
    )

    projected = tensorboard_records(records)

    assert tuple(row["name"] for row in projected) == names
