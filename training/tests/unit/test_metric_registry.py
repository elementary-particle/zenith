import pytest
from types import SimpleNamespace

from zenith_ppo.metric_registry import REGISTRY, validate
from zenith_ppo.metrics import completed_match_metric_values


def test_registry_rejects_wrong_axis_and_dynamic_tags():
    validate("ppo/policy_loss", "match", "scalar", "batch", "mean")
    with pytest.raises(ValueError): validate("ppo/policy_loss", "step", "scalar", "batch", "mean")
    with pytest.raises(KeyError): validate("unknown/x", "match", "count", "instant", "last")


def test_registry_contains_only_current_public_metrics():
    removed = {
        "teacher/discard_loss_rows",
        "teacher/reaction_loss_rows",
        "teacher/riichi_loss_rows",
        "rollout/decisions",
        "rollout/ppo_eligible",
        "population/checkpoint_cohort_size",
        "population/inference_model_count",
        "evaluation/games",
        "encoding/mean_token_length",
        "encoding/padding_fraction",
        "system/writer_failures",
        "ppo/value_loss",
        "ppo/explained_variance",
        "ppo/value_clip_fraction",
        "rollout/return_mean",
        "rollout/advantage_mean",
        "game/open_wins",
        "game/closed_wins",
        "game/deal_ins_after_opponent_riichi",
        "game/exhaustive_draws",
        "game/exhaustive_draw_tenpai_score",
        "teacher/reaction_calls_per_hand",
        "teacher/reaction_improving_calls",
        "teacher/riichi_legal_opportunities",
        "teacher/riichi_declarations",
    }
    assert removed.isdisjoint(REGISTRY)
    assert "ppo/score_explained_variance" in REGISTRY
    assert "ppo/rank_explained_variance" in REGISTRY
    assert "game/open_wins_per_kyoku" in REGISTRY
    assert "game/exhaustive_ryukyoku_rate" in REGISTRY

def test_completed_match_metrics_use_ranked_scores_and_kyoku_counts():
    values = completed_match_metric_values((
        SimpleNamespace(
            ranks=(2, 0, 3, 1), scores=(24_000, 41_000, 9_000, 26_000),
            completed_kyoku=8,
        ),
        SimpleNamespace(
            ranks=(0, 2, 1, 3), scores=(35_000, 22_000, 28_000, 15_000),
            completed_kyoku=12,
        ),
    ))

    assert values == {
        "game/first_place_score_mean": 38.0,
        "game/fourth_place_score_mean": 12.0,
        "game/kyoku_per_match_mean": 10.0,
    }
    assert completed_match_metric_values(()) == {}
