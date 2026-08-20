from copy import deepcopy

import pytest

from zenith_ppo.config import evaluation_seeds, load, validate
from zenith_ppo.orchestrator import _learning_rate


def _values():
    return deepcopy(load("training/configs/default.toml").values)


def test_production_and_smoke_profiles_resolve():
    production = load("training/configs/default.toml")
    smoke = load("training/configs/smoke.toml")

    assert production.values["env"]["num_envs"] == 2048
    assert production.values["rollout"]["matches_per_update"] == 2048
    assert production.values["ppo"]["epochs"] == 2
    assert production.values["ppo"]["actor_learning_rate"] == pytest.approx(
        2e-4
    )
    assert production.values["ppo"]["critic_learning_rate"] == pytest.approx(
        5e-4
    )
    assert production.values["ppo"]["critic_epochs"] == 4
    assert production.values["ppo"]["actor_learning_starts_matches"] == 2048
    assert production.values["ppo"]["actor_warmup_matches"] == 4096
    assert production.values["ppo"]["critic_warmup_matches"] == 0
    assert production.values["ppo"]["target_kl"] == pytest.approx(2e-4)
    assert production.values["ppo"]["magnet_kl_coefficient"] == pytest.approx(
        0.10
    )
    assert production.values["ppo"]["magnet_half_life_matches"] == 32768
    assert production.values["model"]["architecture"] == \
        "verified-public-state-value-ppo-v1"
    assert production.values["model"]["d_model"] == 256
    assert production.values["ppo"]["value_coefficient"] == pytest.approx(0.5)
    assert production.values["ppo"]["gae_lambda"] == pytest.approx(0.9)
    assert production.values["behavior_cloning"]["outcome_coefficient"] \
        == pytest.approx(0.05)
    assert production.values["curriculum"]["total_matches"] == 262144
    assert len(evaluation_seeds(production.values["evaluation"])) == 256
    assert production.values["evaluation"]["batch_size"] == 1024
    assert production.values["evaluation"]["checkpoint_matches"] == [
        16384, 32768, 65536, 131072, 262144,
    ]
    assert len(evaluation_seeds(smoke.values["evaluation"])) == 4
    assert production.values["checkpoint"] == {
        "cadence_matches": 4096,
        "keep": 64,
    }
    assert smoke.values["model"]["d_model"] == 16


def test_unknown_keys_fail():
    values = _values()
    values["ppo"]["unknown_setting"] = 1.0
    with pytest.raises(ValueError, match="unknown ppo keys"):
        validate(values)


def test_state_value_settings_are_validated():
    values = _values()
    values["ppo"]["value_coefficient"] = 0.0
    with pytest.raises(ValueError, match="value_coefficient"):
        validate(values)
    values = _values()
    values["ppo"]["gae_lambda"] = 1.1
    with pytest.raises(ValueError, match="gae_lambda"):
        validate(values)


def test_learning_start_and_warmup_counts_are_validated():
    values = _values()
    values["ppo"]["actor_learning_starts_matches"] = -1
    with pytest.raises(ValueError, match="learning-start"):
        validate(values)


def test_actor_schedule_warms_up_then_keeps_the_calibrated_rate():
    base = 2e-4
    assert _learning_rate(base, 2048, 4096, starts=2048) == 0.0
    assert _learning_rate(base, 4096, 4096, starts=2048) == pytest.approx(1e-4)
    assert _learning_rate(base, 6144, 4096, starts=2048) == pytest.approx(base)
    assert _learning_rate(base, 262144, 4096, starts=2048) == pytest.approx(base)


def test_model_dimensions_are_validated():
    values = _values()
    values["model"]["d_model"] = 191
    with pytest.raises(ValueError, match="d_model"):
        validate(values)


def test_ppo_low_bias_scale_and_safety_are_validated():
    values = _values()
    values["ppo"]["target_kl"] = 0.0
    with pytest.raises(ValueError, match="target_kl"):
        validate(values)
    values = _values()
    values["ppo"]["magnet_half_life_matches"] = 0
    with pytest.raises(ValueError, match="magnet_half_life_matches"):
        validate(values)
    values = _values()
    values["ppo"]["magnet_kl_coefficient"] = 0
    with pytest.raises(ValueError, match="magnet_kl_coefficient"):
        validate(values)
    values = _values()
    values["env"]["num_envs"] = 1
    values["ppo"]["critic_epochs"] = 2
    with pytest.raises(ValueError, match="retained logical batch"):
        validate(values)


def test_multiple_actor_epochs_require_a_retained_logical_batch():
    values = _values()
    values["env"]["num_envs"] = 1024
    with pytest.raises(ValueError, match="retained logical batch"):
        validate(values)
    values["ppo"]["epochs"] = 1
    values["ppo"]["critic_epochs"] = 1
    validate(values)


def test_schedule_horizon_can_exceed_the_run_budget_but_not_undershoot_it():
    values = _values()
    values["curriculum"]["total_matches"] = 16384
    values["curriculum"]["schedule_matches"] = 262144
    validate(values)

    values["curriculum"]["schedule_matches"] = 8192
    with pytest.raises(ValueError, match="schedule_matches"):
        validate(values)


def test_ema_self_play_requires_a_positive_opponent_half_life():
    values = _values()
    values["rollout"]["training_mode"] = "ema_self_play"
    with pytest.raises(ValueError, match="ema_opponent_half_life_matches"):
        validate(values)
    values["rollout"]["ema_opponent_half_life_matches"] = 8192
    validate(values)
    values["rollout"]["ema_opponent_half_life_matches"] = 0
    with pytest.raises(ValueError, match="positive"):
        validate(values)


def test_unilateral_learner_seats_require_checkpoint_league():
    values = _values()
    values["rollout"]["league_learner_seats"] = 1
    with pytest.raises(ValueError, match="requires checkpoint_league"):
        validate(values)
    values["rollout"]["training_mode"] = "checkpoint_league"
    values["rollout"]["league_checkpoints"] = ["target"]
    validate(values)
    values["rollout"]["league_learner_seats"] = 3
    with pytest.raises(ValueError, match="one or two"):
        validate(values)


def test_boundary_rank_bc_settings_are_positive():
    values = _values()
    values["behavior_cloning"]["boundary_rank_learning_rate"] = 0.0
    with pytest.raises(ValueError, match="learning rate"):
        validate(values)
    values = _values()
    values["behavior_cloning"]["boundary_rank_coefficient"] = 0.0
    with pytest.raises(ValueError, match="boundary_rank_coefficient"):
        validate(values)


def test_bc_zero_train_decisions_means_entire_archive():
    values = _values()
    values["behavior_cloning"]["train_decisions"] = 0
    validate(values)


def test_redaction_preserves_normal_values():
    config = load("training/configs/default.toml")
    redacted = config.redacted()
    assert redacted["run"]["seed"] == config.values["run"]["seed"]
