from copy import deepcopy
import pytest
from zenith_ppo.config import load, validate


def test_profiles_resolve_and_unknown_keys_fail():
    config = load("training/configs/smoke.toml")
    assert config.values["env"]["state_schema"] == 2
    invalid = deepcopy(config.values); invalid["model"]["mystery"] = True
    with pytest.raises(ValueError, match="unknown model"): validate(invalid)


def test_production_default_balances_rollout_depth_and_parallelism():
    config = load("training/configs/default.toml")
    assert config.values["env"]["num_envs"] == 128
    assert config.values["rollout"]["learner_decisions_per_update"] == 8192
    assert config.values["rollout"]["complete_kyoku_per_env"] is True
    assert config.values["population"]["checkpoint_cohort_size"] == 2
    assert config.values["evaluation"]["cadence_updates"] == 250
    assert config.compatibility.token_schema == 5
    assert config.compatibility.action_schema == 2
    assert config.compatibility.model_schema == 5
    assert config.compatibility.curriculum_schema == 2
    assert config.compatibility.metric_schema == 3


def test_oracle_configuration_is_obsolete_and_privilege_is_required():
    config = load("training/configs/smoke.toml")
    obsolete = deepcopy(config.values)
    obsolete["observation"]["actor_oracle_groups"] = ["opponent_concealed_hands"]
    with pytest.raises(ValueError, match="unknown observation"):
        validate(obsolete)
    ordinary_native = deepcopy(config.values)
    ordinary_native["env"]["privileged"] = False
    with pytest.raises(ValueError, match="privileged native state"):
        validate(ordinary_native)


def test_redaction_and_no_cuda_fallback(monkeypatch):
    config = load("training/configs/smoke.toml")
    values = deepcopy(config.values); values["run"]["api_token"] = "secret"
    assert config.redacted()["run"]["profile"] == "cpu-smoke"


def test_training_loop_counts_and_cadences_must_be_positive():
    config = load("training/configs/smoke.toml")
    for group, key in (
        ("curriculum", "total_updates"),
        ("population", "admit_every_updates"),
        ("evaluation", "cadence_updates"),
        ("checkpoint", "cadence_updates"),
    ):
        invalid = deepcopy(config.values)
        invalid[group][key] = 0
        with pytest.raises(ValueError, match="must be positive"):
            validate(invalid)


def test_checkpoint_cohort_covers_distinct_historical_seats():
    config = load("training/configs/smoke.toml")
    invalid = deepcopy(config.values)
    invalid["population"]["checkpoint_cohort_size"] = 1
    with pytest.raises(ValueError, match="cohort_size"):
        validate(invalid)


def test_padding_waste_limit_is_a_fraction():
    config = load("training/configs/smoke.toml")
    invalid = deepcopy(config.values)
    invalid["encoding"]["packing_max_waste"] = 1.0
    with pytest.raises(ValueError, match="packing_max_waste"):
        validate(invalid)
