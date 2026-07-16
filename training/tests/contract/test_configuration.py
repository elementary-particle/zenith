from copy import deepcopy
import pytest
from zenith_ppo.config import load, validate


def test_profiles_resolve_and_unknown_keys_fail():
    config = load("training/configs/smoke.toml")
    assert config.values["env"]["num_envs"] == 1
    invalid = deepcopy(config.values); invalid["model"]["mystery"] = True
    with pytest.raises(ValueError, match="unknown model"): validate(invalid)


def test_production_default_balances_rollout_depth_and_parallelism():
    config = load("training/configs/default.toml")
    assert config.values["env"]["num_envs"] == 128
    assert config.values["rollout"]["matches_per_update"] == 128
    assert config.values["ppo"]["minibatches"] == 4
    assert config.values["population"]["retained_checkpoints_max"] == 64
    assert config.values["evaluation"]["cadence_matches"] == 32000
    assert config.values["checkpoint"]["cadence_matches"] == 1024
    assert config.values["ppo"]["learning_rate"] == pytest.approx(0.00025)
    assert config.values["ppo"]["score_gae_lambda"] == pytest.approx(0.98)
    assert config.values["ppo"]["rank_gae_lambda"] == 1.0
    assert config.values["ppo"]["score_value_scale"] == 10.0
    assert config.values["model"]["critic_layers"] == 4
    assert "value_loss" not in config.values["ppo"]
    assert "critic_actor_gradient_scale" not in config.values["ppo"]
    assert config.values["ppo"]["value_clip"] == 0.0
    assert config.values["curriculum"]["total_matches"] == 46_000
    assert config.values["curriculum"]["taper_matches"] == 10_000
    assert config.values["teacher"]["discard_coefficient"] == pytest.approx(.50)
    assert 4 * len(config.values["evaluation"]["held_out_seeds"]) == 16
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
        ("curriculum", "total_matches"),
        ("checkpoint", "cadence_matches"),
        ("evaluation", "cadence_matches"),
        ("metrics", "progress_every_matches"),
    ):
        invalid = deepcopy(config.values)
        invalid[group][key] = 0
        with pytest.raises(ValueError, match="must be positive"):
            validate(invalid)


def test_checkpoint_pool_retains_four_evaluation_participants():
    config = load("training/configs/smoke.toml")
    invalid = deepcopy(config.values)
    invalid["population"]["retained_checkpoints_max"] = 3
    with pytest.raises(ValueError, match="retain four"):
        validate(invalid)


def test_padding_waste_limit_is_a_fraction():
    config = load("training/configs/smoke.toml")
    invalid = deepcopy(config.values)
    invalid["encoding"]["packing_max_waste"] = 1.0
    with pytest.raises(ValueError, match="packing_max_waste"):
        validate(invalid)


def test_ppo_safety_parameters_are_strictly_validated():
    config = load("training/configs/smoke.toml")
    for key, value, message in (
        ("score_gae_lambda", 1.01, "score_gae_lambda"),
        ("rank_gae_lambda", -0.01, "rank_gae_lambda"),
        ("target_kl", 0, "target_kl"),
        ("value_clip", -0.01, "value clip"),
        ("max_grad_norm", 0, "max_grad_norm"),
        ("epochs", 0, "epochs"),
        ("minibatches", 0, "minibatches"),
    ):
        invalid = deepcopy(config.values)
        invalid["ppo"][key] = value
        with pytest.raises(ValueError, match=message):
            validate(invalid)

    invalid = deepcopy(config.values)
    invalid["model"]["dropout"] = 0.1
    with pytest.raises(ValueError, match="exact PPO behavior accounting"):
        validate(invalid)

    invalid = deepcopy(config.values)
    invalid["ppo"]["score_value_scale"] = 0
    with pytest.raises(ValueError, match="score_value_scale"):
        validate(invalid)
