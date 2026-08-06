from copy import deepcopy

import pytest
import torch

from zenith_ppo.model.actor_critic import ActorCritic
from zenith_ppo.ppo.trainer import PPOTrainer, _finalize_statistic_metrics


def _model():
    return ActorCritic({
        "layers": 1,
        "d_model": 16,
        "query_heads": 2,
        "kv_heads": 1,
        "head_dim": 8,
        "ffn_dim": 32,
        "context_tokens": 64,
        "action_memory_layers": 1,
        "action_memory_ffn_dim": 32,
        "share_all_action_tiles": True,
        "concealed_shape_channels": 4,
        "concealed_shape_blocks": 1,
        "rank_critic_width": 16,
    })


def _config(**changes):
    return {
        "actor_learning_rate": 1e-3,
        "critic_learning_rate": 2e-3,
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_epsilon": 1e-8,
        "weight_decay": 0.0,
        "epochs": 1,
        "critic_epochs": 2,
        "minibatches": 1,
        "ratio_clip": 0.2,
        "boundary_rank_coefficient": 1.0,
        "max_grad_norm": 100.0,
        "target_kl": 100.0,
        "kl_coefficient_initial": 0.2,
        "kl_coefficient_minimum": 0.0001,
        "kl_coefficient_maximum": 10.0,
        "kl_adaptation_factor": 1.5,
        "magnet_kl_coefficient": 0.1,
        "magnet_half_life_matches": 100.0,
        "entropy_floor": 0.0,
    } | changes


def _inputs():
    actions = torch.zeros(2, 3, 15, dtype=torch.long)
    actions[..., 1] = torch.tensor([[1, 2, 3], [4, 5, 0]])
    tokens = torch.zeros(2, 7, 10, dtype=torch.long)
    tokens[:, 0, (0, 1, 2, 4, 5, 7)] = torch.tensor(
        [3, 4, 1, 1, 1, 2]
    )
    return {
        "token_factors": tokens,
        "token_numeric": torch.ones(2, 7, 8) * 0.1,
        "lengths": torch.tensor([7, 6]),
        "actor_query_indices": torch.tensor([6, 5]),
        "action_factors": actions,
        "action_lengths": torch.tensor([3, 2]),
        "action_offsets": torch.tensor([0, 3, 5]),
        "decision_seats": torch.tensor([0, 1]),
        "rank_boundary_features": torch.zeros(2, 28),
        "backend": "eager",
    }


def _batches(model):
    inputs = _inputs()
    with torch.no_grad():
        output = model.forward_actor(**inputs)
    selected = torch.tensor([0, 3])
    actor = {
        "model_inputs": inputs,
        "selected": selected,
        "old_logp": output.log_probabilities[selected].clone(),
        "advantages": torch.tensor([1.0, -1.0]),
        "raw_advantages": torch.tensor([1.0, -1.0]),
        "action_counts": torch.tensor([3, 2]),
    }
    critic = {
        "model_inputs": {
            "decision_seats": inputs["decision_seats"],
            "rank_boundary_features": inputs["rank_boundary_features"],
        },
        "rank_boundary_supervision": torch.ones(2, dtype=torch.bool),
        "rank_order_targets": torch.tensor([0, 23]),
    }
    return actor, critic


def _state(parameters):
    return [parameter.detach().clone() for parameter in parameters]


def test_actor_and_critic_optimizers_use_independent_learning_rates():
    trainer = PPOTrainer(
        _model(),
        _config(actor_learning_rate=5e-5, critic_learning_rate=2e-4),
    )
    assert trainer.actor_optimizer.param_groups[0]["lr"] == pytest.approx(5e-5)
    assert trainer.critic_optimizer.param_groups[0]["lr"] == pytest.approx(2e-4)


def test_full_policy_and_boundary_critic_update_commit_together():
    model = _model()
    trainer = PPOTrainer(model, _config())
    actor_batch, critic_batch = _batches(model)
    actor_before = {
        name: parameter.detach().clone()
        for name, parameter in model.actor_named_parameters()
    }
    critic_before = _state(trainer.critic_parameter_list)

    result = trainer.update([actor_batch], critic_minibatches=[critic_batch])

    assert result.committed, result.reason
    assert result.metrics["actor_optimization_fraction"] == 1.0
    assert result.metrics["critic_optimization_fraction"] == 1.0
    for prefix in (
        "token_embedding.",
        "canonical_tile_embedding.",
        "backbone.",
        "action_memory.concealed_shape.",
        "action_memory.blocks.",
        "policy_head.",
    ):
        assert any(
            name.startswith(prefix)
            and not torch.equal(actor_before[name], parameter)
            for name, parameter in model.actor_named_parameters()
        ), prefix
    assert any(
        not torch.equal(before, after)
        for before, after in zip(critic_before, trainer.critic_parameter_list)
    )
    assert result.metrics["magnet_kl"] == pytest.approx(0.0, abs=1e-7)
    assert result.metrics["magnet_ema_tau"] == pytest.approx(
        1 - 2 ** (-1 / 100)
    )
    assert result.metrics["magnet_parameter_rms_distance"] > 0


def test_failed_update_rolls_back_both_models_and_optimizers():
    model = _model()
    trainer = PPOTrainer(model, _config(target_kl=1e-12))
    actor_batch, critic_batch = _batches(model)
    model_before = deepcopy(model.state_dict())
    actor_optimizer_before = deepcopy(trainer.actor_optimizer.state_dict())
    magnet_before = trainer.ema_magnet.state_dict()

    result = trainer.update([actor_batch], critic_minibatches=[critic_batch])

    assert not result.committed
    assert all(
        torch.equal(value, model.state_dict()[name])
        for name, value in model_before.items()
    )
    assert trainer.actor_optimizer.state_dict() == actor_optimizer_before
    assert trainer.ema_magnet.updates == 0
    assert all(
        torch.equal(value, trainer.ema_magnet.state_dict()["actor_parameters"][name])
        for name, value in magnet_before["actor_parameters"].items()
    )


def test_ema_magnet_state_round_trips_with_optimizer_state():
    model = _model()
    trainer = PPOTrainer(model, _config())
    actor_batch, critic_batch = _batches(model)
    result = trainer.update(
        [actor_batch], critic_minibatches=[critic_batch], ema_matches=25
    )
    assert result.committed, result.reason

    restored_model = _model()
    restored_model.load_state_dict(model.state_dict())
    restored = PPOTrainer(restored_model, _config())
    restored.load_optimizer_state_dict(trainer.optimizer_state_dict())

    assert restored.ema_magnet.updates == 1
    assert restored.ema_magnet.completed_matches == 25
    assert restored.ema_magnet.last_tau == pytest.approx(1 - 2 ** (-0.25))
    assert all(
        torch.equal(value, restored.ema_magnet.state_dict()["actor_parameters"][name])
        for name, value in trainer.ema_magnet.state_dict()["actor_parameters"].items()
    )


def test_global_boundary_and_clip_statistics_are_finalized():
    metrics = _finalize_statistic_metrics({}, {
        "boundary_rows": 4,
        "boundary_order_cross_entropy_sum": 6,
        "boundary_order_accuracy_sum": 3,
        "boundary_rank_brier_sum": 2,
        "actor_steps": 4,
        "actor_clipped_steps": 1,
        "critic_steps": 2,
        "critic_clipped_steps": 1,
    })
    assert metrics["boundary_order_cross_entropy"] == pytest.approx(1.5)
    assert metrics["boundary_order_accuracy"] == pytest.approx(0.75)
    assert metrics["boundary_rank_brier"] == pytest.approx(0.5)
    assert metrics["actor_gradient_clip_fraction"] == pytest.approx(0.25)
    assert metrics["critic_gradient_clip_fraction"] == pytest.approx(0.5)


def test_streaming_update_accumulates_chunks_before_one_transactional_step():
    model = _model()
    trainer = PPOTrainer(model, _config(critic_epochs=1))
    actor_batch, critic_batch = _batches(model)
    before = deepcopy(model.state_dict())

    trainer.begin_streaming_update(ema_matches=8, post_kl_probe_rows=2)
    trainer.accumulate_streaming_chunk(
        [actor_batch], critic_minibatches=[critic_batch]
    )
    trainer.accumulate_streaming_chunk(
        [actor_batch], critic_minibatches=[critic_batch]
    )

    assert all(
        torch.equal(value, model.state_dict()[name])
        for name, value in before.items()
    )
    result = trainer.finish_streaming_update()
    assert result.committed, result.reason
    assert result.policy_version == 1
    assert result.epochs == 1
    assert result.metrics["actor_optimization_fraction"] == 1.0
    assert result.metrics["critic_optimization_fraction"] == 1.0
    assert trainer.ema_magnet.updates == 1
    assert trainer.ema_magnet.completed_matches == 8
    assert any(
        not torch.equal(value, model.state_dict()[name])
        for name, value in before.items()
    )


def test_streaming_logical_batch_matches_materialized_global_normalization():
    torch.manual_seed(19)
    initial = _model()
    materialized_model = deepcopy(initial)
    streaming_model = deepcopy(initial)
    config = _config(critic_epochs=1, entropy_floor=0.03)
    materialized = PPOTrainer(materialized_model, config)
    streaming = PPOTrainer(streaming_model, config)

    # Move both live actors away from their identical EMA magnets so the
    # comparison covers magnet-gradient scaling as well as entropy scaling.
    with torch.no_grad():
        for parameter in materialized.actor_parameter_list:
            parameter.add_(torch.randn_like(parameter) * 1e-3)
        streaming_model.load_state_dict(materialized_model.state_dict())

    materialized_actor, materialized_critic = _batches(materialized_model)
    streaming_actor, streaming_critic = _batches(streaming_model)
    # Keep ratios inside the PPO clip interval while making adaptive-KL's
    # gradient nonzero in both paths.
    materialized_actor["old_logp"] += 0.03
    streaming_actor["old_logp"] += 0.03
    raw_chunks = (
        torch.tensor([4.0, 2.0]),
        torch.tensor([-1.0, -3.0]),
    )
    raw = torch.cat(raw_chunks)
    normalized = (raw - raw.mean()) / (raw.std(unbiased=False) + 1e-8)
    materialized_batches = []
    streaming_batches = []
    materialized_critics = []
    streaming_critics = []
    for index, chunk in enumerate(raw_chunks):
        real_batch = deepcopy(materialized_actor)
        real_batch["advantages"] = normalized[index * 2:(index + 1) * 2]
        real_batch["raw_advantages"] = chunk
        materialized_batches.append(real_batch)
        logical_batch = deepcopy(streaming_actor)
        logical_batch["raw_advantages"] = chunk
        streaming_batches.append(logical_batch)
        materialized_critics.append(deepcopy(materialized_critic))
        streaming_critics.append(deepcopy(streaming_critic))

    materialized_result = materialized.update_logical_batch(
        materialized_batches,
        critic_minibatches=materialized_critics,
        ema_matches=8,
    )
    streaming.begin_streaming_update(ema_matches=8, post_kl_probe_rows=4)
    for actor, critic in zip(
        streaming_batches, streaming_critics, strict=True
    ):
        streaming.accumulate_streaming_chunk(
            [actor], critic_minibatches=[critic]
        )
    streaming_result = streaming.finish_streaming_update()

    assert materialized_result.committed, materialized_result.reason
    assert streaming_result.committed, streaming_result.reason
    for metric in (
        "policy_loss",
        "entropy_loss",
        "kl_loss",
        "magnet_loss",
        "actor_total_loss",
        "actor_gradient_norm",
    ):
        assert streaming_result.metrics[metric] == pytest.approx(
            materialized_result.metrics[metric], rel=2e-5, abs=2e-7
        )
    assert materialized_result.metrics["kl_loss"] > 0
    assert materialized_result.metrics["magnet_loss"] > 0
    assert streaming_result.metrics["logical_advantage_mean"] == pytest.approx(
        float(raw.mean())
    )
    assert streaming_result.metrics["logical_advantage_std"] == pytest.approx(
        float(raw.std(unbiased=False))
    )
    assert materialized_result.metrics[
        "logical_advantage_mean"
    ] == pytest.approx(float(raw.mean()))
    assert materialized_result.metrics[
        "logical_advantage_std"
    ] == pytest.approx(float(raw.std(unbiased=False)))
    for name, expected in materialized_model.state_dict().items():
        torch.testing.assert_close(
            streaming_model.state_dict()[name], expected, rtol=2e-5, atol=2e-7
        )


def test_streaming_update_rejects_multi_epoch_critic_replay():
    trainer = PPOTrainer(_model(), _config(critic_epochs=2))
    with pytest.raises(ValueError, match="one accumulated critic pass"):
        trainer.begin_streaming_update(ema_matches=1)
