from copy import deepcopy

import pytest
import torch

import zenith_ppo.ppo.trainer as trainer_module
from zenith_ppo.model.actor_critic import ActorCritic
from zenith_ppo.ppo.loss import ActorLoss
from zenith_ppo.ppo.trainer import PPOTrainer


def _model():
    return ActorCritic({
        "layers": 1, "critic_layers": 1, "d_model": 32,
        "query_heads": 2, "kv_heads": 1, "head_dim": 16,
        "ffn_dim": 64, "context_tokens": 64,
    })


def _config(**changes):
    return {
        "learning_rate": 1e-3, "adam_beta1": .9, "adam_beta2": .999,
        "adam_epsilon": 1e-8, "weight_decay": 0., "epochs": 1,
        "minibatches": 1, "ratio_clip": .2, "value_clip": 0.,
        "score_value_scale": 10., "value_coefficient": .5,
        "entropy_start": .01, "max_grad_norm": 100., "target_kl": 100.,
        "belief_coefficient": .1, "belief_tenpai_coefficient": .25,
    } | changes


def _inputs():
    torch.manual_seed(7)
    return dict(
        token_factors=torch.randint(0, 2, (2, 7, 10)),
        token_numeric=torch.zeros(2, 7, 8), lengths=torch.tensor([7, 6]),
        actor_query_indices=torch.tensor([6, 5]),
        action_factors=torch.randint(0, 2, (2, 3, 15)),
        action_lengths=torch.tensor([3, 2]), action_offsets=torch.tensor([0, 3, 5]),
        oracle_factors=torch.randint(0, 2, (1, 10, 10)),
        oracle_numeric=torch.zeros(1, 10, 8), oracle_lengths=torch.tensor([10]),
        decision_oracle_indices=torch.tensor([0, 0]), decision_seats=torch.tensor([0, 1]),
        backend="eager",
    )


def _batch(model, *, eligible=None):
    inputs = _inputs()
    with torch.no_grad():
        output = model(**inputs)
    selected = torch.tensor([0, 3])
    batch = {
        "model_inputs": inputs, "selected": selected,
        "old_logp": output.log_probabilities[selected].clone(),
        "old_score_values": output.score_values.clone(),
        "old_rank_values": output.rank_values.clone(),
        "advantages": torch.tensor([1., -1.]),
        "score_returns": torch.tensor([2., -1.]),
        "rank_returns": torch.tensor([1., -1.]),
        "rank_targets": torch.tensor([0, 3]),
        "opponent_count_targets": torch.zeros(2, 3, 34, dtype=torch.long),
        "opponent_tenpai_targets": torch.zeros(2, 3),
        "teacher_coefficients": {
            "discard": 0., "reaction": 0., "riichi": 0., "reaction_entropy": 0.,
        },
    }
    if eligible is not None:
        batch["ppo_eligible"] = eligible
    return batch


def _state(parameters):
    return [parameter.detach().clone() for parameter in parameters]


def test_disjoint_optimizers_run_one_actor_and_all_four_critic_epochs():
    model = _model()
    trainer = PPOTrainer(model, _config())
    actor_before = _state(trainer.actor_parameter_list)
    critic_before = _state(trainer.critic_parameter_list)
    result = trainer.update([_batch(model)])
    assert result.committed
    assert result.metrics["actor_optimization_fraction"] == 1.0
    assert result.metrics["critic_optimization_fraction"] == 1.0
    assert result.minibatches == 5
    assert any(not torch.equal(a, b) for a, b in zip(actor_before, trainer.actor_parameter_list))
    assert any(not torch.equal(a, b) for a, b in zip(critic_before, trainer.critic_parameter_list))
    state = trainer.optimizer_state_dict()
    assert state["architecture"] == "contextual-actor-shared-oracle-v1"
    assert set(state) == {"architecture", "actor", "critic"}


def test_early_actor_kl_stop_never_truncates_critic(monkeypatch):
    model = _model()
    trainer = PPOTrainer(model, _config(epochs=2, target_kl=.02))
    calls = 0

    def controlled(new_logp, old_logp, advantages, entropy, **kwargs):
        nonlocal calls
        calls += 1
        total = -(new_logp * advantages).mean()
        zero = total.detach() * 0
        kl = zero if calls == 1 else zero + .1
        return ActorLoss(total, total.detach(), zero, kl, zero)

    monkeypatch.setattr(trainer_module, "actor_loss", controlled)
    result = trainer.update([_batch(model)])
    assert result.committed
    assert result.epochs == 1
    assert result.metrics["kl_early_stop"] == 1.0
    assert result.metrics["actor_optimization_fraction"] == .5
    assert result.metrics["critic_optimization_fraction"] == 1.0
    assert result.minibatches == 5


def test_failure_in_either_update_rolls_back_both_models_and_optimizers(monkeypatch):
    model = _model()
    trainer = PPOTrainer(model, _config())
    model_before = deepcopy(model.state_dict())
    actor_before = deepcopy(trainer.actor_optimizer.state_dict())
    critic_before = deepcopy(trainer.critic_optimizer.state_dict())

    def fail(*args, **kwargs):
        raise FloatingPointError("forced critic failure")

    monkeypatch.setattr(trainer_module, "critic_loss", fail)
    result = trainer.update([_batch(model)])
    assert not result.committed
    assert "forced critic failure" in result.reason
    assert all(torch.equal(value, model.state_dict()[name])
               for name, value in model_before.items())
    assert trainer.actor_optimizer.state_dict() == actor_before
    assert trainer.critic_optimizer.state_dict() == critic_before


def test_ineligible_bot_rows_never_become_critic_targets():
    model = _model()
    trainer = PPOTrainer(model, _config())
    batch = _batch(model, eligible=torch.tensor([True, False]))
    # An invalid bot label would fail categorical supervision if it leaked.
    batch["rank_targets"][1] = -1
    result = trainer.update([batch])
    assert result.committed
    assert result.metrics["critic_optimization_fraction"] == 1.0


def test_old_single_optimizer_checkpoint_is_rejected():
    trainer = PPOTrainer(_model(), _config())
    with pytest.raises(ValueError, match="current actor/oracle architecture"):
        trainer.load_optimizer_state_dict({"state": {}, "param_groups": []})
