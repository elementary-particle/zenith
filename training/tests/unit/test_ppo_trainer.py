from types import SimpleNamespace

import pytest
import torch

import zenith_ppo.ppo.trainer as trainer_module
from zenith_ppo.ppo.loss import PPOLoss
from zenith_ppo.ppo.trainer import PPOTrainer


class _MetricModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, entropy):
        return SimpleNamespace(
            log_probabilities=self.weight.expand(1),
            values=self.weight.expand(1),
            entropy=entropy,
        )


def _config():
    return {
        "learning_rate": 1e-3,
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_epsilon": 1e-8,
        "weight_decay": 0.0,
        "epochs": 2,
        "ratio_clip": 0.2,
        "value_coefficient": 0.5,
        "entropy_start": 0.01,
        "max_grad_norm": 100.0,
        "target_kl": 100.0,
    }


def _batch(entropy):
    return {
        "model_inputs": {"entropy": torch.tensor([entropy])},
        "selected": torch.tensor([0]),
        "old_logp": torch.zeros(1),
        "advantages": torch.ones(1),
        "returns": torch.zeros(1),
    }


def test_update_metrics_average_every_minibatch_and_expose_entropy_and_gradient(monkeypatch):
    def fake_loss(new_logp, old_logp, advantages, new_values, returns, entropy, **kwargs):
        marker = entropy.float().mean()
        # Keep the reported total deterministic while giving its backward pass norm ``marker``.
        total = marker + marker * (new_logp.mean() - new_logp.mean().detach())
        return PPOLoss(total, marker, marker + 10, -marker, marker / 100, marker / 10)

    monkeypatch.setattr(trainer_module, "ppo_loss", fake_loss)
    trainer = PPOTrainer(_MetricModel(), _config())

    result = trainer.update([_batch(1.0), _batch(3.0)])

    assert result.committed
    assert result.epochs == 2
    assert result.minibatches == 4
    assert result.metrics == pytest.approx({
        "policy_loss": 2.0,
        "value_loss": 12.0,
        "entropy": 2.0,
        "entropy_loss": -2.0,
        "belief_loss": 0.0,
        "count_loss": 0.0,
        "tenpai_loss": 0.0,
        "count_accuracy": 0.0,
        "tenpai_accuracy": 0.0,
        "total_loss": 2.0,
        "approximate_kl": 0.02,
        "clip_fraction": 0.2,
        "gradient_norm": 2.0,
    })
    assert {point.name: point.value for point in trainer.metric_points(1, result)}["ppo/entropy"] == 2.0


def test_belief_loss_uses_fixed_weight_and_reports_canonical_components(monkeypatch):
    class BeliefModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.0))

        def forward(self):
            return SimpleNamespace(
                log_probabilities=self.weight.expand(1), values=self.weight.expand(1),
                entropy=self.weight.expand(1),
                opponent_count_logits=self.weight.expand(1, 3, 34, 5),
                opponent_tenpai_logits=self.weight.expand(1, 3),
            )

    def zero_ppo(new_logp, *args, **kwargs):
        zero = new_logp.sum() * 0
        return PPOLoss(zero, zero, zero, zero, zero, zero)

    monkeypatch.setattr(trainer_module, "ppo_loss", zero_ppo)
    config = _config() | {"epochs": 1, "belief_coefficient": .10,
                          "belief_tenpai_coefficient": .25}
    batch = _batch(0.0) | {
        "model_inputs": {},
        "opponent_count_targets": torch.zeros(1, 3, 34, dtype=torch.long),
        "opponent_tenpai_targets": torch.ones(1, 3),
    }
    result = PPOTrainer(BeliefModel(), config).update([batch])
    assert result.committed
    assert result.metrics["count_loss"] == pytest.approx(torch.log(torch.tensor(5.)).item())
    assert result.metrics["tenpai_loss"] == pytest.approx(torch.log(torch.tensor(2.)).item())
    assert result.metrics["belief_loss"] == pytest.approx(
        .10 * (result.metrics["count_loss"] + .25 * result.metrics["tenpai_loss"])
    )
    assert result.metrics["total_loss"] == pytest.approx(result.metrics["belief_loss"])
