"""Transactional AdamW PPO updates."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import torch
from torch.nn import functional as F

from .loss import ppo_loss


@dataclass(frozen=True)
class UpdateResult:
    committed: bool
    policy_version: int
    epochs: int
    minibatches: int
    metrics: dict[str, float]
    reason: str | None = None


class PPOTrainer:
    def __init__(self, model, config, *, device_type="cpu", use_bf16=False):
        self.model, self.config = model, dict(config)
        self.device_type = device_type
        self.use_bf16 = bool(use_bf16 and device_type == "cuda")
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
            betas=(config["adam_beta1"], config["adam_beta2"]), eps=config["adam_epsilon"],
            weight_decay=config["weight_decay"], fused=self.use_bf16)
        self.policy_version = 0

    def update(self, minibatches):
        model_before = {name: value.detach().clone() for name, value in self.model.state_dict().items()}
        optimizer_before = deepcopy(self.optimizer.state_dict())
        count, metric_totals, metrics, completed_epochs = 0, {}, {}, 0
        metric_names = ("policy_loss", "value_loss", "entropy", "entropy_loss",
                        "belief_loss", "count_loss", "tenpai_loss", "count_accuracy",
                        "tenpai_accuracy", "total_loss", "approximate_kl", "clip_fraction",
                        "gradient_norm")

        def read_metrics():
            if not count:
                return {}
            # One device-to-host synchronization replaces one scalar synchronization per
            # metric per minibatch. Epoch boundaries are the only place Python needs values.
            values = torch.stack([metric_totals[name] for name in metric_names]).div(count)
            return dict(zip(metric_names, values.tolist()))

        try:
            for epoch in range(int(self.config["epochs"])):
                for batch in minibatches:
                    self.optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=self.device_type, dtype=torch.bfloat16,
                                        enabled=self.use_bf16):
                        output = self.model(**batch["model_inputs"])
                    selected = batch["selected"]
                    eligible = batch.get("ppo_eligible")
                    if eligible is None:
                        old_logp, advantages = batch["old_logp"], batch["advantages"]
                        values, returns, entropy_values = output.values, batch["returns"], output.entropy
                        belief_available = hasattr(output, "opponent_count_logits") and \
                            "opponent_count_targets" in batch
                        if belief_available:
                            count_logits = output.opponent_count_logits
                            tenpai_logits = output.opponent_tenpai_logits
                            count_targets = batch["opponent_count_targets"]
                            tenpai_targets = batch["opponent_tenpai_targets"]
                    else:
                        if not eligible.any():
                            raise ValueError("minibatch has no current-policy rows")
                        # Historical rows remain available to environment histories but never contribute
                        # to targets, normalization, entropy, or any learner objective.
                        selected = selected[eligible]
                        old_logp, advantages = batch["old_logp"][eligible], batch["advantages"][eligible]
                        values, returns = output.values[eligible], batch["returns"][eligible]
                        entropy_values = output.entropy[eligible]
                        belief_available = hasattr(output, "opponent_count_logits") and \
                            "opponent_count_targets" in batch
                        if belief_available:
                            count_logits = output.opponent_count_logits[eligible]
                            tenpai_logits = output.opponent_tenpai_logits[eligible]
                            count_targets = batch["opponent_count_targets"][eligible]
                            tenpai_targets = batch["opponent_tenpai_targets"][eligible]
                    losses = ppo_loss(output.log_probabilities[selected], old_logp,
                        advantages, values, returns, entropy_values,
                        ratio_clip=self.config["ratio_clip"],
                        value_coefficient=self.config["value_coefficient"],
                        entropy_coefficient=batch.get("entropy_coefficient", self.config["entropy_start"]))
                    if belief_available:
                        count_loss = F.cross_entropy(
                            count_logits.reshape(-1, 5), count_targets.long().reshape(-1)
                        )
                        tenpai_loss = F.binary_cross_entropy_with_logits(
                            tenpai_logits, tenpai_targets.float()
                        )
                        belief_loss = float(self.config.get("belief_coefficient", .10)) * (
                            count_loss + float(self.config.get("belief_tenpai_coefficient", .25)) * tenpai_loss
                        )
                        count_accuracy = (count_logits.argmax(-1) == count_targets).float().mean()
                        tenpai_accuracy = ((tenpai_logits >= 0) == (tenpai_targets >= .5)).float().mean()
                    else:
                        belief_loss = losses.total.new_zeros(())
                        count_loss = tenpai_loss = count_accuracy = tenpai_accuracy = belief_loss
                    total_loss = losses.total + belief_loss
                    total_loss.backward()
                    parameters = [parameter for parameter in self.model.parameters() if parameter.grad is not None]
                    gradient = torch.nn.utils.clip_grad_norm_(parameters, self.config["max_grad_norm"],
                                                              error_if_nonfinite=True)
                    self.optimizer.step()
                    batch_metrics = {"policy_loss": losses.policy.detach(),
                        "value_loss": losses.value.detach(),
                        "entropy": entropy_values.float().mean().detach(),
                        "entropy_loss": losses.entropy_loss.detach(),
                        "belief_loss": belief_loss.detach(),
                        "count_loss": count_loss.detach(),
                        "tenpai_loss": tenpai_loss.detach(),
                        "count_accuracy": count_accuracy.detach(),
                        "tenpai_accuracy": tenpai_accuracy.detach(),
                        "total_loss": total_loss.detach(),
                        "approximate_kl": losses.approximate_kl.detach(),
                        "clip_fraction": losses.clip_fraction.detach(),
                        "gradient_norm": gradient.detach()}
                    count += 1
                    for name, value in batch_metrics.items():
                        metric_totals[name] = metric_totals.get(name, 0.0) + value
                completed_epochs += 1
                metrics = read_metrics()
                if metrics and metrics["approximate_kl"] > self.config["target_kl"]: break
        except Exception as exc:
            metrics = read_metrics()
            self.model.load_state_dict(model_before); self.optimizer.load_state_dict(optimizer_before)
            return UpdateResult(False, self.policy_version, completed_epochs, count, metrics, str(exc))
        self.policy_version += 1
        return UpdateResult(True, self.policy_version, completed_epochs, count, metrics)

    def checkpoint_state(self, *, counters=None, seeds=None, curriculum=None, population=None,
                         rating=None, metrics=None, env=None):
        counters = dict(counters or {})
        return {"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
            "trainer": {"policy_version": self.policy_version, "counters": counters,
                "seeds": seeds.state_dict() if seeds is not None else None,
                "curriculum": curriculum, "population": population, "rating": rating,
                "metrics": metrics, "env": env},
            "state": {"policy_version": self.policy_version,
                "update": int(counters.get("update", self.policy_version)),
                "metric_cursor": metrics or {}}}

    def publish_checkpoint(self, root, *, compatibility, metadata=None, **state):
        """Publish only after a committed update at the caller's safe rollout boundary."""
        if self.policy_version <= 0: raise RuntimeError("cannot checkpoint before a committed update")
        from ..checkpoint import publish
        return publish(root, self.checkpoint_state(**state), compatibility=compatibility,
                       metadata=metadata)

    @staticmethod
    def metric_points(update, result, *, source="update"):
        from ..metric_registry import REGISTRY
        from ..types import MetricPoint
        mapping = {"ppo/policy_loss": result.metrics.get("policy_loss", 0),
            "ppo/value_loss": result.metrics.get("value_loss", 0),
            "ppo/entropy": result.metrics.get("entropy", 0),
            "belief/total_loss": result.metrics.get("belief_loss", 0),
            "belief/count_loss": result.metrics.get("count_loss", 0),
            "belief/tenpai_loss": result.metrics.get("tenpai_loss", 0),
            "belief/count_accuracy": result.metrics.get("count_accuracy", 0),
            "belief/tenpai_accuracy": result.metrics.get("tenpai_accuracy", 0),
            "ppo/total_loss": result.metrics.get("total_loss", 0),
            "ppo/approximate_kl": result.metrics.get("approximate_kl", 0),
            "ppo/clip_fraction": result.metrics.get("clip_fraction", 0),
            "ppo/gradient_norm": result.metrics.get("gradient_norm", 0)}
        return [MetricPoint(name, definition.axis, update, value, definition.unit,
            definition.window, definition.reduction, source) for name, value in mapping.items()
            for definition in (REGISTRY[name],)]
