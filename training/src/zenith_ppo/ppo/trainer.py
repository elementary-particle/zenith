"""Transactional disjoint AdamW updates for actor and oracle critic."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import torch

from .loss import actor_loss, critic_loss
from .objectives import (
    BatchResult,
    MetricAccumulator,
    TEACHER_COEFFICIENTS,
    TEACHER_LOSS_WEIGHTS,
    TEACHER_ROW_METRICS,
    belief_objective,
    learner_rows,
    teacher_objective,
    teacher_row_totals,
)


@dataclass(frozen=True)
class UpdateResult:
    committed: bool
    policy_version: int
    epochs: int
    minibatches: int
    metrics: dict[str, float]
    reason: str | None = None


class PPOTrainer:
    def __init__(self, model, config, *, device_type="cpu", use_bf16=False,
                 profiler=None):
        self.model, self.config = model, dict(config)
        self.device_type = device_type
        self.use_bf16 = bool(use_bf16 and device_type == "cuda")
        common = dict(
            lr=config["learning_rate"],
            betas=(config["adam_beta1"], config["adam_beta2"]),
            eps=config["adam_epsilon"], weight_decay=config["weight_decay"],
            fused=self.use_bf16,
        )
        self.actor_parameter_list = list(model.actor_parameters())
        self.critic_parameter_list = list(model.critic_parameters())
        if set(map(id, self.actor_parameter_list)) & set(map(id, self.critic_parameter_list)):
            raise ValueError("actor and oracle critic parameters must be disjoint")
        if len(self.actor_parameter_list) + len(self.critic_parameter_list) != sum(
            1 for _ in model.parameters()
        ):
            raise ValueError("every model parameter must belong to exactly one optimizer")
        self.actor_optimizer = torch.optim.AdamW(self.actor_parameter_list, **common)
        self.critic_optimizer = torch.optim.AdamW(self.critic_parameter_list, **common)
        self.policy_version = 0
        if profiler is None:
            from ..profiling import StageProfiler
            profiler = StageProfiler()
        self.profiler = profiler

    @staticmethod
    def _select(value, eligible):
        return value if eligible is None else value[eligible]

    def _process_actor(self, batch, *, total_rows, teacher_totals):
        with torch.autocast(
            device_type=self.device_type, dtype=torch.bfloat16,
            enabled=self.use_bf16,
        ):
            output = self.model.forward_actor(**batch["model_inputs"])
        eligible = batch.get("ppo_eligible")
        selected = self._select(batch["selected"], eligible)
        old_logp = self._select(batch["old_logp"], eligible)
        advantages = self._select(batch["advantages"], eligible)
        entropy = self._select(output.entropy, eligible)
        losses = actor_loss(
            output.log_probabilities.index_select(0, selected), old_logp,
            advantages, entropy, ratio_clip=self.config["ratio_clip"],
            entropy_coefficient=batch.get(
                "entropy_coefficient", self.config["entropy_start"]
            ),
        )
        zero = losses.total.new_zeros(())
        belief = belief_objective(output, batch, eligible, self.config, zero)
        auxiliary, teacher_metrics, coefficients = teacher_objective(
            output, batch, eligible, teacher_totals, zero
        )
        row_count = int(old_logp.numel())
        scaled = (losses.total + belief.total) * row_count / total_rows + auxiliary
        metrics = {
            "policy_loss": losses.policy,
            "entropy": entropy.float().mean(),
            "entropy_loss": losses.entropy_loss,
            "belief_loss": belief.total,
            "count_loss": belief.count_loss,
            "tenpai_loss": belief.tenpai_loss,
            "count_accuracy": belief.count_accuracy,
            "tenpai_accuracy": belief.tenpai_accuracy,
            "approximate_kl": losses.approximate_kl,
            "clip_fraction": losses.clip_fraction,
            **teacher_metrics,
            **{
                f"{name}_coefficient": zero.new_tensor(value)
                for name, value in coefficients.items()
            },
        }
        teacher_weights = {
            name: teacher_metrics[row_metric]
            for name, row_metric in TEACHER_LOSS_WEIGHTS.items()
        }
        return BatchResult(
            scaled, losses.approximate_kl.detach() * row_count / total_rows,
            auxiliary, row_count, metrics, teacher_weights,
        )

    def _process_critic(self, batch, *, total_rows):
        with torch.autocast(
            device_type=self.device_type, dtype=torch.bfloat16,
            enabled=self.use_bf16,
        ):
            output = self.model.forward_critic(**batch["model_inputs"])
        eligible = batch.get("ppo_eligible")
        select = lambda value: self._select(value, eligible)
        losses = critic_loss(
            select(output.score_values), select(batch["old_score_values"]),
            select(batch["score_returns"]), select(output.rank_logits),
            select(batch["rank_targets"]), select(batch["rank_returns"]),
            score_value_scale=self.config["score_value_scale"],
            value_clip=self.config["value_clip"],
            value_coefficient=self.config["value_coefficient"],
        )
        row_count = int(select(batch["score_returns"]).numel())
        metrics = {
            "critic_loss": losses.total,
            "score_value_loss": losses.score_mse,
            "score_normalized_mse": losses.score_mse,
            "score_raw_mae": losses.score_mae,
            "score_raw_rmse": losses.score_rmse,
            "score_explained_variance": losses.score_explained_variance,
            "score_value_clip_fraction": losses.score_clip_fraction,
            "rank_value_loss": losses.rank_cross_entropy,
            "rank_cross_entropy": losses.rank_cross_entropy,
            "rank_accuracy": losses.rank_accuracy,
            "rank_brier": losses.rank_brier,
            "rank_utility_explained_variance": (
                losses.rank_utility_explained_variance
            ),
        }
        return BatchResult(
            losses.total * row_count / total_rows,
            losses.total.new_zeros(()), losses.total.new_zeros(()),
            row_count, metrics, {},
        )

    @staticmethod
    def _record_batch_metrics(accumulator, batch_result):
        coefficient_metrics = {f"{name}_coefficient" for name in TEACHER_COEFFICIENTS}
        for name, value in batch_result.metrics.items():
            if name in batch_result.teacher_weights:
                weight = batch_result.teacher_weights[name]
                if weight:
                    accumulator.add(name, value, weight)
            elif name in TEACHER_ROW_METRICS:
                accumulator.add(name, value)
            elif name in coefficient_metrics:
                accumulator.set(name, value)
            else:
                accumulator.add(name, value, batch_result.row_count)

    @staticmethod
    def _optimizer_groups(packed_batches, requested_groups):
        group_count = min(int(requested_groups), len(packed_batches))
        groups = [[] for _ in range(group_count)]
        row_counts = [0] * group_count
        for batch in packed_batches:
            index = min(range(group_count), key=lambda row: (row_counts[row], row))
            groups[index].append(batch)
            row_counts[index] += learner_rows(batch)
        return tuple(tuple(group) for group in groups)

    def update(self, minibatches, *, rng=None):
        minibatches = tuple(minibatches)
        if not minibatches:
            return UpdateResult(False, self.policy_version, 0, 0, {},
                                "PPO update has no minibatches")
        total_rows = sum(map(learner_rows, minibatches))
        if total_rows <= 0:
            return UpdateResult(False, self.policy_version, 0, 0, {},
                                "PPO update has no learner rows")
        model_before = {
            name: value.detach().clone() for name, value in self.model.state_dict().items()
        }
        actor_optimizer_before = deepcopy(self.actor_optimizer.state_dict())
        critic_optimizer_before = deepcopy(self.critic_optimizer.state_dict())
        self.model.train()
        template = next(self.model.parameters()).new_zeros(())
        accumulator = MetricAccumulator(template)
        actor_steps = critic_steps = actor_epochs = count = 0
        kl_early_stop = False
        kl_stop_value = 0.0
        optimizer_minibatches = min(
            int(self.config.get("minibatches", 1)), len(minibatches)
        )
        actor_planned_epochs = int(self.config["epochs"])
        critic_planned_epochs = 4
        actor_planned_steps = actor_planned_epochs * optimizer_minibatches
        critic_planned_steps = critic_planned_epochs * optimizer_minibatches

        def shuffled_groups():
            order = list(range(len(minibatches)))
            if rng is not None:
                rng.shuffle(order)
            return self._optimizer_groups(
                tuple(minibatches[index] for index in order), optimizer_minibatches
            )

        def read_metrics():
            result = accumulator.read()
            result.update({
                "epochs_completed": float(actor_epochs),
                "actor_optimization_fraction": actor_steps / max(1, actor_planned_steps),
                "critic_optimization_fraction": critic_steps / max(1, critic_planned_steps),
                "optimization_fraction": actor_steps / max(1, actor_planned_steps),
                "kl_early_stop": float(kl_early_stop),
                "kl_stop_value": float(kl_stop_value),
            })
            return result

        try:
            for _ in range(actor_planned_epochs):
                for group in shuffled_groups():
                    group_rows = sum(map(learner_rows, group))
                    totals = teacher_row_totals(group)
                    self.actor_optimizer.zero_grad(set_to_none=True)
                    results = []
                    observed_kl = template.clone()
                    total_loss = template.clone()
                    auxiliary = template.clone()
                    for batch in group:
                        with self.profiler.measure("ppo.actor_optimization"):
                            result = self._process_actor(
                                batch, total_rows=group_rows, teacher_totals=totals
                            )
                        result.loss.backward()
                        observed_kl += result.approximate_kl
                        total_loss += result.loss.detach()
                        auxiliary += result.auxiliary_loss.detach()
                        results.append(result)
                    observed = float(observed_kl)
                    if observed > float(self.config["target_kl"]):
                        self.actor_optimizer.zero_grad(set_to_none=True)
                        if actor_steps == 0:
                            raise RuntimeError(
                                "target KL exceeded before the first actor optimizer step; "
                                "rollout behavior log-probabilities do not match the model"
                            )
                        kl_early_stop, kl_stop_value = True, observed
                        break
                    gradient = torch.nn.utils.clip_grad_norm_(
                        [p for p in self.actor_parameter_list if p.grad is not None],
                        self.config["max_grad_norm"], error_if_nonfinite=True,
                    )
                    self.actor_optimizer.step()
                    actor_steps += 1
                    count += len(results)
                    for result in results:
                        self._record_batch_metrics(accumulator, result)
                    accumulator.add("actor_total_loss", total_loss)
                    accumulator.add("auxiliary_loss", auxiliary)
                    accumulator.add("actor_gradient_norm", gradient)
                if kl_early_stop:
                    break
                actor_epochs += 1

            # Critic work is intentionally outside the actor KL-controlled loop.
            for _ in range(critic_planned_epochs):
                for group in shuffled_groups():
                    group_rows = sum(map(learner_rows, group))
                    self.critic_optimizer.zero_grad(set_to_none=True)
                    results = []
                    total_loss = template.clone()
                    for batch in group:
                        with self.profiler.measure("ppo.critic_optimization"):
                            result = self._process_critic(batch, total_rows=group_rows)
                        result.loss.backward()
                        total_loss += result.loss.detach()
                        results.append(result)
                    gradient = torch.nn.utils.clip_grad_norm_(
                        [p for p in self.critic_parameter_list if p.grad is not None],
                        self.config["max_grad_norm"], error_if_nonfinite=True,
                    )
                    self.critic_optimizer.step()
                    critic_steps += 1
                    count += len(results)
                    for result in results:
                        self._record_batch_metrics(accumulator, result)
                    accumulator.add("critic_total_loss", total_loss)
                    accumulator.add("critic_gradient_norm", gradient)
            if not actor_steps or critic_steps != critic_planned_steps:
                raise RuntimeError("actor or oracle critic update was incomplete")
            if not all(
                bool(torch.isfinite(parameter).all()) for parameter in self.model.parameters()
            ):
                raise FloatingPointError("optimizer produced non-finite model parameters")
            metrics = read_metrics()
        except Exception as exc:
            metrics = read_metrics()
            self.model.load_state_dict(model_before)
            self.actor_optimizer.load_state_dict(actor_optimizer_before)
            self.critic_optimizer.load_state_dict(critic_optimizer_before)
            return UpdateResult(False, self.policy_version, actor_epochs, count, metrics, str(exc))
        self.policy_version += 1
        return UpdateResult(True, self.policy_version, actor_epochs, count, metrics)

    def optimizer_state_dict(self):
        return {
            "architecture": "contextual-actor-shared-oracle-v1",
            "actor": self.actor_optimizer.state_dict(),
            "critic": self.critic_optimizer.state_dict(),
        }

    def load_optimizer_state_dict(self, state):
        if state.get("architecture") != "contextual-actor-shared-oracle-v1":
            raise ValueError("checkpoint is not the current actor/oracle architecture")
        self.actor_optimizer.load_state_dict(state["actor"])
        self.critic_optimizer.load_state_dict(state["critic"])

    def checkpoint_state(self, *, counters=None, seeds=None, curriculum=None,
                         population=None, rating=None, metrics=None, env=None):
        counters = dict(counters or {})
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer_state_dict(),
            "trainer": {
                "architecture": "contextual-actor-shared-oracle-v1",
                "policy_version": self.policy_version, "counters": counters,
                "seeds": seeds.state_dict() if seeds is not None else None,
                "curriculum": curriculum, "population": population, "rating": rating,
                "metrics": metrics, "env": env,
            },
            "state": {
                "architecture": "contextual-actor-shared-oracle-v1",
                "policy_version": self.policy_version,
                "update": int(counters.get("update", self.policy_version)),
                "metric_cursor": metrics or {},
            },
        }

    def publish_checkpoint(self, root, *, metadata=None, **state):
        if self.policy_version <= 0:
            raise RuntimeError("cannot checkpoint before a committed update")
        from ..checkpoint import publish
        return publish(root, self.checkpoint_state(**state), metadata=metadata)

    @staticmethod
    def metric_points(update, result, *, source="update"):
        from ..metric_registry import REGISTRY
        from ..types import MetricPoint
        mapping = {
            "ppo/policy_loss": result.metrics.get("policy_loss", 0),
            "ppo/critic_loss": result.metrics.get("critic_loss", 0),
            "ppo/score_value_loss": result.metrics.get("score_value_loss", 0),
            "ppo/rank_value_loss": result.metrics.get("rank_value_loss", 0),
            "ppo/score_normalized_mse": result.metrics.get("score_normalized_mse", 0),
            "ppo/score_raw_mae": result.metrics.get("score_raw_mae", 0),
            "ppo/score_raw_rmse": result.metrics.get("score_raw_rmse", 0),
            "ppo/score_critic_explained_variance": result.metrics.get(
                "score_explained_variance", 0
            ),
            "ppo/rank_cross_entropy": result.metrics.get("rank_cross_entropy", 0),
            "ppo/rank_accuracy": result.metrics.get("rank_accuracy", 0),
            "ppo/rank_brier": result.metrics.get("rank_brier", 0),
            "ppo/rank_utility_explained_variance": result.metrics.get(
                "rank_utility_explained_variance", 0
            ),
            "ppo/entropy": result.metrics.get("entropy", 0),
            "belief/total_loss": result.metrics.get("belief_loss", 0),
            "belief/count_loss": result.metrics.get("count_loss", 0),
            "belief/tenpai_loss": result.metrics.get("tenpai_loss", 0),
            "belief/count_accuracy": result.metrics.get("count_accuracy", 0),
            "belief/tenpai_accuracy": result.metrics.get("tenpai_accuracy", 0),
            "teacher/auxiliary_loss": result.metrics.get("auxiliary_loss", 0),
            "teacher/discard_loss": result.metrics.get("discard_teacher_loss", 0),
            "teacher/reaction_loss": result.metrics.get("reaction_teacher_loss", 0),
            "teacher/reaction_entropy": result.metrics.get("reaction_entropy", 0),
            "teacher/riichi_loss": result.metrics.get("riichi_teacher_loss", 0),
            "teacher/discard_coefficient": result.metrics.get("discard_coefficient", 0),
            "teacher/reaction_coefficient": result.metrics.get("reaction_coefficient", 0),
            "teacher/riichi_coefficient": result.metrics.get("riichi_coefficient", 0),
            "teacher/reaction_entropy_coefficient": result.metrics.get(
                "reaction_entropy_coefficient", 0
            ),
            "ppo/total_loss": result.metrics.get("actor_total_loss", 0),
            "ppo/approximate_kl": result.metrics.get("approximate_kl", 0),
            "ppo/kl_stop_value": result.metrics.get("kl_stop_value", 0),
            "ppo/kl_early_stop": result.metrics.get("kl_early_stop", 0),
            "ppo/optimization_fraction": result.metrics.get("optimization_fraction", 0),
            "ppo/actor_optimization_fraction": result.metrics.get(
                "actor_optimization_fraction", 0
            ),
            "ppo/critic_optimization_fraction": result.metrics.get(
                "critic_optimization_fraction", 0
            ),
            "ppo/clip_fraction": result.metrics.get("clip_fraction", 0),
            "ppo/score_value_clip_fraction": result.metrics.get(
                "score_value_clip_fraction", 0
            ),
            "ppo/gradient_norm": result.metrics.get("actor_gradient_norm", 0),
            "ppo/actor_gradient_norm": result.metrics.get("actor_gradient_norm", 0),
            "ppo/critic_gradient_norm": result.metrics.get("critic_gradient_norm", 0),
            "ppo/gradient_clip_fraction": 0,
        }
        return [
            MetricPoint(name, definition.axis, update, value, definition.unit,
                        definition.window, definition.reduction, source)
            for name, value in mapping.items() for definition in (REGISTRY[name],)
        ]
