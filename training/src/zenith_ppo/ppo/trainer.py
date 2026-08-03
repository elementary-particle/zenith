"""Transactional PPO updates for the policy and boundary-rank critic."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, fields, is_dataclass, replace

import torch
from torch.nn import functional as F

from .ema_magnet import EMAMagnet
from .loss import (
    actor_loss,
    conditional_family_entropy,
    segmented_forward_kl,
)
from .objectives import BatchResult, MetricAccumulator, learner_rows


@dataclass(frozen=True)
class UpdateResult:
    committed: bool
    policy_version: int
    epochs: int
    minibatches: int
    metrics: dict[str, float]
    reason: str | None = None
    metric_statistics: dict[str, float] = field(default_factory=dict)


def _finalize_statistic_metrics(metrics, statistics):
    """Derive sparse metrics from additive update-wide statistics."""
    boundary_rows = float(statistics.get("boundary_rows", 0.0))
    for name in (
        "boundary_order_cross_entropy",
        "boundary_order_accuracy",
        "boundary_rank_brier",
    ):
        metrics[name] = (
            float(statistics.get(f"{name}_sum", 0.0)) / boundary_rows
            if boundary_rows else 0.0
        )
    for owner in ("actor", "critic"):
        steps = float(statistics.get(f"{owner}_steps", 0.0))
        metrics[f"{owner}_gradient_clip_fraction"] = (
            float(statistics.get(f"{owner}_clipped_steps", 0.0)) / steps
            if steps else 0.0
        )
    metrics["gradient_clip_fraction"] = metrics["actor_gradient_clip_fraction"]
    return metrics


class PPOTrainer:
    """Update all policy parameters from low-bias boundary-rank credit."""

    def __init__(
        self, model, config, *, device_type="cpu", use_bf16=False,
        profiler=None,
    ):
        self.model = model
        self.config = dict(config)
        self.architecture = "shared-shape-emagnet-current-kyoku-ppo-v1"
        self.device_type = device_type
        self.device = next(model.parameters()).device
        self.use_bf16 = bool(use_bf16 and device_type == "cuda")
        common = dict(
            betas=(config["adam_beta1"], config["adam_beta2"]),
            eps=config["adam_epsilon"],
            weight_decay=config["weight_decay"],
            fused=self.use_bf16,
        )
        self.actor_parameter_list = list(model.actor_parameters())
        self.critic_parameter_list = list(model.critic_parameters())
        actor_ids = set(map(id, self.actor_parameter_list))
        critic_ids = set(map(id, self.critic_parameter_list))
        if actor_ids & critic_ids:
            raise ValueError("actor and critic parameters must be disjoint")
        if len(actor_ids | critic_ids) != sum(1 for _ in model.parameters()):
            raise ValueError("every parameter must belong to one optimizer")
        self.actor_optimizer = torch.optim.AdamW(
            self.actor_parameter_list,
            lr=float(config["actor_learning_rate"]),
            **common,
        )
        self.critic_optimizer = torch.optim.AdamW(
            self.critic_parameter_list,
            lr=float(config["critic_learning_rate"]),
            **common,
        )
        self.policy_version = 0
        self.rollout_rank_explained_variance = None
        self.kl_coefficient = float(config["kl_coefficient_initial"])
        self.ema_magnet = EMAMagnet(
            model,
            half_life_matches=float(config["magnet_half_life_matches"]),
        )
        self._streaming_update = None
        if profiler is None:
            from ..profiling import StageProfiler
            profiler = StageProfiler()
        self.profiler = profiler

    def set_rollout_critic_evidence(self, *, rank_explained_variance):
        """Record critic telemetry; never gate or freeze policy learning."""
        value = float(rank_explained_variance)
        if not torch.isfinite(torch.tensor(value)):
            raise ValueError("rollout rank explained variance must be finite")
        self.rollout_rank_explained_variance = value

    def reset_magnet(self):
        """Start a fresh magnet from externally loaded actor weights."""
        self.ema_magnet.reset(self.model)

    def _to_device(self, value):
        if torch.is_tensor(value):
            return value.to(
                device=self.device,
                non_blocking=(
                    self.device.type == "cuda" and value.device.type == "cpu"
                ),
            )
        if is_dataclass(value):
            return replace(value, **{
                field.name: self._to_device(getattr(value, field.name))
                for field in fields(value)
            })
        if isinstance(value, dict):
            return {key: self._to_device(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return tuple(self._to_device(item) for item in value)
        if isinstance(value, list):
            return [self._to_device(item) for item in value]
        return value

    def _materialize_batch(self, batch):
        if "encoded_rows" not in batch:
            return self._to_device(batch)
        from ..encoding.packing import model_batch

        materialized = dict(batch)
        encoded = materialized.pop("encoded_rows")
        backend = materialized.pop("backend", "sdpa")
        include_actions = materialized.pop("include_actions", True)
        boundary_only = materialized.pop("boundary_only", False)
        if boundary_only:
            import numpy as np

            materialized["model_inputs"] = {
                "decision_seats": torch.as_tensor(
                    [row.decision_seat for row in encoded],
                    dtype=torch.long,
                    device=self.device,
                ),
                "rank_boundary_features": torch.as_tensor(
                    np.stack([row.rank_boundary_features for row in encoded]),
                    dtype=torch.float32,
                    device=self.device,
                ),
                "backend": backend,
            }
        else:
            materialized["model_inputs"] = model_batch(
                encoded,
                device=self.device,
                backend=backend,
                include_actions=include_actions,
            )
        return self._to_device(materialized)

    @staticmethod
    def _select(value, eligible):
        return value if eligible is None else value[eligible]

    def _entropy_normalizers(self, batch):
        if "model_inputs" not in batch:
            applicable = []
            for row in batch["encoded_rows"]:
                kinds = [int(factors[0]) for factors in row.action_factors]
                families = [3 if 3 <= kind <= 5 else kind for kind in kinds]
                applicable.append(any(
                    families.count(family) > 1 for family in set(families)
                ))
            return torch.tensor(applicable, dtype=torch.float32)
        factors = batch["model_inputs"]["action_factors"]
        lengths = batch["model_inputs"]["action_lengths"]
        valid = torch.arange(
            factors.shape[1], device=lengths.device
        )[None] < lengths[:, None]
        kinds = factors[..., 0].long()
        families = torch.where(
            kinds.ge(3) & kinds.le(5), torch.full_like(kinds, 3), kinds
        )
        applicable = torch.zeros_like(lengths, dtype=torch.bool)
        for family in range(11):
            applicable |= ((families.eq(family) & valid).sum(-1) > 1)
        return self._select(
            applicable.float(), batch.get("ppo_eligible")
        )

    def _process_actor(self, batch, *, total_rows, total_entropy_rows):
        batch = self._materialize_batch(batch)
        with torch.autocast(
            device_type=self.device_type,
            dtype=torch.bfloat16,
            enabled=self.use_bf16,
        ):
            output = self.model.forward_actor(**batch["model_inputs"])
            with torch.no_grad():
                magnet_output = self.ema_magnet.forward(batch["model_inputs"])
        eligible = batch.get("ppo_eligible")
        selected = self._select(batch["selected"], eligible)
        old_logp = self._select(batch["old_logp"], eligible).float()
        advantages = self._select(batch["advantages"], eligible).float()
        conditional_entropy, _ = conditional_family_entropy(
            output.log_probabilities,
            batch["model_inputs"]["action_offsets"],
            batch["model_inputs"]["action_factors"],
            batch["model_inputs"]["action_lengths"],
        )
        entropy = self._select(conditional_entropy, eligible)
        entropy_normalizers = self._entropy_normalizers(batch)
        magnet_kl_rows = segmented_forward_kl(
            magnet_output.log_probabilities,
            output.log_probabilities,
            batch["model_inputs"]["action_offsets"],
        )
        magnet_kl_rows = self._select(magnet_kl_rows, eligible)
        magnet_kl = magnet_kl_rows.mean()
        losses = actor_loss(
            output.log_probabilities.index_select(0, selected),
            old_logp,
            advantages,
            entropy,
            entropy_normalizers,
            ratio_clip=float(self.config["ratio_clip"]),
            entropy_coefficient=float(self.config["entropy_floor"]),
            kl_coefficient=self.kl_coefficient,
            magnet_kl=magnet_kl,
            magnet_coefficient=float(self.config["magnet_kl_coefficient"]),
        )
        row_count = int(selected.numel())
        entropy_rows = int(losses.entropy_rows)
        scaled = (
            losses.policy + losses.kl_loss + losses.magnet_loss
        ) * row_count / total_rows
        if total_entropy_rows:
            scaled = scaled + losses.entropy_loss \
                * entropy_rows / total_entropy_rows
        selected_new = output.log_probabilities.index_select(0, selected).float()
        log_ratio = selected_new - old_logp
        metrics = {
            "policy_loss": losses.policy,
            "entropy": entropy.mean() if entropy.numel() else entropy.sum(),
            "entropy_efficiency": losses.entropy_efficiency,
            "entropy_applicable_rows": entropy_rows,
            "entropy_loss": losses.entropy_loss,
            "kl_loss": losses.kl_loss,
            "magnet_kl": losses.magnet_kl,
            "magnet_loss": losses.magnet_loss,
            "approximate_kl": losses.approximate_kl,
            "clip_fraction": losses.clip_fraction,
        }
        statistics = {
            "policy_old_logp_sum": old_logp.detach().sum(),
            "policy_new_logp_sum": selected_new.detach().sum(),
            "policy_log_ratio_sum": log_ratio.detach().sum(),
            "policy_log_ratio_abs_sum": log_ratio.detach().abs().sum(),
            "policy_rows": row_count,
        }
        return BatchResult(
            scaled,
            losses.approximate_kl.detach() * row_count / total_rows,
            row_count,
            metrics,
            statistics,
        )

    def _process_critic(self, batch, *, total_rows):
        batch = self._materialize_batch(batch)
        with torch.autocast(
            device_type=self.device_type,
            dtype=torch.bfloat16,
            enabled=self.use_bf16,
        ):
            output = self.model.forward_critic(**batch["model_inputs"])
        eligible = batch.get("ppo_eligible")
        boundary = self._select(
            batch["rank_boundary_supervision"], eligible
        ).bool()
        if not bool(boundary.all()):
            raise ValueError("rank critic batches must contain boundary rows only")
        targets = self._select(batch["rank_order_targets"], eligible).long()
        logits = self._select(output.rank_order_logits, eligible)[boundary].float()
        targets = targets[boundary]
        row_count = int(targets.numel())
        if not row_count:
            raise ValueError("rank critic batch has no supervised boundaries")
        nll = F.cross_entropy(logits, targets)
        accuracy = logits.argmax(-1).eq(targets).float().mean()
        target_marginals = self.model.match_boundary_critic \
            .order_to_marginals.index_select(0, targets)
        predicted_marginals = self._select(
            output.rank_marginals, eligible
        )[boundary].float()
        brier = (
            predicted_marginals - target_marginals
        ).square().sum((-1, -2)).mean()
        loss = float(self.config["boundary_rank_coefficient"]) * nll
        return BatchResult(
            loss * row_count / total_rows,
            loss.new_zeros(()),
            row_count,
            {
                "critic_loss": loss,
                "rank_cross_entropy": nll,
                "rank_accuracy": accuracy,
                "rank_brier": brier,
            },
            {
                "boundary_rows": row_count,
                "boundary_order_cross_entropy_sum": nll.detach() * row_count,
                "boundary_order_accuracy_sum": accuracy.detach() * row_count,
                "boundary_rank_brier_sum": brier.detach() * row_count,
            },
        )

    def _post_update_kl(self, minibatches):
        total = count = 0.0
        with torch.no_grad():
            for batch in minibatches:
                batch = self._materialize_batch(batch)
                with torch.autocast(
                    device_type=self.device_type,
                    dtype=torch.bfloat16,
                    enabled=self.use_bf16,
                ):
                    output = self.model.forward_actor(**batch["model_inputs"])
                eligible = batch.get("ppo_eligible")
                selected = self._select(batch["selected"], eligible)
                old_logp = self._select(batch["old_logp"], eligible).float()
                new_logp = output.log_probabilities.index_select(
                    0, selected
                ).float()
                estimate = torch.expm1(new_logp - old_logp) - (new_logp - old_logp)
                total += float(estimate.sum())
                count += int(estimate.numel())
        return total / max(1, count)

    @staticmethod
    def _record_batch_metrics(accumulator, result):
        for name, value in result.metrics.items():
            if name == "entropy_applicable_rows":
                accumulator.accumulate(name, value)
            elif name in {"entropy_efficiency", "entropy_loss"}:
                accumulator.add(
                    name, value, result.metrics["entropy_applicable_rows"]
                )
            else:
                accumulator.add(name, value, result.row_count)
        for name, value in result.statistics.items():
            accumulator.accumulate(name, value)

    @staticmethod
    def _optimizer_groups(packed_batches, requested_groups):
        group_count = min(int(requested_groups), len(packed_batches))
        groups = [[] for _ in range(group_count)]
        counts = [0] * group_count
        for batch in packed_batches:
            index = min(range(group_count), key=lambda row: (counts[row], row))
            groups[index].append(batch)
            counts[index] += learner_rows(batch)
        return tuple(tuple(group) for group in groups)

    def update(
        self, minibatches, *, critic_minibatches=None, rng=None,
        actor_rng=None, critic_rng=None, ema_matches=1,
    ):
        minibatches = tuple(minibatches)
        if not minibatches:
            return UpdateResult(
                False, self.policy_version, 0, 0, {},
                "PPO update has no minibatches",
            )
        total_rows = sum(map(learner_rows, minibatches))
        if total_rows <= 0:
            return UpdateResult(
                False, self.policy_version, 0, 0, {},
                "PPO update has no learner rows",
            )
        critic_minibatches = tuple(critic_minibatches or minibatches)
        total_critic_rows = sum(map(learner_rows, critic_minibatches))
        if total_critic_rows <= 0:
            return UpdateResult(
                False, self.policy_version, 0, 0, {},
                "PPO update has no critic rows",
            )
        # Validate the cadence before any optimizer mutation.  The actual EMA
        # update happens only after the actor and critic transaction succeeds.
        self.ema_magnet.tau_for_matches(ema_matches)
        model_before = {
            name: value.detach().clone()
            for name, value in self.model.state_dict().items()
        }
        actor_optimizer_before = deepcopy(self.actor_optimizer.state_dict())
        critic_optimizer_before = deepcopy(self.critic_optimizer.state_dict())
        kl_before = self.kl_coefficient
        self.model.train()
        template = next(self.model.parameters()).new_zeros(())
        accumulator = MetricAccumulator(template)
        actor_steps = critic_steps = processed = completed_epochs = 0
        actor_groups = min(int(self.config["minibatches"]), len(minibatches))
        critic_groups = min(
            int(self.config["minibatches"]), len(critic_minibatches)
        )
        actor_rng = rng if actor_rng is None else actor_rng
        critic_rng = rng if critic_rng is None else critic_rng

        def shuffled_groups(values, count, generator):
            order = list(range(len(values)))
            if generator is not None:
                generator.shuffle(order)
            return self._optimizer_groups(
                tuple(values[index] for index in order), count
            )

        def metrics():
            result = _finalize_statistic_metrics(
                accumulator.read(), accumulator.statistic_values()
            )
            result.update({
                "entropy_applicable_rows": float(
                    accumulator.statistics.get("entropy_applicable_rows", 0)
                ),
                "actor_optimization_fraction": actor_steps / max(
                    1, int(self.config["epochs"]) * actor_groups
                ),
                "critic_optimization_fraction": critic_steps / max(
                    1, int(self.config["critic_epochs"]) * critic_groups
                ),
                "optimization_fraction": actor_steps / max(
                    1, int(self.config["epochs"]) * actor_groups
                ),
            })
            return result

        try:
            with self.profiler.measure("ppo.pre_update_kl"):
                pre_update_kl = self._post_update_kl(minibatches)
            accumulator.set("pre_update_approximate_kl", pre_update_kl)
            replay_limit = max(
                0.5 * float(self.config["target_kl"]), 1e-4
            )
            if pre_update_kl > replay_limit:
                raise RuntimeError(
                    "rollout policy differs before optimization: "
                    f"KL {pre_update_kl:.6g} exceeds {replay_limit:.6g}"
                )
            for _ in range(int(self.config["epochs"])):
                for group in shuffled_groups(minibatches, actor_groups, actor_rng):
                    group_rows = sum(map(learner_rows, group))
                    entropy_rows = sum(
                        int((self._entropy_normalizers(batch) > 0).sum())
                        for batch in group
                    )
                    self.actor_optimizer.zero_grad(set_to_none=True)
                    total_loss = template.clone()
                    for batch in group:
                        with self.profiler.measure("ppo.actor_optimization"):
                            result = self._process_actor(
                                batch,
                                total_rows=group_rows,
                                total_entropy_rows=entropy_rows,
                            )
                        result.loss.backward()
                        total_loss += result.loss.detach()
                        self._record_batch_metrics(accumulator, result)
                        processed += 1
                    gradient = torch.nn.utils.clip_grad_norm_(
                        self.actor_parameter_list,
                        float(self.config["max_grad_norm"]),
                        error_if_nonfinite=True,
                    )
                    self.actor_optimizer.step()
                    actor_steps += 1
                    accumulator.add("actor_total_loss", total_loss)
                    accumulator.add("actor_gradient_norm", gradient)
                    accumulator.accumulate("actor_steps", 1)
                    accumulator.accumulate(
                        "actor_clipped_steps",
                        float(gradient > float(self.config["max_grad_norm"])),
                    )
                completed_epochs += 1
            with self.profiler.measure("ppo.post_update_kl"):
                post_kl = self._post_update_kl(minibatches)
            accumulator.set("post_update_approximate_kl", post_kl)
            hard_limit = max(
                4.0 * float(self.config["target_kl"]), 1e-4
            )
            if post_kl > hard_limit:
                raise RuntimeError(
                    f"post-update KL {post_kl:.6g} exceeds {hard_limit:.6g}"
                )

            for _ in range(int(self.config["critic_epochs"])):
                for group in shuffled_groups(
                    critic_minibatches, critic_groups, critic_rng
                ):
                    group_rows = sum(map(learner_rows, group))
                    self.critic_optimizer.zero_grad(set_to_none=True)
                    total_loss = template.clone()
                    for batch in group:
                        with self.profiler.measure("ppo.critic_optimization"):
                            result = self._process_critic(
                                batch, total_rows=group_rows
                            )
                        result.loss.backward()
                        total_loss += result.loss.detach()
                        self._record_batch_metrics(accumulator, result)
                        processed += 1
                    gradient = torch.nn.utils.clip_grad_norm_(
                        self.critic_parameter_list,
                        float(self.config["max_grad_norm"]),
                        error_if_nonfinite=True,
                    )
                    self.critic_optimizer.step()
                    critic_steps += 1
                    accumulator.add("critic_total_loss", total_loss)
                    accumulator.add("critic_gradient_norm", gradient)
                    accumulator.accumulate("critic_steps", 1)
                    accumulator.accumulate(
                        "critic_clipped_steps",
                        float(gradient > float(self.config["max_grad_norm"])),
                    )
            if not all(
                bool(torch.isfinite(parameter).all())
                for parameter in self.model.parameters()
            ):
                raise FloatingPointError("optimizer produced non-finite parameters")
        except Exception as exc:
            self.model.load_state_dict(model_before)
            self.actor_optimizer.load_state_dict(actor_optimizer_before)
            self.critic_optimizer.load_state_dict(critic_optimizer_before)
            self.kl_coefficient = kl_before
            return UpdateResult(
                False,
                self.policy_version,
                completed_epochs,
                processed,
                metrics(),
                str(exc),
                accumulator.statistic_values(),
            )

        self.ema_magnet.update(self.model, matches=ema_matches)
        target = float(self.config["target_kl"])
        factor = float(self.config["kl_adaptation_factor"])
        if post_kl > 1.5 * target:
            self.kl_coefficient *= factor
        elif post_kl < target / 1.5:
            self.kl_coefficient /= factor
        self.kl_coefficient = min(
            float(self.config["kl_coefficient_maximum"]),
            max(
                float(self.config["kl_coefficient_minimum"]),
                self.kl_coefficient,
            ),
        )
        self.policy_version += 1
        result_metrics = metrics()
        result_metrics["kl_coefficient"] = self.kl_coefficient
        result_metrics.update({
            f"magnet_{key}": value
            for key, value in self.ema_magnet.metrics(self.model).items()
        })
        result_metrics["magnet_kl_coefficient"] = float(
            self.config["magnet_kl_coefficient"]
        )
        result_metrics["entropy_floor"] = float(self.config["entropy_floor"])
        return UpdateResult(
            True,
            self.policy_version,
            completed_epochs,
            processed,
            result_metrics,
            None,
            accumulator.statistic_values(),
        )

    def begin_streaming_update(self, *, ema_matches, post_kl_probe_rows=65_536):
        """Begin one frozen-policy update accumulated over bounded chunks.

        Parameters are not mutated until :meth:`finish_streaming_update`.
        Chunk losses are accumulated as row sums and normalized once at the
        transaction boundary, so variable decision counts do not reweight
        chunks.  Advantage standardization remains local to each large rollout
        chunk; subtracting a chunk baseline is policy-gradient unbiased.
        """
        if self._streaming_update is not None:
            raise RuntimeError("a streaming PPO update is already active")
        if int(self.config["epochs"]) != 1 or int(self.config["minibatches"]) != 1:
            raise ValueError("streaming PPO requires one actor epoch and optimizer group")
        if int(self.config["critic_epochs"]) != 1:
            raise ValueError("streaming PPO requires one accumulated critic pass")
        self.ema_magnet.tau_for_matches(ema_matches)
        self.actor_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        template = next(self.model.parameters()).new_zeros(())
        self._streaming_update = {
            "model_before": {
                name: value.detach().clone()
                for name, value in self.model.state_dict().items()
            },
            "actor_optimizer_before": deepcopy(self.actor_optimizer.state_dict()),
            "critic_optimizer_before": deepcopy(self.critic_optimizer.state_dict()),
            "kl_before": self.kl_coefficient,
            "accumulator": MetricAccumulator(template),
            "actor_rows": 0,
            "critic_rows": 0,
            "entropy_rows": 0,
            "processed": 0,
            "chunks": 0,
            "pre_kl_sum": 0.0,
            "probe_rows": 0,
            "probe_limit": int(post_kl_probe_rows),
            "probe_batches": [],
            "actor_loss_sum": template.clone(),
            "critic_loss_sum": template.clone(),
            "ema_matches": int(ema_matches),
        }

    def _rollback_streaming_update(self):
        state = self._streaming_update
        if state is None:
            return
        self.model.load_state_dict(state["model_before"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer_before"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer_before"])
        self.kl_coefficient = state["kl_before"]
        self.actor_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        self._streaming_update = None

    def abort_streaming_update(self):
        """Rollback an active streaming transaction."""
        self._rollback_streaming_update()

    def accumulate_streaming_chunk(self, minibatches, *, critic_minibatches):
        """Accumulate one complete-match chunk without retaining prior chunks."""
        state = self._streaming_update
        if state is None:
            raise RuntimeError("no streaming PPO update is active")
        minibatches = tuple(minibatches)
        critic_minibatches = tuple(critic_minibatches)
        actor_rows = sum(map(learner_rows, minibatches))
        critic_rows = sum(map(learner_rows, critic_minibatches))
        if actor_rows <= 0 or critic_rows <= 0:
            raise ValueError("streaming PPO chunk has no trainable rows")
        self.model.train()
        try:
            with self.profiler.measure("ppo.pre_update_kl"):
                pre_kl = self._post_update_kl(minibatches)
            state["pre_kl_sum"] += pre_kl * actor_rows
            for batch in minibatches:
                with self.profiler.measure("ppo.actor_optimization"):
                    result = self._process_actor(
                        batch, total_rows=1, total_entropy_rows=1,
                    )
                result.loss.backward()
                state["actor_loss_sum"] += result.loss.detach()
                state["entropy_rows"] += int(
                    result.metrics["entropy_applicable_rows"]
                )
                self._record_batch_metrics(state["accumulator"], result)
                state["processed"] += 1
                if state["probe_rows"] < state["probe_limit"]:
                    state["probe_batches"].append(batch)
                    state["probe_rows"] += result.row_count
            for batch in critic_minibatches:
                with self.profiler.measure("ppo.critic_optimization"):
                    result = self._process_critic(batch, total_rows=1)
                result.loss.backward()
                state["critic_loss_sum"] += result.loss.detach()
                self._record_batch_metrics(state["accumulator"], result)
                state["processed"] += 1
        except Exception:
            self._rollback_streaming_update()
            raise
        state["actor_rows"] += actor_rows
        state["critic_rows"] += critic_rows
        state["chunks"] += 1
        return {
            "actor_rows": actor_rows,
            "critic_rows": critic_rows,
            "pre_update_approximate_kl": pre_kl,
        }

    @staticmethod
    def _divide_gradients(parameters, denominator):
        denominator = float(denominator)
        if denominator <= 0:
            raise ValueError("gradient denominator must be positive")
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad.div_(denominator)

    def finish_streaming_update(self):
        """Commit one optimizer step from all accumulated rollout chunks."""
        state = self._streaming_update
        if state is None:
            raise RuntimeError("no streaming PPO update is active")
        accumulator = state["accumulator"]

        def metrics():
            result = _finalize_statistic_metrics(
                accumulator.read(), accumulator.statistic_values()
            )
            result.update({
                "entropy_applicable_rows": float(state["entropy_rows"]),
                "actor_optimization_fraction": 1.0 if state["chunks"] else 0.0,
                "critic_optimization_fraction": 1.0 if state["chunks"] else 0.0,
                "optimization_fraction": 1.0 if state["chunks"] else 0.0,
            })
            return result

        try:
            pre_kl = state["pre_kl_sum"] / max(1, state["actor_rows"])
            accumulator.set("pre_update_approximate_kl", pre_kl)
            replay_limit = max(0.5 * float(self.config["target_kl"]), 1e-4)
            if pre_kl > replay_limit:
                raise RuntimeError(
                    "rollout policy differs before optimization: "
                    f"KL {pre_kl:.6g} exceeds {replay_limit:.6g}"
                )
            self._divide_gradients(
                self.actor_parameter_list, state["actor_rows"]
            )
            self._divide_gradients(
                self.critic_parameter_list, state["critic_rows"]
            )
            actor_gradient = torch.nn.utils.clip_grad_norm_(
                self.actor_parameter_list,
                float(self.config["max_grad_norm"]),
                error_if_nonfinite=True,
            )
            critic_gradient = torch.nn.utils.clip_grad_norm_(
                self.critic_parameter_list,
                float(self.config["max_grad_norm"]),
                error_if_nonfinite=True,
            )
            self.actor_optimizer.step()
            self.critic_optimizer.step()
            accumulator.add(
                "actor_total_loss",
                state["actor_loss_sum"] / state["actor_rows"],
            )
            accumulator.add(
                "critic_total_loss",
                state["critic_loss_sum"] / state["critic_rows"],
            )
            accumulator.add("actor_gradient_norm", actor_gradient)
            accumulator.add("critic_gradient_norm", critic_gradient)
            accumulator.accumulate("actor_steps", 1)
            accumulator.accumulate("critic_steps", 1)
            accumulator.accumulate(
                "actor_clipped_steps",
                float(actor_gradient > float(self.config["max_grad_norm"])),
            )
            accumulator.accumulate(
                "critic_clipped_steps",
                float(critic_gradient > float(self.config["max_grad_norm"])),
            )
            with self.profiler.measure("ppo.post_update_kl"):
                post_kl = self._post_update_kl(state["probe_batches"])
            accumulator.set("post_update_approximate_kl", post_kl)
            hard_limit = max(4.0 * float(self.config["target_kl"]), 1e-4)
            if post_kl > hard_limit:
                raise RuntimeError(
                    f"post-update KL {post_kl:.6g} exceeds {hard_limit:.6g}"
                )
            if not all(
                bool(torch.isfinite(parameter).all())
                for parameter in self.model.parameters()
            ):
                raise FloatingPointError("optimizer produced non-finite parameters")
        except Exception as exc:
            processed = int(state["processed"])
            result_metrics = metrics()
            statistics = accumulator.statistic_values()
            self._rollback_streaming_update()
            return UpdateResult(
                False, self.policy_version, 0, processed, result_metrics,
                str(exc), statistics,
            )

        ema_matches = int(state["ema_matches"])
        processed = int(state["processed"])
        self.ema_magnet.update(self.model, matches=ema_matches)
        target = float(self.config["target_kl"])
        factor = float(self.config["kl_adaptation_factor"])
        if post_kl > 1.5 * target:
            self.kl_coefficient *= factor
        elif post_kl < target / 1.5:
            self.kl_coefficient /= factor
        self.kl_coefficient = min(
            float(self.config["kl_coefficient_maximum"]),
            max(float(self.config["kl_coefficient_minimum"]), self.kl_coefficient),
        )
        self.policy_version += 1
        result_metrics = metrics()
        result_metrics["kl_coefficient"] = self.kl_coefficient
        result_metrics.update({
            f"magnet_{key}": value
            for key, value in self.ema_magnet.metrics(self.model).items()
        })
        result_metrics["magnet_kl_coefficient"] = float(
            self.config["magnet_kl_coefficient"]
        )
        result_metrics["entropy_floor"] = float(self.config["entropy_floor"])
        statistics = accumulator.statistic_values()
        self.actor_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        self._streaming_update = None
        return UpdateResult(
            True, self.policy_version, 1, processed, result_metrics, None,
            statistics,
        )

    def optimizer_state_dict(self):
        return {
            "architecture": self.architecture,
            "actor": self.actor_optimizer.state_dict(),
            "critic": self.critic_optimizer.state_dict(),
            "adaptive_kl": {
                "version": 1,
                "coefficient": self.kl_coefficient,
            },
            "ema_magnet": self.ema_magnet.state_dict(),
        }

    def load_optimizer_state_dict(self, state):
        if state.get("architecture") != self.architecture:
            raise ValueError("checkpoint uses a different PPO architecture")
        self.actor_optimizer.load_state_dict(state["actor"])
        self.critic_optimizer.load_state_dict(state["critic"])
        adaptive = dict(state.get("adaptive_kl") or {})
        if int(adaptive.get("version", -1)) != 1:
            raise ValueError("checkpoint is missing adaptive-KL state")
        coefficient = float(adaptive["coefficient"])
        if not float(self.config["kl_coefficient_minimum"]) <= coefficient \
                <= float(self.config["kl_coefficient_maximum"]):
            raise ValueError("checkpoint adaptive-KL coefficient is out of bounds")
        self.kl_coefficient = coefficient
        self.ema_magnet.load_state_dict(state.get("ema_magnet"), self.model)

    def checkpoint_state(
        self, *, counters=None, seeds=None, curriculum=None, population=None,
        rating=None, metrics=None, env=None, controls=None, provenance=None,
    ):
        counters = dict(counters or {})
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer_state_dict(),
            "trainer": {
                "architecture": self.architecture,
                "policy_version": self.policy_version,
                "counters": counters,
                "seeds": seeds.state_dict() if seeds is not None else None,
                "curriculum": curriculum,
                "population": population,
                "rating": rating,
                "metrics": metrics,
                "env": env,
                "controls": controls,
                "provenance": provenance,
            },
            "state": {
                "architecture": self.architecture,
                "policy_version": self.policy_version,
                "update": int(counters.get("update", self.policy_version)),
                "metric_cursor": metrics or {},
            },
        }

    def publish_checkpoint(self, root, *, metadata=None, **state):
        if self.policy_version <= 0:
            raise RuntimeError("cannot checkpoint before a committed update")
        from ..checkpoint import publish

        return publish(
            root, self.checkpoint_state(**state), metadata=metadata
        )

    @staticmethod
    def metric_points(update, result, *, source="update"):
        from ..metric_registry import REGISTRY
        from ..types import MetricPoint

        mapping = {
            "ppo/policy_loss": result.metrics.get("policy_loss", 0),
            "ppo/critic_loss": result.metrics.get("critic_loss", 0),
            "critic/boundary_order_cross_entropy": result.metrics.get(
                "boundary_order_cross_entropy", 0
            ),
            "critic/boundary_order_accuracy": result.metrics.get(
                "boundary_order_accuracy", 0
            ),
            "critic/boundary_rank_brier": result.metrics.get(
                "boundary_rank_brier", 0
            ),
            "ppo/entropy": result.metrics.get("entropy", 0),
            "ppo/entropy_efficiency": result.metrics.get(
                "entropy_efficiency", 0
            ),
            "ppo/entropy_applicable_rows": result.metrics.get(
                "entropy_applicable_rows", 0
            ),
            "ppo/total_loss": result.metrics.get("actor_total_loss", 0),
            "ppo/approximate_kl": result.metrics.get("approximate_kl", 0),
            "ppo/pre_update_approximate_kl": result.metrics.get(
                "pre_update_approximate_kl", 0
            ),
            "ppo/post_update_approximate_kl": result.metrics.get(
                "post_update_approximate_kl", 0
            ),
            "ppo/kl_loss": result.metrics.get("kl_loss", 0),
            "ppo/kl_coefficient": result.metrics.get("kl_coefficient", 0),
            "ppo/magnet_kl": result.metrics.get("magnet_kl", 0),
            "ppo/magnet_loss": result.metrics.get("magnet_loss", 0),
            "ppo/magnet_kl_coefficient": result.metrics.get(
                "magnet_kl_coefficient", 0
            ),
            "ppo/magnet_ema_tau": result.metrics.get("magnet_ema_tau", 0),
            "ppo/magnet_parameter_rms_distance": result.metrics.get(
                "magnet_parameter_rms_distance", 0
            ),
            "ppo/magnet_relative_parameter_rms_distance": result.metrics.get(
                "magnet_relative_parameter_rms_distance", 0
            ),
            "ppo/entropy_floor": result.metrics.get("entropy_floor", 0),
            "ppo/optimization_fraction": result.metrics.get(
                "optimization_fraction", 0
            ),
            "ppo/actor_optimization_fraction": result.metrics.get(
                "actor_optimization_fraction", 0
            ),
            "ppo/critic_optimization_fraction": result.metrics.get(
                "critic_optimization_fraction", 0
            ),
            "ppo/clip_fraction": result.metrics.get("clip_fraction", 0),
            "ppo/actor_gradient_norm": result.metrics.get(
                "actor_gradient_norm", 0
            ),
            "ppo/critic_gradient_norm": result.metrics.get(
                "critic_gradient_norm", 0
            ),
            "ppo/actor_gradient_clip_fraction": result.metrics.get(
                "actor_gradient_clip_fraction", 0
            ),
            "ppo/critic_gradient_clip_fraction": result.metrics.get(
                "critic_gradient_clip_fraction", 0
            ),
        }
        return [
            MetricPoint(
                name,
                definition.axis,
                update,
                value,
                definition.unit,
                definition.window,
                definition.reduction,
                source,
            )
            for name, value in mapping.items()
            for definition in (REGISTRY[name],)
        ]
