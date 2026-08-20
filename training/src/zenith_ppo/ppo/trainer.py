"""Transactional PPO updates for the policy and boundary-rank critic."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, fields, is_dataclass, replace
from math import sqrt

import torch
from torch.nn import functional as F

from .ema_magnet import EMAMagnet
from .loss import (
    actor_loss,
    legal_action_entropy,
    segmented_forward_kl,
)
from .objectives import BatchResult, MetricAccumulator, learner_rows


def state_value_mse_loss(predicted, targets, *, reduction="mean"):
    """Direct squared error for bounded expected rank utility."""
    predicted = predicted.float()
    targets = targets.to(predicted.dtype)
    if predicted.shape != targets.shape:
        raise ValueError("state-value prediction and target shapes differ")
    losses = F.mse_loss(predicted, targets, reduction="none")
    if reduction == "sum":
        loss = losses.sum()
    elif reduction == "mean":
        loss = losses.mean()
    else:
        raise ValueError("state-value reduction must be mean or sum")
    return loss


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
    state_rows = float(statistics.get("state_value_rows", 0.0))
    if state_rows:
        target_sum = float(statistics.get("state_value_target_sum", 0.0))
        target_square = float(statistics.get(
            "state_value_target_square_sum", 0.0,
        ))
        residual_sum = float(statistics.get(
            "state_value_residual_sum", 0.0,
        ))
        residual_square = float(statistics.get(
            "state_value_residual_square_sum", 0.0,
        ))
        target_variance = max(
            0.0, target_square / state_rows - (target_sum / state_rows) ** 2,
        )
        residual_variance = max(
            0.0,
            residual_square / state_rows - (residual_sum / state_rows) ** 2,
        )
        metrics["state_value_rmse"] = (
            float(statistics.get("state_value_squared_error_sum", 0.0))
            / state_rows
        ) ** 0.5
        metrics["state_value_explained_variance"] = (
            1.0 - residual_variance / target_variance
            if target_variance > 1e-12 else 0.0
        )
    for owner in ("actor", "critic"):
        steps = float(statistics.get(f"{owner}_steps", 0.0))
        metrics[f"{owner}_gradient_clip_fraction"] = (
            float(statistics.get(f"{owner}_clipped_steps", 0.0)) / steps
            if steps else 0.0
        )
    return metrics


class PPOTrainer:
    """Update all policy parameters from low-bias boundary-rank credit."""

    def __init__(
        self, model, config, *, device_type="cpu", use_bf16=False,
        profiler=None,
    ):
        self.model = model
        self.config = dict(config)
        from ..model.factory import checkpoint_architecture
        self.architecture = checkpoint_architecture(model)
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
        self.boundary_critic_parameter_list = list(
            model.boundary_critic_parameters()
        )
        self.decision_critic_parameter_list = list(
            model.decision_critic_parameters()
        )
        actor_ids = set(map(id, self.actor_parameter_list))
        critic_ids = set(map(id, self.critic_parameter_list))
        if actor_ids & critic_ids:
            raise ValueError("actor and critic parameters must be disjoint")
        if len(actor_ids | critic_ids) != sum(1 for _ in model.parameters()):
            raise ValueError("every parameter must belong to one optimizer")
        if set(map(id, self.boundary_critic_parameter_list)) \
                | set(map(id, self.decision_critic_parameter_list)) != critic_ids:
            raise ValueError("critic parameter partitions are incomplete")
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
        # Native RolloutChunk batches are bulk NumPy views over Rust-owned
        # columnar arenas. Convert the complete batch here without rebuilding
        # EncodedActionSpace objects or gathering Python rows.
        try:
            import numpy as np
        except ImportError:  # pragma: no cover - NumPy is a required runtime dep
            np = None
        if np is not None and isinstance(value, np.ndarray):
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message="The given NumPy array is not writable"
                )
                value = torch.as_tensor(value)
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

    def _process_actor(
        self, batch, *, total_rows,
        magnet_log_probabilities=None,
    ):
        batch = self._materialize_batch(batch)
        with torch.autocast(
            device_type=self.device_type,
            dtype=torch.bfloat16,
            enabled=self.use_bf16,
        ):
            output = self.model.forward_actor(
                **batch["model_inputs"],
                compute_entropy=False,
                compute_value=False,
                compute_auxiliary=False,
            )
            if magnet_log_probabilities is None:
                with torch.no_grad():
                    magnet_log_probabilities = self.ema_magnet.forward(
                        batch["model_inputs"]
                    ).log_probabilities
        eligible = batch.get("ppo_eligible")
        selected = self._select(batch["selected"], eligible)
        old_logp = self._select(batch["old_logp"], eligible).float()
        advantages = self._select(batch["advantages"], eligible).float()
        entropy, entropy_efficiency, entropy_applicable = legal_action_entropy(
            output.log_probabilities,
            batch["model_inputs"]["action_offsets"],
            batch["model_inputs"]["action_lengths"],
        )
        entropy = self._select(entropy, eligible)
        entropy_efficiency = self._select(entropy_efficiency, eligible)
        entropy_applicable = self._select(entropy_applicable, eligible)
        magnet_kl_rows = segmented_forward_kl(
            magnet_log_probabilities,
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
            entropy_efficiency,
            entropy_applicable,
            ratio_clip=float(self.config["ratio_clip"]),
            entropy_coefficient=float(self.config["entropy_coefficient"]),
            kl_coefficient=self.kl_coefficient,
            magnet_kl=magnet_kl,
            magnet_coefficient=float(self.config["magnet_kl_coefficient"]),
        )
        row_count = int(selected.numel())
        entropy_rows = int(losses.entropy_rows)
        scaled = (
            losses.policy + losses.entropy_loss
            + losses.kl_loss + losses.magnet_loss
        ) * row_count / total_rows
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

    def _process_streaming_actor(self, batch):
        """Build exact one-pass sufficient gradients for a frozen PPO actor.

        Global advantage centering cannot be applied until every logical-batch
        chunk has arrived.  While the rollout policy is frozen, the unclipped
        PPO policy gradient is affine in the advantage, so accumulating the
        raw-advantage and unit-advantage gradients is sufficient to apply the
        logical-batch mean and standard deviation at commit time.
        """
        batch = self._materialize_batch(batch)
        if "raw_advantages" not in batch:
            raise KeyError(
                "streaming actor batches require unnormalized raw_advantages"
            )
        with torch.autocast(
            device_type=self.device_type,
            dtype=torch.bfloat16,
            enabled=self.use_bf16,
        ):
            output = self.model.forward_actor(
                **batch["model_inputs"],
                compute_entropy=False,
                compute_value=False,
                compute_auxiliary=False,
            )
            with torch.no_grad():
                magnet_output = self.ema_magnet.forward(batch["model_inputs"])
        eligible = batch.get("ppo_eligible")
        selected = self._select(batch["selected"], eligible)
        old_logp = self._select(batch["old_logp"], eligible).float()
        raw_advantages = self._select(
            batch["raw_advantages"], eligible
        ).float()
        entropy, entropy_efficiency, entropy_applicable = legal_action_entropy(
            output.log_probabilities,
            batch["model_inputs"]["action_offsets"],
            batch["model_inputs"]["action_lengths"],
        )
        entropy = self._select(entropy, eligible)
        entropy_efficiency = self._select(entropy_efficiency, eligible)
        entropy_applicable = self._select(entropy_applicable, eligible)
        magnet_kl_rows = segmented_forward_kl(
            magnet_output.log_probabilities,
            output.log_probabilities,
            batch["model_inputs"]["action_offsets"],
        )
        magnet_kl_rows = self._select(magnet_kl_rows, eligible)
        magnet_kl = magnet_kl_rows.mean()
        selected_new = output.log_probabilities.index_select(
            0, selected
        ).float()
        log_ratio = selected_new - old_logp
        ratio = torch.exp(log_ratio)
        ratio_clip = float(self.config["ratio_clip"])
        if bool(((ratio - 1.0).abs() > ratio_clip).any()):
            raise RuntimeError(
                "streaming PPO requires an unclipped frozen rollout policy"
            )
        losses = actor_loss(
            selected_new,
            old_logp,
            raw_advantages,
            entropy,
            entropy_efficiency,
            entropy_applicable,
            ratio_clip=ratio_clip,
            entropy_coefficient=float(self.config["entropy_coefficient"]),
            kl_coefficient=self.kl_coefficient,
            magnet_kl=magnet_kl,
            magnet_coefficient=float(self.config["magnet_kl_coefficient"]),
        )
        row_count = int(selected.numel())
        entropy_rows = int(losses.entropy_rows)
        # These are sums. Each receives its own logical-batch denominator only
        # after all chunks have been accumulated.
        policy_advantage_sum = -(ratio * raw_advantages).sum()
        policy_baseline_sum = -ratio.sum()
        row_regular_sum = (
            losses.entropy_loss + losses.kl_loss + losses.magnet_loss
        ) * row_count
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
        result = BatchResult(
            row_regular_sum,
            losses.approximate_kl.detach() * row_count,
            row_count,
            metrics,
            statistics,
        )
        terms = {
            "raw_advantages": raw_advantages.detach(),
            "policy_advantage_sum": policy_advantage_sum,
            "policy_baseline_sum": policy_baseline_sum,
            "row_regular_sum": row_regular_sum,
        }
        return result, terms

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
            {},
            {
                "boundary_rows": row_count,
                "boundary_order_cross_entropy_sum": nll.detach() * row_count,
                "boundary_order_accuracy_sum": accuracy.detach() * row_count,
                "boundary_rank_brier_sum": brier.detach() * row_count,
            },
        )

    def _process_state_critic(self, batch, *, total_rows):
        """Fit the detached per-decision baseline to boundary return targets."""
        batch = self._materialize_batch(batch)
        with torch.autocast(
            device_type=self.device_type,
            dtype=torch.bfloat16,
            enabled=self.use_bf16,
        ):
            predicted = self.model.forward_decision_critic(
                **batch["model_inputs"],
            )
        return self._state_critic_result(
            predicted, batch, total_rows=total_rows,
        )

    def _process_cached_state_critic(self, batch, *, total_rows):
        """Fit the value head from one post-actor feature extraction pass."""
        batch = self._to_device(batch)
        with torch.autocast(
            device_type=self.device_type,
            dtype=torch.bfloat16,
            enabled=self.use_bf16,
        ):
            boundary = self.model.forward_critic(
                decision_seats=batch["decision_seats"],
                rank_boundary_features=batch["rank_boundary_features"],
            ).rank_values
            predicted = self.model.forward_decision_head(
                batch["decision_features"], boundary,
            )
        return self._state_critic_result(
            predicted, batch, total_rows=total_rows,
        )

    def _state_critic_result(self, predicted, batch, *, total_rows):
        eligible = batch.get("ppo_eligible")
        predicted = self._select(predicted, eligible).float()
        targets = self._select(
            batch["value_targets"], eligible,
        ).float()
        row_count = int(targets.numel())
        if not row_count:
            raise ValueError("state critic batch has no learner rows")
        value_loss = state_value_mse_loss(predicted, targets)
        weighted = float(self.config["value_coefficient"]) * value_loss
        return BatchResult(
            weighted * row_count / total_rows,
            weighted.new_zeros(()),
            row_count,
            {
                "state_value_loss": value_loss,
                "state_value_prediction_mean": predicted.mean(),
                "state_value_target_mean": targets.mean(),
            },
            {
                "state_value_rows": row_count,
                "state_value_squared_error_sum": (
                    predicted.detach() - targets
                ).square().sum(),
                "state_value_target_sum": targets.detach().sum(),
                "state_value_target_square_sum": targets.detach().square().sum(),
                "state_value_residual_sum": (
                    targets.detach() - predicted.detach()
                ).sum(),
                "state_value_residual_square_sum": (
                    targets.detach() - predicted.detach()
                ).square().sum(),
            },
        )

    def _cache_magnet_log_probabilities(self, minibatches):
        """Evaluate the update-fixed EMA actor once for multi-epoch replay."""
        result = {}
        with torch.no_grad():
            for batch in minibatches:
                materialized = self._materialize_batch(batch)
                with torch.autocast(
                    device_type=self.device_type,
                    dtype=torch.bfloat16,
                    enabled=self.use_bf16,
                ):
                    output = self.ema_magnet.forward(
                        materialized["model_inputs"]
                    )
                result[id(batch)] = output.log_probabilities.detach()
        return result

    def _post_update_kl_and_state_cache(self, minibatches):
        """Compute exact KL while extracting post-actor critic features once."""
        total = next(self.model.parameters()).new_zeros((), dtype=torch.float64)
        count = 0
        cached = []
        cache_device = self.device
        if self.device.type == "cuda":
            row_count = 0
            for batch in minibatches:
                rows = batch["old_logp"]
                row_count += int(
                    rows.numel() if hasattr(rows, "numel") else rows.size
                )
            feature_width = int(self.model.decision_value[1].in_features)
            projected_bytes = row_count * (
                feature_width * 4 + 28 * 4 + 8 + 4
            )
            free_bytes, _ = torch.cuda.mem_get_info(self.device)
            if projected_bytes > free_bytes // 2:
                cache_device = torch.device("cpu")
            self.profiler.observe(
                "ppo.state_cache_projected_bytes", projected_bytes,
            )
            self.profiler.observe(
                "ppo.state_cache_on_device",
                float(cache_device.type == "cuda"),
            )
        with torch.no_grad():
            for batch in minibatches:
                materialized = self._materialize_batch(batch)
                inputs = materialized["model_inputs"]
                with torch.autocast(
                    device_type=self.device_type,
                    dtype=torch.bfloat16,
                    enabled=self.use_bf16,
                ):
                    output = self.model.forward_actor(
                        **inputs,
                        compute_entropy=False,
                        compute_value=False,
                        compute_auxiliary=False,
                    )
                eligible = materialized.get("ppo_eligible")
                selected = self._select(materialized["selected"], eligible)
                old_logp = self._select(
                    materialized["old_logp"], eligible,
                ).float()
                new_logp = output.log_probabilities.index_select(
                    0, selected,
                ).float()
                difference = new_logp - old_logp
                total += (
                    torch.expm1(difference) - difference
                ).double().sum()
                count += int(difference.numel())
                cached_batch = {
                    "decision_features": self.model.decision_features(
                        output, inputs["action_lengths"],
                    ).detach().to(cache_device),
                    "decision_seats": inputs["decision_seats"].detach().to(
                        cache_device
                    ),
                    "rank_boundary_features": inputs[
                        "rank_boundary_features"
                    ].detach().to(cache_device),
                    "value_targets": materialized["value_targets"].detach().to(
                        cache_device
                    ),
                }
                if eligible is not None:
                    cached_batch["ppo_eligible"] = eligible.detach().to(
                        cache_device
                    )
                cached.append(cached_batch)
        return float(total.cpu()) / max(1, count), tuple(cached)

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
                    output = self.model.forward_actor(
                        **batch["model_inputs"],
                        compute_entropy=False,
                        compute_value=False,
                        compute_auxiliary=False,
                    )
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

    def update_logical_batch(
        self, minibatches, *, critic_minibatches=None, rng=None,
        actor_rng=None, critic_rng=None, ema_matches=1,
        rollout_policy_verified=False,
    ):
        """Normalize a retained logical batch, then run the ordinary update.

        Unlike the bounded-memory streaming path, this path sees every raw
        advantage before actor autograd begins. It therefore needs one normal
        actor backward per packed tensor batch instead of separate sufficient
        VJPs for the advantage, centering baseline, entropy, and regularizers.
        """
        minibatches = tuple(minibatches)
        if not minibatches:
            return self.update(
                minibatches,
                critic_minibatches=critic_minibatches,
                rng=rng,
                actor_rng=actor_rng,
                critic_rng=critic_rng,
                ema_matches=ema_matches,
                rollout_policy_verified=rollout_policy_verified,
            )
        count = 0
        total = 0.0
        square_total = 0.0
        raw_batches = []
        for batch in minibatches:
            if "raw_advantages" not in batch:
                raise KeyError(
                    "logical actor batches require unnormalized raw_advantages"
                )
            raw = torch.as_tensor(batch["raw_advantages"]).detach().double()
            eligible = batch.get("ppo_eligible")
            selected = raw if eligible is None else raw[
                torch.as_tensor(eligible, device=raw.device).bool()
            ]
            count += int(selected.numel())
            total += float(selected.sum())
            square_total += float(selected.square().sum())
            raw_batches.append((batch, raw, eligible))
        if count == 0:
            return self.update(
                minibatches,
                critic_minibatches=critic_minibatches,
                rng=rng,
                actor_rng=actor_rng,
                critic_rng=critic_rng,
                ema_matches=ema_matches,
                rollout_policy_verified=rollout_policy_verified,
            )
        mean = total / count
        deviation = sqrt(max(0.0, square_total / count - mean * mean))
        normalized_batches = []
        for batch, raw, eligible in raw_batches:
            normalized = torch.zeros_like(raw, dtype=torch.float32)
            mask = None if eligible is None else torch.as_tensor(
                eligible, device=raw.device,
            ).bool()
            if deviation >= 1e-8:
                if mask is None:
                    normalized = ((raw - mean) / (deviation + 1e-8)).float()
                else:
                    normalized[mask] = (
                        (raw[mask] - mean) / (deviation + 1e-8)
                    ).float()
            materialized = dict(batch)
            materialized["advantages"] = normalized
            normalized_batches.append(materialized)
        result = self.update(
            normalized_batches,
            critic_minibatches=critic_minibatches,
            rng=rng,
            actor_rng=actor_rng,
            critic_rng=critic_rng,
            ema_matches=ema_matches,
            rollout_policy_verified=rollout_policy_verified,
        )
        return replace(result, metrics=result.metrics | {
            "logical_advantage_mean": mean,
            "logical_advantage_std": deviation,
        })

    def update(
        self, minibatches, *, critic_minibatches=None, rng=None,
        actor_rng=None, critic_rng=None, ema_matches=1,
        rollout_policy_verified=False,
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
        actor_enabled = any(
            float(group["lr"]) > 0 for group in self.actor_optimizer.param_groups
        )

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
            })
            return result

        try:
            if rollout_policy_verified:
                # The native orchestrator hashes every rollout policy before
                # collection and again immediately before this transaction.
                # Replaying the full retained actor here duplicated a complete
                # forward pass without adding a stronger identity check.
                pre_update_kl = 0.0
            else:
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
            magnet_log_probability_cache = {}
            if actor_enabled and int(self.config["epochs"]) > 1:
                with self.profiler.measure("ppo.magnet_cache"):
                    magnet_log_probability_cache = (
                        self._cache_magnet_log_probabilities(minibatches)
                    )
            for _ in range(int(self.config["epochs"]) if actor_enabled else 0):
                for group in shuffled_groups(minibatches, actor_groups, actor_rng):
                    group_rows = sum(map(learner_rows, group))
                    self.actor_optimizer.zero_grad(set_to_none=True)
                    total_loss = template.clone()
                    for batch in group:
                        with self.profiler.measure("ppo.actor_optimization"):
                            result = self._process_actor(
                                batch,
                                total_rows=group_rows,
                                magnet_log_probabilities=(
                                    magnet_log_probability_cache.get(id(batch))
                                ),
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
            del magnet_log_probability_cache
            with self.profiler.measure("ppo.post_update_kl"):
                post_kl, state_critic_batches = (
                    self._post_update_kl_and_state_cache(minibatches)
                )
            accumulator.set("post_update_approximate_kl", post_kl)
            hard_limit = max(
                4.0 * float(self.config["target_kl"]), 1e-4
            )
            if post_kl > hard_limit:
                raise RuntimeError(
                    f"post-update KL {post_kl:.6g} exceeds {hard_limit:.6g}"
                )

            # Retained logical batches replay the dense public-state regression
            # so it tracks faster than the actor at low GAE lambda. The sparse
            # structured boundary critic remains a one-pass objective.
            for critic_epoch in range(int(self.config["critic_epochs"])):
                self.critic_optimizer.zero_grad(set_to_none=True)
                total_loss = template.clone()
                for batch in state_critic_batches:
                    with self.profiler.measure("ppo.state_critic_optimization"):
                        result = self._process_cached_state_critic(
                            batch, total_rows=total_rows,
                        )
                    result.loss.backward()
                    total_loss += result.loss.detach()
                    self._record_batch_metrics(accumulator, result)
                    processed += 1
                # Sparse final-order labels are not a tracking bottleneck and
                # are seen once. Only the dense decision critic is replayed.
                if critic_epoch == 0:
                    for batch in critic_minibatches:
                        with self.profiler.measure(
                            "ppo.boundary_critic_optimization"
                        ):
                            result = self._process_critic(
                                batch, total_rows=total_critic_rows,
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
        if actor_enabled:
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
        result_metrics["entropy_coefficient"] = float(
            self.config["entropy_coefficient"]
        )
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
        Losses and advantage sufficient gradients are accumulated as row sums
        and normalized once at the transaction boundary. Thus chunk boundaries
        cannot change the update produced by a fixed set of rollout rows.
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
            "processed": 0,
            "chunks": 0,
            "pre_kl_sum": 0.0,
            "probe_rows": 0,
            "probe_limit": int(post_kl_probe_rows),
            "probe_batches": [],
            "policy_advantage_loss_sum": template.clone(),
            "policy_baseline_loss_sum": template.clone(),
            "row_regular_loss_sum": template.clone(),
            "advantage_sum": 0.0,
            "advantage_square_sum": 0.0,
            "policy_advantage_gradients": [
                torch.zeros_like(parameter)
                for parameter in self.actor_parameter_list
            ],
            "policy_baseline_gradients": [
                torch.zeros_like(parameter)
                for parameter in self.actor_parameter_list
            ],
            "state_value_loss_sum": template.clone(),
            "boundary_loss_sum": template.clone(),
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

    @staticmethod
    def _accumulate_gradient_sum(targets, loss, parameters, *, retain_graph):
        gradients = torch.autograd.grad(
            loss,
            parameters,
            retain_graph=retain_graph,
            allow_unused=True,
        )
        with torch.no_grad():
            for target, gradient in zip(targets, gradients, strict=True):
                if gradient is not None:
                    target.add_(gradient)

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
                    result, terms = self._process_streaming_actor(batch)
                self._accumulate_gradient_sum(
                    state["policy_advantage_gradients"],
                    terms["policy_advantage_sum"],
                    self.actor_parameter_list,
                    retain_graph=True,
                )
                self._accumulate_gradient_sum(
                    state["policy_baseline_gradients"],
                    terms["policy_baseline_sum"],
                    self.actor_parameter_list,
                    retain_graph=True,
                )
                terms["row_regular_sum"].backward()
                raw_advantages = terms["raw_advantages"].double()
                state["advantage_sum"] += float(raw_advantages.sum().cpu())
                state["advantage_square_sum"] += float(
                    raw_advantages.square().sum().cpu()
                )
                state["policy_advantage_loss_sum"] += (
                    terms["policy_advantage_sum"].detach()
                )
                state["policy_baseline_loss_sum"] += (
                    terms["policy_baseline_sum"].detach()
                )
                state["row_regular_loss_sum"] += (
                    terms["row_regular_sum"].detach()
                )
                self._record_batch_metrics(state["accumulator"], result)
                state["processed"] += 1
                with self.profiler.measure("ppo.state_critic_optimization"):
                    value_result = self._process_state_critic(
                        batch, total_rows=1,
                    )
                value_result.loss.backward()
                state["state_value_loss_sum"] += value_result.loss.detach()
                self._record_batch_metrics(
                    state["accumulator"], value_result,
                )
                state["processed"] += 1
                if state["probe_rows"] < state["probe_limit"]:
                    state["probe_batches"].append(batch)
                    state["probe_rows"] += result.row_count
            for batch in critic_minibatches:
                with self.profiler.measure(
                    "ppo.boundary_critic_optimization"
                ):
                    result = self._process_critic(batch, total_rows=1)
                result.loss.backward()
                state["boundary_loss_sum"] += result.loss.detach()
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

    def _finalize_streaming_actor_gradients(self, state):
        rows = int(state["actor_rows"])
        mean = state["advantage_sum"] / rows
        variance = max(
            0.0,
            state["advantage_square_sum"] / rows - mean * mean,
        )
        deviation = sqrt(variance)
        policy_scale = 0.0 if deviation < 1e-8 else 1.0 / (
            rows * (deviation + 1e-8)
        )
        regular_scale = 1.0 / rows
        with torch.no_grad():
            for parameter, advantage, baseline in zip(
                self.actor_parameter_list,
                state["policy_advantage_gradients"],
                state["policy_baseline_gradients"],
                strict=True,
            ):
                if parameter.grad is None:
                    parameter.grad = torch.zeros_like(parameter)
                parameter.grad.mul_(regular_scale)
                if policy_scale:
                    parameter.grad.add_(advantage, alpha=policy_scale)
                    parameter.grad.add_(baseline, alpha=-mean * policy_scale)
        normalized_policy_loss = state["policy_advantage_loss_sum"].new_zeros(())
        if policy_scale:
            normalized_policy_loss = (
                state["policy_advantage_loss_sum"]
                - mean * state["policy_baseline_loss_sum"]
            ) * policy_scale
        actor_total_loss = (
            normalized_policy_loss
            + state["row_regular_loss_sum"] * regular_scale
        )
        return normalized_policy_loss, actor_total_loss, mean, deviation

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
                "entropy_applicable_rows": float(
                    accumulator.statistics.get("entropy_applicable_rows", 0)
                ),
                "actor_optimization_fraction": 1.0 if state["chunks"] else 0.0,
                "critic_optimization_fraction": 1.0 if state["chunks"] else 0.0,
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
            (
                normalized_policy_loss,
                actor_total_loss,
                advantage_mean,
                advantage_deviation,
            ) = self._finalize_streaming_actor_gradients(state)
            accumulator.set("policy_loss", normalized_policy_loss)
            accumulator.set("logical_advantage_mean", advantage_mean)
            accumulator.set("logical_advantage_std", advantage_deviation)
            self._divide_gradients(
                self.decision_critic_parameter_list, state["actor_rows"]
            )
            self._divide_gradients(
                self.boundary_critic_parameter_list, state["critic_rows"]
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
                actor_total_loss,
            )
            accumulator.add(
                "critic_total_loss",
                state["state_value_loss_sum"] / state["actor_rows"]
                + state["boundary_loss_sum"] / state["critic_rows"],
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
        result_metrics["entropy_coefficient"] = float(
            self.config["entropy_coefficient"]
        )
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
            "critic/boundary_order_cross_entropy": result.metrics.get(
                "boundary_order_cross_entropy", 0
            ),
            "critic/boundary_order_accuracy": result.metrics.get(
                "boundary_order_accuracy", 0
            ),
            "critic/boundary_rank_brier": result.metrics.get(
                "boundary_rank_brier", 0
            ),
            "critic/state_value_loss": result.metrics.get(
                "state_value_loss", 0
            ),
            "critic/state_value_prediction_mean": result.metrics.get(
                "state_value_prediction_mean", 0
            ),
            "critic/state_value_target_mean": result.metrics.get(
                "state_value_target_mean", 0
            ),
            "critic/state_value_rmse": result.metrics.get(
                "state_value_rmse", 0
            ),
            "critic/state_value_explained_variance": result.metrics.get(
                "state_value_explained_variance", 0
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
            "ppo/entropy_coefficient": result.metrics.get(
                "entropy_coefficient", 0
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
