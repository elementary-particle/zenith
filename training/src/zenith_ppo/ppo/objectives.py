"""Batch selection and auxiliary objectives for PPO updates."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


TEACHER_KINDS = ("discard", "reaction", "riichi")
TEACHER_ROW_METRICS = tuple(f"{kind}_teacher_rows" for kind in TEACHER_KINDS)
TEACHER_LOSS_WEIGHTS = {
    "discard_teacher_loss": "discard_teacher_rows",
    "discard_teacher_agreement": "discard_teacher_rows",
    "reaction_teacher_loss": "reaction_teacher_rows",
    "reaction_entropy": "reaction_teacher_rows",
    "riichi_teacher_loss": "riichi_teacher_rows",
}
TEACHER_COEFFICIENTS = {
    "discard": 0.0,
    "reaction": 0.0,
    "riichi": 0.0,
    "reaction_entropy": 0.0,
}


@dataclass(frozen=True)
class LearnerView:
    selected: object
    old_logp: object
    advantages: object
    score_values: object
    rank_values: object
    old_score_values: object
    old_rank_values: object
    score_returns: object
    rank_returns: object
    entropy: object
    eligible: object | None


@dataclass(frozen=True)
class BeliefResult:
    total: object
    count_loss: object
    tenpai_loss: object
    count_accuracy: object
    tenpai_accuracy: object


@dataclass(frozen=True)
class BatchResult:
    loss: object
    approximate_kl: object
    auxiliary_loss: object
    row_count: int
    metrics: dict[str, object]
    teacher_weights: dict[str, object]


class MetricAccumulator:
    """Keep metrics on-device until the update reaches a reporting boundary."""

    def __init__(self, template):
        self.template = template
        self.totals = {}
        self.denominators = {}

    def add(self, name, value, weight=1.0):
        value = metric_tensor(value, self.template)
        weight = float(weight)
        self.totals[name] = self.totals.get(name, 0.0) + value * weight
        self.denominators[name] = self.denominators.get(name, 0.0) + weight

    def set(self, name, value):
        self.totals[name] = metric_tensor(value, self.template)
        self.denominators[name] = 1.0

    def read(self):
        if not self.totals:
            return {}
        names = tuple(self.totals)
        values = torch.stack([
            self.totals[name] / max(self.denominators[name], 1.0)
            for name in names
        ])
        return dict(zip(names, values.tolist(), strict=True))


def metric_tensor(value, reference):
    if torch.is_tensor(value):
        return value.detach().to(device=reference.device)
    return reference.new_tensor(float(value))


def learner_rows(batch) -> int:
    eligible = batch.get("ppo_eligible")
    return int(batch["old_logp"].numel()) if eligible is None else int(eligible.sum())


def teacher_row_totals(minibatches):
    totals = {f"{kind}_rows": 0 for kind in TEACHER_KINDS}
    for batch in minibatches:
        targets = batch.get("teacher_targets")
        if targets is not None and hasattr(targets, "counts"):
            for name in totals:
                totals[name] += int(targets.counts[name])
    return totals


def select_learner_rows(output, batch) -> LearnerView:
    eligible = batch.get("ppo_eligible")
    if eligible is not None and not eligible.any():
        raise ValueError("minibatch has no current-policy rows")

    def select(value):
        return value if eligible is None else value[eligible]

    return LearnerView(
        selected=select(batch["selected"]),
        old_logp=select(batch["old_logp"]),
        advantages=select(batch["advantages"]),
        score_values=select(output.score_values),
        rank_values=select(output.rank_values),
        old_score_values=select(batch["old_score_values"]),
        old_rank_values=select(batch["old_rank_values"]),
        score_returns=select(batch["score_returns"]),
        rank_returns=select(batch["rank_returns"]),
        entropy=select(output.entropy),
        eligible=eligible,
    )


def belief_objective(output, batch, eligible, config, zero) -> BeliefResult:
    available = (
        hasattr(output, "opponent_count_logits")
        and "opponent_count_targets" in batch
    )
    if not available:
        return BeliefResult(zero, zero, zero, zero, zero)

    def select(value):
        return value if eligible is None else value[eligible]

    count_logits = select(output.opponent_count_logits)
    tenpai_logits = select(output.opponent_tenpai_logits)
    count_targets = select(batch["opponent_count_targets"])
    tenpai_targets = select(batch["opponent_tenpai_targets"])
    count_loss = F.cross_entropy(
        count_logits.reshape(-1, 5), count_targets.long().reshape(-1)
    )
    tenpai_loss = F.binary_cross_entropy_with_logits(
        tenpai_logits, tenpai_targets.float()
    )
    total = float(config.get("belief_coefficient", 0.10)) * (
        count_loss
        + float(config.get("belief_tenpai_coefficient", 0.25)) * tenpai_loss
    )
    count_accuracy = (count_logits.argmax(-1) == count_targets).float().mean()
    tenpai_accuracy = (
        (tenpai_logits >= 0) == (tenpai_targets >= 0.5)
    ).float().mean()
    return BeliefResult(
        total, count_loss, tenpai_loss, count_accuracy, tenpai_accuracy
    )


def teacher_objective(output, batch, eligible, totals, zero):
    coefficients = batch.get("teacher_coefficients", TEACHER_COEFFICIENTS)
    if "teacher_targets" not in batch or not any(map(float, coefficients.values())):
        metrics = {
            "discard_teacher_loss": 0.0,
            "discard_teacher_agreement": 0.0,
            "discard_teacher_rows": 0.0,
            "reaction_teacher_loss": 0.0,
            "reaction_teacher_rows": 0.0,
            "reaction_entropy": 0.0,
            "riichi_teacher_loss": 0.0,
            "riichi_teacher_rows": 0.0,
        }
        return zero, metrics, coefficients

    from ..teachers import auxiliary_losses

    auxiliary = auxiliary_losses(
        output.logits,
        output.log_probabilities,
        batch["model_inputs"]["action_offsets"],
        batch["teacher_targets"],
        coefficients,
        eligible=eligible,
    )
    metrics = auxiliary.metrics
    scaled = zero
    for kind, loss in (
        ("discard", auxiliary.discard_loss),
        ("riichi", auxiliary.riichi_loss),
    ):
        rows = int(metrics[f"{kind}_teacher_rows"])
        if totals[f"{kind}_rows"]:
            scaled = scaled + (
                float(coefficients[kind]) * loss * rows / totals[f"{kind}_rows"]
            )
    reaction_rows = int(metrics["reaction_teacher_rows"])
    if totals["reaction_rows"]:
        fraction = reaction_rows / totals["reaction_rows"]
        scaled = scaled + (
            float(coefficients["reaction"]) * auxiliary.reaction_loss
            + auxiliary.reaction_entropy_loss
        ) * fraction
    return scaled, metrics, coefficients
