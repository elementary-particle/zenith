"""Batch selection and auxiliary objectives for PPO updates."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass(frozen=True)
class BatchResult:
    loss: object
    approximate_kl: object
    row_count: int
    metrics: dict[str, object]
    statistics: dict[str, object] = field(default_factory=dict)


class MetricAccumulator:
    """Keep metrics on-device until the update reaches a reporting boundary."""

    def __init__(self, template):
        self.template = template
        self.totals = {}
        self.denominators = {}
        self.statistics = {}

    def add(self, name, value, weight=1.0):
        value = metric_tensor(value, self.template)
        weight = float(weight)
        self.totals[name] = self.totals.get(name, 0.0) + value * weight
        self.denominators[name] = self.denominators.get(name, 0.0) + weight

    def set(self, name, value):
        self.totals[name] = metric_tensor(value, self.template)
        self.denominators[name] = 1.0

    def accumulate(self, name, value):
        value = metric_tensor(value, self.template)
        self.statistics[name] = self.statistics.get(name, 0.0) + value

    def statistic_values(self):
        return {
            name: float(value.detach().cpu())
            for name, value in self.statistics.items()
        }

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
    if eligible is not None:
        return int(eligible.sum())
    if "old_logp" in batch:
        value = batch["old_logp"]
        return int(value.numel() if hasattr(value, "numel") else value.size)
    if "rank_order_targets" in batch:
        value = batch["rank_order_targets"]
        return int(value.numel() if hasattr(value, "numel") else value.size)
    raise KeyError("batch has no learner-row cardinality field")
