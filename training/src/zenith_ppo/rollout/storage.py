"""Immutable-after-seal rollout storage."""

from __future__ import annotations

from dataclasses import dataclass, field
from ..types import CurriculumSnapshot, RolloutSample


@dataclass
class RolloutBatch:
    behavior_policy_version: int
    curriculum: CurriculumSnapshot
    samples: list[RolloutSample] = field(default_factory=list)
    failures: list[dict] = field(default_factory=list)
    sealed: bool = False

    def append(self, sample):
        if self.sealed: raise RuntimeError("rollout is sealed")
        if sample.behavior_policy_version != self.behavior_policy_version:
            raise ValueError("mixed behavior-policy versions")
        self.samples.append(sample)

    def seal(self):
        self.sealed = True
        return self

    @property
    def eligible_indices(self):
        return tuple(index for index, sample in enumerate(self.samples) if sample.ppo_eligible)

