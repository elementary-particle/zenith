"""Small immutable records shared across population, encoding, and metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, order=True, slots=True)
class ActionSpaceBinding:
    environment_id: int
    episode_generation: int
    frame_id: int
    seat: int


@dataclass(frozen=True, slots=True)
class CurriculumSnapshot:
    completed_matches: int
    progress: float
    policy_version: int


@dataclass(frozen=True, slots=True)
class MatchLineup:
    match_id: tuple[int, int]
    seat_policy_ids: tuple[str, str, str, str]
    learner_mask: int
    current_policy_version: int
    pool_snapshot_id: str
    draw_trace: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True, slots=True)
class RolloutMatchOutcome:
    match_id: tuple[int, int]
    checkpoint_ids: tuple[str, str, str, str]
    ranks: tuple[int, int, int, int]
    scores: tuple[int, int, int, int]
    completed_kyoku: int = 0


@dataclass(frozen=True, slots=True)
class MetricPoint:
    name: str
    axis: str
    step: int
    value: float
    unit: str
    window: str = "instant"
    reduction: str = "last"
    source: str = ""
