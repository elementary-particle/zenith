"""Small immutable records shared across rollout, PPO, population, and metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True, order=True, slots=True)
class DecisionBinding:
    environment_id: int
    episode_generation: int
    frame_id: int
    seat: int


@dataclass(frozen=True, slots=True)
class ObservationRecord:
    binding: DecisionBinding
    event_store_id: str
    event_end: int
    token_schema: int = 5
    actor_query_offset: int = -1
    token_count: int = 0


@dataclass(frozen=True, slots=True)
class ActionSegment:
    native_start: int
    native_end: int
    representatives: tuple[int, ...]
    factors: Any = None


@dataclass(frozen=True, slots=True)
class RewardRecord:
    discard_reward: float = 0.0
    kyoku_delta: float = 0.0
    rank_reward: float = 0.0
    weights: tuple[float, float, float] = (1.0, 0.0, 0.0)
    boundary_mode: Literal["kyoku", "match"] = "kyoku"

    @property
    def total(self) -> float:
        return sum(a * b for a, b in zip(self.weights,
            (self.discard_reward, self.kyoku_delta, self.rank_reward)))


@dataclass(slots=True)
class RolloutSample:
    binding: DecisionBinding
    observation: ObservationRecord
    actions: ActionSegment
    checkpoint_id: str
    behavior_policy_version: int
    ppo_eligible: bool
    selected_group: int
    selected_native: int
    old_log_probability: float
    entropy: float
    old_value: float
    reward: RewardRecord = field(default_factory=RewardRecord)
    terminal: bool = False
    kyoku_boundary: bool = False
    match_boundary: bool = False
    truncated: bool = False
    successor: int | None = None
    bootstrap_value: float = 0.0
    encoded: Any = None


@dataclass(frozen=True, slots=True)
class CurriculumSnapshot:
    update: int
    progress: float
    weights: tuple[float, float, float]
    boundary_mode: Literal["kyoku", "match"]
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
class MetricPoint:
    name: str
    axis: str
    step: int
    value: float
    unit: str
    window: str = "instant"
    reduction: str = "last"
    source: str = ""
