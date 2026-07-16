"""Small immutable records shared across rollout, PPO, population, and metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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
    kyoku_delta: float = 0.0
    rank_reward: float = 0.0
    weights: tuple[float, float] = (1.0, 0.0)

    @property
    def total(self) -> float:
        return self.weights[0] * self.kyoku_delta + self.weights[1] * self.rank_reward


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
    old_score_value: float
    old_rank_value: float
    reward: RewardRecord = field(default_factory=RewardRecord)
    kyoku_boundary: bool = False
    terminal: bool = False
    match_boundary: bool = False
    truncated: bool = False
    successor: int | None = None
    bootstrap_score_value: float = float("nan")
    bootstrap_rank_value: float = float("nan")
    terminal_placement: int = -1
    encoded: Any = None


@dataclass(frozen=True, slots=True)
class CurriculumSnapshot:
    completed_matches: int
    progress: float
    weights: tuple[float, float]
    policy_version: int
    guidance_phase: str = "full"
    guidance_scale: float = 1.0
    competence_streak: int = 0
    regression_streak: int = 0
    last_valid_worse_shanten_rate: float | None = None
    taper_matches: int = 0
    taper_progress: float = 0.0
    bot_fraction: float = 0.05


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
