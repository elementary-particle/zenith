"""Small immutable records shared across rollout, PPO, population, and metrics."""

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
class ObservationRecord:
    binding: ActionSpaceBinding
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


@dataclass(slots=True)
class RolloutSample:
    binding: ActionSpaceBinding
    observation: ObservationRecord
    actions: ActionSegment
    checkpoint_id: str
    behavior_policy_version: int
    ppo_eligible: bool
    selected_group: int
    selected_native: int
    old_log_probability: float
    entropy: float
    conditional_entropy_efficiency: float = 0.0
    conditional_entropy_applicable: bool = False
    old_boundary_rank_value: float = float("nan")
    kyoku_boundary: bool = False
    terminal: bool = False
    match_boundary: bool = False
    truncated: bool = False
    successor: int | None = None
    terminal_placement: int = -1
    rank_boundary_supervision: bool = False
    rank_order_target: int = -1
    frame_index: int = -1
    encoded: Any = None


@dataclass(slots=True)
class RolloutFrame:
    """One observer view of every materialized core transition."""

    binding: ActionSpaceBinding
    checkpoint_id: str
    ppo_eligible: bool
    phase: int
    genuine_action: bool
    old_boundary_rank_value: float = float("nan")
    old_boundary_rank_probabilities: tuple[float, float, float, float] = (
        0.25, 0.25, 0.25, 0.25,
    )
    kyoku_boundary: bool = False
    terminal: bool = False
    match_boundary: bool = False
    truncated: bool = False
    successor: int | None = None
    terminal_placement: int = -1
    rank_boundary_supervision: bool = False
    rank_order_target: int = -1
    encoded: Any = None


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
