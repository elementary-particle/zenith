"""Native immutable interface to the deterministic batched riichi env."""

from ._riichi import (
    ABSENT_SENTINEL,
    ACTION_KIND_CODES,
    EVENT_SCHEMA_VERSION,
    FRAME_STATUS_CODES,
    HAND_ANALYSIS_VERSION,
    MJAI_EVENT_NAMES,
    RNG_PROFILE,
    RNG_PROFILE_ID,
    RULES_PROFILE,
    RULES_PROFILE_ID,
    SHANTEN_UNAVAILABLE,
    SNAPSHOT_SCHEMA_VERSION,
    STATE_SCHEMA_VERSION,
    Action,
    ActionKind,
    Decision,
    Event,
    HandAnalysis,
    HiddenState,
    Meld,
    RiverTile,
    State,
    Transition,
    _Env,
    analyze_hands,
    ensure_shanten_cache,
)

Env = _Env

ACTION_KIND_CODES = dict(ACTION_KIND_CODES)
FRAME_STATUS_CODES = dict(FRAME_STATUS_CODES)
MJAI_EVENT_NAMES = dict(MJAI_EVENT_NAMES)

__all__ = [
    "Env",
    "State",
    "Event",
    "Decision",
    "Action",
    "ActionKind",
    "Transition",
    "Meld",
    "RiverTile",
    "HiddenState",
    "HandAnalysis",
    "analyze_hands",
    "ensure_shanten_cache",
    "STATE_SCHEMA_VERSION",
    "EVENT_SCHEMA_VERSION",
    "HAND_ANALYSIS_VERSION",
    "SNAPSHOT_SCHEMA_VERSION",
    "RULES_PROFILE",
    "RULES_PROFILE_ID",
    "RNG_PROFILE",
    "RNG_PROFILE_ID",
    "ABSENT_SENTINEL",
    "SHANTEN_UNAVAILABLE",
    "ACTION_KIND_CODES",
    "FRAME_STATUS_CODES",
    "MJAI_EVENT_NAMES",
]
