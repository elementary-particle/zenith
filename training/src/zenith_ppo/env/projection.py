"""Python-owned ordinary observation projection."""

from __future__ import annotations

from copy import deepcopy


def validate_event_payload(row: dict) -> None:
    kind = int(row.get("kind", 0))
    payload = bytes(row.get("payload") or b"")
    if kind == 2 and payload:
        if payload[0] != 2:
            raise ValueError(f"unsupported start_kyoku payload version {payload[0]}")
        if len(payload) < 24:
            raise ValueError("truncated start_kyoku payload")
        expected = 24 + sum(payload[20:24])
        if len(payload) != expected:
            raise ValueError(
                f"start_kyoku payload length {len(payload)} != {expected}"
            )
    elif kind == 16 and payload and len(payload) != 20:
        raise ValueError("end_game payload must contain four scores and four ranks")


def project_event(row: dict, *, observer: int) -> dict:
    result = deepcopy(row)
    visible = bool(int(row.get("visibility_mask", 0b1111)) & (1 << observer))
    if not visible:
        if "args" in result:
            result["args"] = (0, 0, 0, 0)
        for name in ("arg0", "arg1", "arg2", "arg3", "payload"):
            if name in result:
                result[name] = 0 if name != "payload" else b""
    return result

def project_decision(state, decision, *, observer: int, critic_mode: str = "privileged"):
    """Create Python-owned actor/critic mappings without changing native values."""
    if decision.seat != observer:
        raise ValueError("decision observer must be the acting seat")
    if critic_mode not in {"ordinary", "privileged"}:
        raise ValueError(f"unknown critic mode {critic_mode!r}")
    hidden_ids = None
    hidden_counts = None
    hidden_wall = None
    wall_indices = None
    if state.hidden is not None:
        hidden_ids = state.hidden.concealed_tile_ids
        hidden_counts = state.hidden.concealed_counts
        hidden_wall = state.hidden.wall
        wall_indices = state.hidden.wall_indices
    frame = {
        "environment_id": state.environment_id,
        "episode_generation": state.episode_generation,
        "frame_id": state.frame_id,
        "scores": state.scores,
        "round_wind": state.round_wind,
        "hand_number": state.hand_number,
        "dealer": state.dealer,
        "honba": state.honba,
        "riichi_deposits": state.riichi_deposits,
        "live_wall_remaining": state.live_wall_remaining,
        "dora_indicators": tuple(state.dora_indicators),
        "priv_concealed_tile_ids": hidden_ids,
        "priv_concealed_counts": hidden_counts,
        "priv_wall": hidden_wall if state.hidden is not None else None,
        "priv_wall_indices": wall_indices if state.hidden is not None else None,
    }
    actor = {
        "seat": decision.seat,
        "flags": decision.flags,
        "current_draw": decision.current_draw,
        "concealed_counts": tuple(decision.concealed_counts),
    }
    critic = {
        "mode": critic_mode,
        "priv_concealed_tile_ids": hidden_ids if critic_mode == "privileged" else None,
        "priv_concealed_counts": hidden_counts if critic_mode == "privileged" else None,
        "priv_wall": hidden_wall if critic_mode == "privileged" else None,
        "priv_wall_indices": wall_indices if critic_mode == "privileged" else None,
    }
    return frame, actor, critic
