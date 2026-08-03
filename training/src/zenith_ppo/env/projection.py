"""Python-owned ordinary observation projection."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace


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


def project_action_space(state, space, *, observer: int):
    """Create a Python-owned public observation mapping."""
    if space.seat != observer:
        raise ValueError("action-space observer must be the acting seat")
    frame = {
        "environment_id": state.environment_id,
        "episode_generation": state.episode_generation,
        "frame_id": state.frame_id,
        "phase": state.phase,
        "eligible_mask": state.eligible_mask,
        "scores": state.scores,
        "round_wind": state.round_wind,
        "hand_number": state.hand_number,
        "dealer": state.dealer,
        "honba": state.honba,
        "riichi_deposits": state.riichi_deposits,
        "live_wall_remaining": state.live_wall_remaining,
        "dora_indicators": tuple(state.dora_indicators),
        "seat_flags": tuple(state.seat_flags),
        "rivers": tuple({
            "seat": int(row.seat), "tile": int(row.tile),
            "sequence": int(row.sequence),
            "riichi_declaration": bool(row.riichi_declaration),
            "called": bool(row.called), "tsumogiri": bool(row.tsumogiri),
        } for row in state.rivers),
        "melds": tuple({
            "seat": int(meld.seat), "kind": int(meld.kind),
            "from_seat": None if meld.from_seat is None else int(meld.from_seat),
            "called_tile": None if meld.called_tile is None else int(meld.called_tile),
            "tiles": tuple(int(tile) for tile in meld.tiles),
            "created_sequence": int(meld.created_sequence),
        } for meld in state.melds),
    }
    actor = {
        "seat": space.seat,
        "flags": space.flags,
        "current_draw": space.current_draw,
        "concealed_counts": tuple(space.concealed_counts),
    }
    return frame, actor


def project_observer_frame(state, *, observer: int):
    """Project any materialized core frame from one legal public viewpoint.

    Native privileged state is used only to recover the observer's own hand
    when that seat has no decision object. Opponent hands and wall state remain
    absent in ordinary mode.
    """
    space = next(
        (row for row in state.action_spaces if int(row.seat) == int(observer)),
        None,
    )
    if space is None:
        if state.hidden is None:
            raise ValueError("all-seat frame projection requires privileged materialization")
        space = SimpleNamespace(
            seat=int(observer),
            flags=int(state.seat_flags[observer]),
            current_draw=(
                state.hidden.current_draw
                if int(state.hidden.current_seat) == int(observer)
                else None
            ),
            concealed_counts=tuple(state.hidden.concealed_counts[observer]),
            candidates=(),
            environment_id=int(state.environment_id),
            episode_generation=int(state.episode_generation),
            frame_id=int(state.frame_id),
        )
    frame, actor = project_action_space(state, space, observer=int(observer))
    return frame, actor, space
