"""Ordinary actor state and canonical private-critic encoding."""

from __future__ import annotations

from .schema import (
    MASKED_ID,
    Segment,
    Token,
    TokenKind,
    numeric_value_features,
    physical_tile_factors,
    relative_seat,
    tile_type_factors,
)


_RED_PHYSICAL_IDS = ((4, 16), (13, 52), (22, 88))


def _actor_rows(frame: dict, decision: dict, observer: int):
    segment = int(Segment.ACTOR_STATE)
    rows: list[tuple[int, ...]] = []
    numeric: list[tuple[int, int, float]] = []
    for seat, score in enumerate(frame["scores"]):
        numeric.append((len(rows), 1, float(score)))
        rows.append((segment, int(TokenKind.SCORE), 1,
                     relative_seat(observer, seat), 0, 0, 0, 0, 0, 0))
    counters = (frame.get("round_wind", 0), frame.get("hand_number", 0),
                frame.get("honba", 0), frame.get("riichi_deposits", 0),
                frame.get("live_wall_remaining", 0))
    for field, value in enumerate(counters, 1):
        numeric.append((len(rows), 2, float(value)))
        rows.append((segment, int(TokenKind.COUNTER), field, 0, 0, 0, 0, 0, 0, 0))
    dealer = int(frame.get("dealer", 255))
    rows.append((segment, int(TokenKind.COUNTER), 6,
                 relative_seat(observer, dealer), 0, 0, 0, 0, 0, 0))
    rows.append((segment, int(TokenKind.COUNTER), 7, 0, 0, 0, 0,
                 0 if dealer == 255 else (observer - dealer) % 4 + 1, 0, 0))
    for indicator in frame.get("dora_indicators", ()):
        suit, rank, red = physical_tile_factors(int(indicator))
        rows.append((segment, int(TokenKind.TILE_COUNT), 3, 0,
                     suit, rank, red, 0, 0, 1))
    flags = int(decision.get("flags", 0))
    if not 0 <= flags < 256:
        raise ValueError("decision flags exceed compact token range")
    rows.append((segment, int(TokenKind.COUNTER), 8, 1, 0, 0, 0, 0, flags, 0))
    for tile_type, count in enumerate(decision["concealed_counts"]):
        if count:
            suit, rank, red = tile_type_factors(tile_type)
            rows.append((segment, int(TokenKind.TILE_COUNT), 1, 1,
                         suit, rank, red, int(count), 0, 1))
    # Opponent state is invariantly ordinary, irrespective of native privilege.
    for relative in (2, 3, 4):
        rows.append((segment, int(TokenKind.MASKED), MASKED_ID,
                     relative, 0, 0, 0, 0, 0, 2))
    return rows, numeric


def encode_state_factors(frame: dict, decision: dict, *, observer: int):
    """Encode the public actor suffix directly into compact contiguous arrays."""
    import numpy as np

    rows, numeric = _actor_rows(frame, decision, observer)
    factors = np.asarray(rows, dtype=np.uint8).reshape(-1, 10)
    values = np.zeros((len(rows), 8), dtype=np.float32)
    for row, field, value in numeric:
        values[row] = numeric_value_features(field, value)
    return factors, values


def encode_state(frame: dict, decision: dict, *, observer: int):
    """Reference token encoder used by schema contract tests."""
    rows, numeric = _actor_rows(frame, decision, observer)
    semantic = {(row, field): value for row, field, value in numeric}
    return [Token(*row, numeric_field=next((field for index, field in semantic if index == i), 0),
                  numeric_value=next((value for (index, _), value in semantic.items() if index == i), 0.0))
            for i, row in enumerate(rows)]


def encode_critic_factors(frame: dict, *, observer: int):
    """Encode opponent hands then aggregate live wall in deterministic type order."""
    import numpy as np

    counts = frame.get("priv_concealed_counts")
    wall = frame.get("priv_wall")
    indices = frame.get("priv_wall_indices")
    if counts is None or wall is None or indices is None:
        return np.zeros((0, 10), dtype=np.uint8), np.zeros((0, 8), dtype=np.float32)
    rows: list[tuple[int, ...]] = []
    concealed_ids = frame.get("priv_concealed_tile_ids")
    for relative in (2, 3, 4):
        seat = (observer + relative - 1) % 4
        red_counts = [0] * 34
        if concealed_ids is not None:
            for tile_type, physical in _RED_PHYSICAL_IDS:
                red_counts[tile_type] = int(physical in concealed_ids[seat])
        for tile_type, count in enumerate(counts[seat]):
            if count:
                suit, rank, red = tile_type_factors(
                    tile_type, red=int(bool(red_counts[tile_type])))
                rows.append((Segment.CRITIC_PRIVATE, TokenKind.TILE_COUNT, 2,
                             relative, suit, rank, red, int(count), 0, 1))
    live_counts = [0] * 34
    live_red = [0] * 34
    live_start, live_end = int(indices[0]), int(indices[1])
    for value in wall[live_start:live_end]:
        tile = int(value)
        if 0 <= tile < 136:
            tile_type, copy = divmod(tile, 4)
            live_counts[tile_type] += 1
            live_red[tile_type] += int(tile_type in (4, 13, 22) and copy == 0)
    for tile_type, count in enumerate(live_counts):
        if count:
            suit, rank, red = tile_type_factors(tile_type, red=int(bool(live_red[tile_type])))
            rows.append((Segment.CRITIC_PRIVATE, TokenKind.TILE_COUNT, 4,
                         0, suit, rank, red, count, 0, 0))
    factors = np.asarray(rows, dtype=np.uint8).reshape(-1, 10)
    return factors, np.zeros((len(rows), 8), dtype=np.float32)
