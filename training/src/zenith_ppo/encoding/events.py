"""One semantic numeric token per canonical event."""

from __future__ import annotations

from .schema import Segment, Token, TokenKind, physical_tile_factors, relative_seat

TILE_ARG_KINDS = {4, 5, 6, 7, 8, 9, 10, 13}


def _is_red(tile: int) -> bool:
    if not 0 <= tile < 136:
        return False
    tile_type, copy = divmod(tile, 4)
    return tile_type in (4, 13, 22) and copy == 0


def _compact_detail(kind: int, args: tuple[int, ...], *, observer: int,
                    target: int) -> int:
    """Return a small event-kind-specific categorical detail.

    Large scores/counters belong in Fourier numeric features or the current
    state suffix, never in a mostly-empty 16-bit embedding table.
    """
    if kind == 2:  # start_kyoku: wind and hand number
        wind = min(max(args[0], 0), 3)
        hand = min(max(args[1] - 1, 0), 3)
        return 1 + wind * 4 + hand
    if kind == 4:  # dahai: tedashi / tsumogiri
        return 1 + int(bool(args[1]))
    if kind == 5:  # chi: called-tile position and red-five presence
        types = [tile // 4 for tile in args if 0 <= tile < 136]
        offset = min(max(args[0] // 4 - min(types), 0), 2) if types else 0
        return 1 + offset * 2 + int(any(_is_red(tile) for tile in args))
    if kind in {6, 7, 8, 9}:  # pon/kan: red-five presence
        return 1 + int(any(_is_red(tile) for tile in args))
    if kind == 13:  # hora: relative source, including self-draw
        return relative_seat(observer, target)
    return 0


def encode_event(row: dict, *, observer: int) -> Token | None:
    kind = int(row["kind"])
    # The draw is reflected immediately in the acting player's type-count hand
    # state. Retaining a second draw token adds length without new information.
    if kind == 3:
        return None
    actor = int(row.get("actor_seat", 255))
    target = int(row.get("target_seat", 255))
    visible = int(row.get("visibility_mask", 0b1111)) & (1 << observer) != 0
    raw_args = row.get("args")
    args = tuple(int(value) for value in raw_args) if raw_args is not None else tuple(
        int(row.get(f"arg{index}", 0)) for index in range(4)
    )
    args = args + (0,) * (4 - len(args))
    arg0 = args[0]
    if kind in TILE_ARG_KINDS and visible:
        suit, rank, red = physical_tile_factors(arg0)
    else:
        suit = rank = red = 0
    detail = _compact_detail(kind, args, observer=observer, target=target) \
        if visible else 0
    # Only semantic magnitudes use the numeric channel. In particular, a
    # physical tile ID is an implementation identity, not a tile magnitude;
    # equivalent non-red copies must therefore have identical encodings.
    numeric_field = 1 if kind == 12 and visible else 0
    numeric_value = float(arg0) if numeric_field else 0.0
    return Token(Segment.EVENT, TokenKind.EVENT, field=kind,
        seat=relative_seat(observer, actor), tile_suit=suit, tile_rank=rank, tile_red=red,
        flag=detail, visibility=1 if visible else 2,
        numeric_field=numeric_field,
        numeric_value=numeric_value)


def encode_history(rows, *, observer: int, generation: int):
    expected = 0
    tokens = []
    for row in rows:
        if int(row["episode_generation"]) != generation:
            raise ValueError("event generation changed within history")
        sequence = int(row["sequence"])
        if sequence != expected:
            raise ValueError(f"event sequence gap: expected {expected}, got {sequence}")
        if int(row["kind"]) == 2:
            tokens.clear()
        token = encode_event(row, observer=observer)
        if token is not None:
            tokens.append(token)
        expected += 1
    return tokens
