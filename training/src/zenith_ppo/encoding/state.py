"""Contextual public actor-state encoding."""

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


# Exact-current-state public snapshot flags.  They are separate from ordered
# event history so tile geometry does not have to reconstruct mutable state.
SNAPSHOT_RIVER_RIICHI = 1
SNAPSHOT_RIVER_CALLED = 2
SNAPSHOT_RIVER_TSUMOGIRI = 4
SNAPSHOT_SEAT_FLAGS_FIELD = 9
PUBLIC_SNAPSHOT_SEAT_FLAGS_MASK = 0b111


def _match_rows(frame: dict, observer: int):
    segment = int(Segment.MATCH_STATE)
    rows: list[tuple[int, ...]] = []
    numeric: list[tuple[int, int, float]] = []
    for seat, score in enumerate(frame["scores"]):
        numeric.append((len(rows), 1, float(score)))
        rows.append((segment, int(TokenKind.SCORE), 1,
                     relative_seat(observer, seat), 0, 0, 0, 0, 0, 0))
    counters = (frame.get("round_wind", 0), frame.get("hand_number", 0),
                frame.get("honba", 0), frame.get("riichi_deposits", 0))
    for field, value in enumerate(counters, 1):
        numeric.append((len(rows), 2, float(value)))
        rows.append((segment, int(TokenKind.COUNTER), field, 0, 0, 0, 0, 0, 0, 0))
    dealer = int(frame.get("dealer", 255))
    rows.append((segment, int(TokenKind.COUNTER), 6,
                 relative_seat(observer, dealer), 0, 0, 0, 0, 0, 0))
    rows.append((segment, int(TokenKind.COUNTER), 7, 0, 0, 0, 0,
                 0 if dealer == 255 else (observer - dealer) % 4 + 1, 0, 0))
    return rows, numeric


def _tactical_rows(
    frame: dict, decision: dict, observer: int, *,
    include_public_snapshot: bool = False,
):
    segment = int(Segment.KYOKU_STATE)
    rows: list[tuple[int, ...]] = []
    numeric: list[tuple[int, int, float]] = []
    numeric.append((len(rows), 2, float(frame.get("live_wall_remaining", 0))))
    rows.append((segment, int(TokenKind.COUNTER), 5, 0, 0, 0, 0, 0, 0, 0))
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
    if include_public_snapshot:
        seat_flags = tuple(frame.get("seat_flags", (0, 0, 0, 0)))
        if len(seat_flags) != 4:
            raise ValueError("public snapshot requires four seat flags")
        for seat, seat_flag in enumerate(seat_flags):
            # Declared/accepted riichi and ippatsu are public.  Native bits
            # above bit two contain furiten state that can depend on a
            # concealed wait and must not enter a public observation.
            seat_flag = int(seat_flag) & PUBLIC_SNAPSHOT_SEAT_FLAGS_MASK
            if not 0 <= seat_flag < 256:
                raise ValueError("seat flags exceed compact token range")
            rows.append((
                segment, int(TokenKind.COUNTER), SNAPSHOT_SEAT_FLAGS_FIELD,
                relative_seat(observer, seat), 0, 0, 0, 0, seat_flag, 1,
            ))
        for river in frame.get("rivers", ()):
            suit, rank, red = physical_tile_factors(int(river["tile"]))
            river_flags = (
                SNAPSHOT_RIVER_RIICHI * int(bool(river.get("riichi_declaration")))
                | SNAPSHOT_RIVER_CALLED * int(bool(river.get("called")))
                | SNAPSHOT_RIVER_TSUMOGIRI * int(bool(river.get("tsumogiri")))
            )
            rows.append((
                segment, int(TokenKind.RIVER), 1,
                relative_seat(observer, int(river["seat"])),
                suit, rank, red, 1, river_flags, 1,
            ))
        for meld in frame.get("melds", ()):
            seat = relative_seat(observer, int(meld["seat"]))
            meld_kind = int(meld["kind"])
            if not 0 <= meld_kind < 256:
                raise ValueError("meld kind exceeds compact token range")
            for tile in meld.get("tiles", ()):
                suit, rank, red = physical_tile_factors(int(tile))
                rows.append((
                    segment, int(TokenKind.MELD), meld_kind, seat,
                    suit, rank, red, 1, 0, 1,
                ))
    # Opponent state is invariantly ordinary, irrespective of native privilege.
    for relative in (2, 3, 4):
        rows.append((segment, int(TokenKind.MASKED), MASKED_ID,
                     relative, 0, 0, 0, 0, 0, 2))
    return rows, numeric


def _arrays(rows, numeric):
    import numpy as np

    factors = np.asarray(rows, dtype=np.uint8).reshape(-1, 10)
    values = np.zeros((len(rows), 8), dtype=np.float32)
    for row, field, value in numeric:
        values[row] = numeric_value_features(field, value)
    return factors, values


def encode_match_state_factors(frame: dict, *, observer: int):
    return _arrays(*_match_rows(frame, observer))


def encode_tactical_state_factors(
    frame: dict, decision: dict, *, observer: int,
    include_public_snapshot: bool = False,
):
    return _arrays(*_tactical_rows(
        frame, decision, observer,
        include_public_snapshot=include_public_snapshot,
    ))


def encode_state_factors(frame: dict, decision: dict, *, observer: int):
    """Encode state without history, retaining semantic segment ordering."""
    import numpy as np

    match_factors, match_numeric = encode_match_state_factors(frame, observer=observer)
    tactical_factors, tactical_numeric = encode_tactical_state_factors(
        frame, decision, observer=observer
    )
    summaries = np.asarray([
        (Segment.MATCH_SUMMARY, TokenKind.QUERY, 2, 1, 0, 0, 0, 0, 0, 0),
        (Segment.KYOKU_SUMMARY, TokenKind.QUERY, 3, 1, 0, 0, 0, 0, 0, 0),
    ], dtype=np.uint8)
    zero = np.zeros((1, 8), dtype=np.float32)
    return (
        np.concatenate((match_factors, summaries[:1], tactical_factors, summaries[1:])),
        np.concatenate((match_numeric, zero, tactical_numeric, zero)),
    )


def encode_state(frame: dict, decision: dict, *, observer: int):
    """Reference token encoder used by schema contract tests."""
    match_rows, match_numeric = _match_rows(frame, observer)
    tactical_rows, tactical_numeric = _tactical_rows(frame, decision, observer)
    rows = [
        *match_rows,
        (Segment.MATCH_SUMMARY, TokenKind.QUERY, 2, 1, 0, 0, 0, 0, 0, 0),
        *tactical_rows,
        (Segment.KYOKU_SUMMARY, TokenKind.QUERY, 3, 1, 0, 0, 0, 0, 0, 0),
    ]
    tactical_offset = len(match_rows) + 1
    numeric = [
        *match_numeric,
        *((row + tactical_offset, field, value)
          for row, field, value in tactical_numeric),
    ]
    semantic = {(row, field): value for row, field, value in numeric}
    return [Token(*row, numeric_field=next((field for index, field in semantic if index == i), 0),
                  numeric_value=next((value for (index, _), value in semantic.items() if index == i), 0.0))
            for i, row in enumerate(rows)]
