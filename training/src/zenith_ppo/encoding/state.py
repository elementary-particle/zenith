"""Contextual actor state and dealer-canonical counts-only oracle encoding."""

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


def _tactical_rows(frame: dict, decision: dict, observer: int):
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


def encode_tactical_state_factors(frame: dict, decision: dict, *, observer: int):
    return _arrays(*_tactical_rows(frame, decision, observer))


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


def _dealer_seat(dealer: int, seat: int) -> int:
    """One-based dealer-canonical seat factor (dealer is one)."""
    return (int(seat) - int(dealer)) % 4 + 1


def encode_oracle_factors(frame: dict):
    """Encode one dealer-canonical oracle snapshot without future wall order."""
    import numpy as np

    counts = frame.get("priv_concealed_counts")
    live_counts = frame.get("priv_live_wall_counts")
    if counts is None or live_counts is None:
        return np.zeros((0, 10), dtype=np.uint8), np.zeros((0, 8), dtype=np.float32)
    segment = int(Segment.ORACLE)
    dealer = int(frame.get("dealer", 0))
    rows: list[tuple[int, ...]] = []
    numeric: list[tuple[int, int, float]] = []
    scores = frame.get("scores", (0, 0, 0, 0))
    for relative in range(4):
        seat = (dealer + relative) % 4
        score = scores[seat]
        numeric.append((len(rows), 1, float(score)))
        rows.append((segment, TokenKind.SCORE, 1, _dealer_seat(dealer, seat),
                     0, 0, 0, 0, 0, 0))
    for field, value in enumerate((
        frame.get("round_wind", 0), frame.get("hand_number", 0),
        frame.get("honba", 0), frame.get("riichi_deposits", 0),
        frame.get("live_wall_remaining", 0),
    ), 1):
        numeric.append((len(rows), 2, float(value)))
        rows.append((segment, TokenKind.COUNTER, field, 0, 0, 0, 0, 0, 0, 0))
    rows.append((segment, TokenKind.COUNTER, 6, 1, 0, 0, 0, 0, 0, 0))
    rows.append((segment, TokenKind.COUNTER, 9, 0, 0, 0, 0, 0,
                 min(int(frame.get("phase", 0)), 255), 0))
    rows.append((segment, TokenKind.COUNTER, 10, 0, 0, 0, 0, 0,
                 min(int(frame.get("eligible_mask", 0)), 255), 0))
    seat_flags = frame.get("seat_flags", (0, 0, 0, 0))
    for relative in range(4):
        seat = (dealer + relative) % 4
        flags = seat_flags[seat]
        rows.append((segment, TokenKind.COUNTER, 8, _dealer_seat(dealer, seat),
                     0, 0, 0, 0, min(int(flags), 255), 0))
    for order, indicator in enumerate(frame.get("dora_indicators", ()), 1):
        suit, rank, red = physical_tile_factors(int(indicator))
        rows.append((segment, TokenKind.TILE_COUNT, 3, 0, suit, rank, red,
                     1, min(order, 255), 1))

    for order, river in enumerate(sorted(
        frame.get("rivers", ()), key=lambda row: int(row["sequence"])
    ), 1):
        suit, rank, red = physical_tile_factors(int(river["tile"]))
        flags = (int(bool(river.get("riichi_declaration")))
                 | int(bool(river.get("called"))) << 1
                 | int(bool(river.get("tsumogiri"))) << 2)
        rows.append((segment, TokenKind.RIVER, 1 + flags,
                     _dealer_seat(dealer, river["seat"]), suit, rank, red,
                     0, min(order, 255), 1))
    for meld_order, meld in enumerate(sorted(
        frame.get("melds", ()), key=lambda row: int(row["created_sequence"])
    ), 1):
        for tile_order, tile in enumerate(meld.get("tiles", ()), 1):
            suit, rank, red = physical_tile_factors(int(tile))
            source = 0 if meld.get("from_seat") is None else _dealer_seat(
                dealer, meld["from_seat"]
            )
            rows.append((segment, TokenKind.MELD, min(int(meld["kind"]) + 1, 255),
                         _dealer_seat(dealer, meld["seat"]), suit, rank, red,
                         min(tile_order, 15), min(meld_order, 255), source))

    concealed_ids = frame.get("priv_concealed_tile_ids")
    for relative in range(4):
        seat = (dealer + relative) % 4
        red_counts = [0] * 34
        if concealed_ids is not None:
            for tile_type, physical in _RED_PHYSICAL_IDS:
                red_counts[tile_type] = int(physical in concealed_ids[seat])
        for tile_type, count in enumerate(counts[seat]):
            if count:
                suit, rank, red = tile_type_factors(
                    tile_type, red=int(bool(red_counts[tile_type])))
                rows.append((segment, TokenKind.TILE_COUNT, 2,
                             _dealer_seat(dealer, seat), suit, rank, red,
                             int(count), 0, 1))
    for tile_type, count in enumerate(live_counts):
        if count:
            suit, rank, red = tile_type_factors(tile_type)
            rows.append((segment, TokenKind.TILE_COUNT, 4,
                         0, suit, rank, red, count, 0, 0))
    return _arrays(rows, numeric)
