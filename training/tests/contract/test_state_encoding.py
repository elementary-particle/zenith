import numpy as np

from zenith_ppo.encoding.schema import Segment, TokenKind, numeric_features
from zenith_ppo.encoding.state import (
    encode_oracle_factors, encode_state, encode_state_factors,
)


def _frame():
    counts = [[0] * 34 for _ in range(4)]
    for seat in range(4):
        counts[seat][seat] = seat + 1
    return {
        "scores": [12_300, 25_000, 31_700, 31_000],
        "round_wind": 1, "hand_number": 3, "honba": 2,
        "riichi_deposits": 1, "live_wall_remaining": 44,
        "dealer": 2, "dora_indicators": [52], "seat_flags": [0, 1, 2, 4],
        "rivers": ({"seat": 1, "tile": 8, "sequence": 7,
                    "riichi_declaration": True, "called": False,
                    "tsumogiri": True},),
        "melds": ({"seat": 3, "kind": 1, "from_seat": 1,
                   "tiles": (0, 4, 8), "created_sequence": 8},),
        "priv_concealed_counts": counts,
        "priv_concealed_tile_ids": [[], [], [], []],
        "priv_live_wall_counts": [1] * 34,
    }


def test_actor_state_has_match_then_kyoku_summaries_and_reference_parity():
    frame = _frame()
    decision = {"concealed_counts": [0] * 34, "flags": 7}
    decision["concealed_counts"][4] = 1
    reference = encode_state(frame, decision, observer=0)
    factors, numeric = encode_state_factors(frame, decision, observer=0)
    np.testing.assert_array_equal(
        factors, np.asarray([token.categorical() for token in reference], np.uint8)
    )
    np.testing.assert_allclose(
        numeric, np.asarray([numeric_features(token) for token in reference], np.float32),
        rtol=1e-6, atol=1e-6,
    )
    segments = factors[:, 0].tolist()
    assert segments.index(Segment.MATCH_SUMMARY) < segments.index(Segment.KYOKU_STATE)
    assert segments[-1] == Segment.KYOKU_SUMMARY
    assert sum(token.kind == TokenKind.MASKED for token in reference) == 3


def test_oracle_is_dealer_canonical_counts_only_and_carries_order_fields():
    frame = _frame()
    factors, _ = encode_oracle_factors(frame)
    hand = factors[(factors[:, 1] == TokenKind.TILE_COUNT) & (factors[:, 2] == 2)]
    assert hand[:, 3].tolist() == [1, 2, 3, 4]
    assert hand[:, 7].tolist() == [3, 4, 1, 2]
    assert set(factors[factors[:, 2] == 4, 2]) == {4}
    river = factors[factors[:, 1] == TokenKind.RIVER]
    meld = factors[factors[:, 1] == TokenKind.MELD]
    assert river[0, 8] == 1
    assert meld[:, 8].tolist() == [1, 1, 1]

    # Exact wall order is not an oracle input. Incidental legacy fields cannot
    # affect the aggregate count encoding.
    changed = dict(frame, priv_wall=list(reversed(range(136))),
                   priv_wall_indices=(1, 99, 120, 2))
    actual, _ = encode_oracle_factors(changed)
    np.testing.assert_array_equal(actual, factors)


def test_rotating_absolute_seats_with_the_dealer_preserves_canonical_factors():
    frame = _frame()
    expected, expected_numeric = encode_oracle_factors(frame)
    shift = 1
    rotated = dict(frame)
    rotated["dealer"] = (frame["dealer"] + shift) % 4
    for name in ("scores", "seat_flags", "priv_concealed_counts",
                 "priv_concealed_tile_ids"):
        values = frame[name]
        rotated[name] = [values[(seat - shift) % 4] for seat in range(4)]
    rotated["rivers"] = tuple(
        dict(row, seat=(row["seat"] + shift) % 4) for row in frame["rivers"]
    )
    rotated["melds"] = tuple(
        dict(row, seat=(row["seat"] + shift) % 4,
             from_seat=(row["from_seat"] + shift) % 4)
        for row in frame["melds"]
    )
    actual, actual_numeric = encode_oracle_factors(rotated)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual_numeric, expected_numeric)
