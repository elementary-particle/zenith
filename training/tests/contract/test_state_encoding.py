import numpy as np

from zenith_ppo.encoding.schema import Segment, TokenKind, numeric_features
from zenith_ppo.encoding.state import (
    encode_state, encode_state_factors,
)


def _frame():
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
