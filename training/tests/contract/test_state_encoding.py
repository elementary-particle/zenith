import numpy as np

from zenith_ppo.encoding.schema import numeric_features
from zenith_ppo.encoding.state import encode_critic_factors, encode_state, encode_state_factors


def test_concealed_hands_use_sparse_type_counts_and_one_mask_token_per_opponent():
    concealed = [[16], [0, 1, 16], [36], [72]]
    frame = {"scores": [25000] * 4, "dealer": 2, "dora_indicators": [52],
             "priv_concealed_tile_ids": concealed}
    decision = {"concealed_counts": [0] * 34, "flags": 0b101100}
    decision["concealed_counts"][4] = 1
    ordinary = encode_state(frame, decision, observer=0)
    assert sum(token.kind == 7 for token in ordinary) == 3
    assert {token.seat for token in ordinary if token.kind == 7} == {2, 3, 4}
    own = next(token for token in ordinary if token.kind == 4 and token.seat == 1)
    assert own.tile_rank == 5 and own.count == 1 and own.tile_red == 0
    assert not [token for token in ordinary if token.kind == 4 and token.field == 2]
    dealer = next(token for token in ordinary if token.kind == 3 and token.field == 6)
    seat_wind = next(token for token in ordinary if token.kind == 3 and token.field == 7)
    own_flags = next(token for token in ordinary if token.kind == 3 and token.field == 8)
    dora = next(token for token in ordinary if token.kind == 4 and token.field == 3)
    assert dealer.seat == 3 and seat_wind.count == 3
    assert own_flags.flag == 0b101100
    assert (dora.tile_suit, dora.tile_rank) == (2, 5)


def test_contiguous_training_encoder_matches_semantic_reference():
    concealed = [[16, 20], [0, 1, 16], [36], [72]]
    frame = {
        "scores": [12_300, 25_000, 31_700, 31_000],
        "round_wind": 1,
        "hand_number": 3,
        "honba": 2,
        "riichi_deposits": 1,
        "live_wall_remaining": 44,
        "dealer": 2,
        "dora_indicators": [52],
        "priv_concealed_tile_ids": concealed,
    }
    decision = {"concealed_counts": [0] * 34, "flags": 7}
    decision["concealed_counts"][4] = 1
    decision["concealed_counts"][5] = 1
    kwargs = dict(observer=0)
    reference = encode_state(frame, decision, **kwargs)
    factors, numeric = encode_state_factors(frame, decision, **kwargs)
    np.testing.assert_array_equal(
        factors, np.asarray([token.categorical() for token in reference], dtype=np.uint8)
    )
    np.testing.assert_allclose(
        numeric,
        np.asarray([numeric_features(token) for token in reference], dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )


def test_critic_tokens_are_canonical_and_ignore_wall_order_and_dead_wall():
    counts = [[0] * 34 for _ in range(4)]
    counts[1][7] = 4
    counts[2][1] = 2
    counts[3][33] = 1
    wall = list(range(136))
    frame = {
        "priv_concealed_counts": counts,
        "priv_concealed_tile_ids": [[], [], [], []],
        "priv_wall": wall,
        "priv_wall_indices": (10, 50, 0, 1),
    }
    factors, _ = encode_critic_factors(frame, observer=0)
    hand = factors[factors[:, 2] == 2]
    assert hand[:, 3].tolist() == [2, 3, 4]
    assert hand[:, 7].tolist() == [4, 2, 1]
    assert set(factors[factors[:, 2] == 4, 2]) == {4}

    reordered = dict(frame)
    reordered["priv_wall"] = wall[:10] + list(reversed(wall[10:50])) + wall[50:]
    reordered["ura_indicators"] = [135, 134]
    actual, _ = encode_critic_factors(reordered, observer=0)
    np.testing.assert_array_equal(actual, factors)

    dead_changed = dict(frame)
    dead_changed["priv_wall"] = [135] * 10 + wall[10:50] + [0] * 86
    actual, _ = encode_critic_factors(dead_changed, observer=0)
    np.testing.assert_array_equal(actual, factors)


def test_critic_opponent_order_rotates_with_observer():
    counts = [[0] * 34 for _ in range(4)]
    for seat in range(4):
        counts[seat][seat] = seat + 1
    frame = {"priv_concealed_counts": counts, "priv_concealed_tile_ids": [[], [], [], []],
             "priv_wall": [], "priv_wall_indices": (0, 0, 0, 0)}
    factors, _ = encode_critic_factors(frame, observer=2)
    hand = factors[factors[:, 2] == 2]
    assert hand[:, 3].tolist() == [2, 3, 4]
    assert hand[:, 7].tolist() == [4, 1, 2]
