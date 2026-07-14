from zenith_ppo.rewards.kyoku import rewards


def test_score_delta_is_per_seat_points_over_thousand():
    assert rewards([25000] * 4, [26000, 24000, 25000, 25000]) == (1, -1, 0, 0)

