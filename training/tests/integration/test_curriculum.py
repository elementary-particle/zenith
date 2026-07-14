from zenith_ppo.rewards.curriculum import Curriculum


def test_short_schedule_visits_both_blends():
    config = {"total_updates": 20, "discard_only_end": .2, "discard_kyoku_blend_end": .35,
              "kyoku_only_end": .6, "kyoku_rank_blend_end": .75, "oracle_reveal_end": .6,
              "oracle_groups": ["opponent_concealed_hands"]}
    values = [Curriculum(config).snapshot(i, i) for i in range(21)]
    assert any(v.weights[0] and v.weights[1] for v in values)
    assert any(v.weights[1] and v.weights[2] for v in values)

