from zenith_ppo.rewards.curriculum import Curriculum


CONFIG = {"total_updates": 100, "discard_only_end": .2, "discard_kyoku_blend_end": .35,
          "kyoku_only_end": .6, "kyoku_rank_blend_end": .75}


def test_weights_are_convex_and_finish_rank_only():
    values = [Curriculum(CONFIG).snapshot(update, 0) for update in range(101)]
    assert all(abs(sum(value.weights) - 1) < 1e-7 for value in values)
    assert values[-1].weights == (0.0, 0.0, 1.0)
