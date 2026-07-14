from zenith_ppo.rewards.discard import DiscardScore, public_remaining, regret


def test_shanten_dominates_ukeire_and_public_tiles_deduplicate():
    assert regret([DiscardScore(1, 0), DiscardScore(0, 0)], 0) == -1
    assert public_remaining([1] + [0] * 33, 1, [0, 0]) == 2

