from zenith_ppo.rewards.ranking import rewards


def test_rank_reward_is_terminal_only():
    assert rewards([0, 1, 2, 3]) == (1, 1/3, -1/3, -1)
    assert rewards([0, 1, 2, 3], terminal=False) == (0, 0, 0, 0)
