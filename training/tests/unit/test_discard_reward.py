from zenith_ppo.types import RewardRecord


def test_reward_record_contains_only_outcome_components():
    reward = RewardRecord(kyoku_delta=2.5, rank_reward=-1.0, weights=(1.0, .25))
    assert reward.total == 2.25
    assert not hasattr(reward, "discard_reward")
