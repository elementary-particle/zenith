from zenith_ppo.ppo.gae import compute
from zenith_ppo.types import *


def sample(index, reward, successor=None, terminal=False):
    binding = DecisionBinding(0, 1, index + 1, 0)
    return RolloutSample(binding, ObservationRecord(binding, "s", index, 0), ActionSegment(0, 1, (0,)),
        "current", 1, True, 0, 0, 0.0, 0.0, 0.0, reward=RewardRecord(discard_reward=reward),
        successor=successor, terminal=terminal)


def test_explicit_successors_and_constant_normalization():
    result = compute([sample(0, 1, 1), sample(1, 1, terminal=True)], gamma=1, gae_lambda=1)
    assert result.advantages.tolist() == [2.0, 1.0]
    constant = compute([sample(0, 0, terminal=True)], gamma=1, gae_lambda=1)
    assert constant.normalized.tolist() == [0.0]


def test_truncation_uses_actual_next_observation_value():
    value = sample(0, -1)
    value.truncated = True
    value.bootstrap_value = -4

    result = compute([value], gamma=1, gae_lambda=.98)

    assert result.returns.tolist() == [-5.0]
