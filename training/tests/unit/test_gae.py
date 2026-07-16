import pytest

from zenith_ppo.ppo.gae import compute, explained_variance
from zenith_ppo.types import *


def sample(index, score_reward, successor=None, terminal=False, *, rank_reward=0.0,
           weights=(1.0, 0.0), score_value=0.0, rank_value=0.0):
    binding = DecisionBinding(0, 1, index + 1, 0)
    return RolloutSample(binding, ObservationRecord(binding, "s", index, 0), ActionSegment(0, 1, (0,)),
        "current", 1, True, 0, 0, 0.0, 0.0, score_value, rank_value,
        reward=RewardRecord(
            kyoku_delta=score_reward, rank_reward=rank_reward, weights=weights
        ),
        successor=successor, terminal=terminal, terminal_placement=0)


def test_explicit_successors_and_constant_normalization():
    result = compute(
        [sample(0, 1, 1), sample(1, 1, terminal=True)],
        gamma=1,
        score_gae_lambda=1,
        rank_gae_lambda=1,
    )
    assert result.advantages.tolist() == [2.0, 1.0]
    assert result.score_advantages.tolist() == [2.0, 1.0]
    assert result.rank_advantages.tolist() == [0.0, 0.0]
    constant = compute(
        [sample(0, 0, terminal=True)],
        gamma=1,
        score_gae_lambda=1,
        rank_gae_lambda=1,
    )
    assert constant.normalized.tolist() == [0.0]


def test_truncation_is_rejected_for_complete_match_trajectories():
    value = sample(0, -1)
    value.truncated = True
    value.bootstrap_score_value = -4
    value.bootstrap_rank_value = 3

    with pytest.raises(ValueError, match="cannot be truncated"):
        compute(
            [value], gamma=1, score_gae_lambda=.98, rank_gae_lambda=1,
        )


def test_score_and_rank_gae_are_combined_only_after_curriculum_weighting():
    result = compute([
        sample(
            0, 4.0, terminal=True, rank_reward=-2.0,
            weights=(0.25, 0.75), score_value=1.0, rank_value=-1.0,
        )
    ], gamma=1, score_gae_lambda=1, rank_gae_lambda=1)

    assert result.score_advantages.tolist() == [3.0]
    assert result.rank_advantages.tolist() == [-1.0]
    assert result.advantages.tolist() == [0.0]
    assert result.score_returns.tolist() == [4.0]
    assert result.rank_returns.tolist() == [-2.0]


def test_score_and_rank_traces_use_separate_lambdas():
    first = sample(0, 0.0, successor=1, weights=(0.5, 0.5))
    second = sample(
        1, 2.0, terminal=True, rank_reward=4.0, weights=(0.5, 0.5),
    )

    result = compute(
        [first, second], gamma=1, score_gae_lambda=.5, rank_gae_lambda=1,
    )

    assert result.score_advantages.tolist() == [1.0, 2.0]
    assert result.rank_advantages.tolist() == [4.0, 4.0]
    assert result.score_returns.tolist() == [1.0, 2.0]
    assert result.rank_returns.tolist() == [4.0, 4.0]


def test_score_trace_stops_at_kyoku_while_rank_trace_reaches_match_end():
    first = sample(0, 1.0, successor=1, weights=(0.5, 0.5))
    first.kyoku_boundary = True
    second = sample(
        1, 2.0, terminal=True, rank_reward=4.0, weights=(0.5, 0.5),
    )

    result = compute(
        [first, second], gamma=1, score_gae_lambda=1, rank_gae_lambda=1,
    )

    assert result.score_advantages.tolist() == [1.0, 2.0]
    assert result.rank_advantages.tolist() == [4.0, 4.0]
    assert result.score_returns.tolist() == [1.0, 2.0]
    assert result.rank_returns.tolist() == [4.0, 4.0]


def test_explained_variance_uses_rollout_values_and_handles_constant_targets():
    assert explained_variance([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    assert explained_variance([3, 2, 1], [1, 2, 3]) == pytest.approx(-3.0)
    assert explained_variance([1, 2], [4, 4]) == 0.0
    with pytest.raises(ValueError, match="same shape"):
        explained_variance([1], [1, 2])


def test_nonterminal_tail_cannot_silently_turn_into_a_terminal_target():
    broken = sample(0, 1)
    with pytest.raises(ValueError, match="neither linked nor truncated"):
        compute(
            [broken], gamma=1, score_gae_lambda=.98, rank_gae_lambda=1,
        )


def test_successor_cannot_cross_seat_trajectories():
    first = sample(0, 0, successor=1)
    second = sample(1, 0, terminal=True)
    second.binding = DecisionBinding(0, 1, 2, 1)
    with pytest.raises(ValueError, match="crosses a seat trajectory"):
        compute(
            [first, second], gamma=1, score_gae_lambda=.98, rank_gae_lambda=1,
        )


def test_truncated_tail_is_rejected_even_without_bootstrap():
    broken = sample(0, 0)
    broken.truncated = True
    with pytest.raises(ValueError, match="cannot be truncated"):
        compute(
            [broken], gamma=1, score_gae_lambda=.98, rank_gae_lambda=1,
        )


def test_gamma_must_be_one():
    with pytest.raises(ValueError, match="gamma = 1"):
        compute(
            [sample(0, 0, terminal=True)],
            gamma=.99,
            score_gae_lambda=.98,
            rank_gae_lambda=1,
        )


def test_rank_lambda_must_be_exact_match_long():
    with pytest.raises(ValueError, match="must equal 1"):
        compute(
            [sample(0, 0, terminal=True)], gamma=1,
            score_gae_lambda=.98, rank_gae_lambda=.9975,
        )


def test_one_gae_batch_rejects_mixed_curriculum_weights():
    first = sample(0, 0, successor=1, weights=(1.0, 0.0))
    second = sample(1, 0, terminal=True, weights=(0.0, 1.0))
    with pytest.raises(ValueError, match="cannot mix curriculum weights"):
        compute(
            [first, second], gamma=1, score_gae_lambda=.98, rank_gae_lambda=1,
        )
