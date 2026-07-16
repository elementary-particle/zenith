import pytest
import torch

from zenith_ppo.ppo.loss import actor_loss, critic_loss


def test_actor_clipped_objective_is_finite_and_rejects_nan():
    value = actor_loss(
        torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([1.0]),
        torch.tensor([0.5]),
    )
    assert torch.isfinite(value.total)
    with pytest.raises(FloatingPointError):
        actor_loss(torch.tensor([float("nan")]), *(torch.zeros(1) for _ in range(3)))


def test_score_mse_is_normalized_while_raw_errors_remain_thousand_points():
    result = critic_loss(
        torch.tensor([20.0, -10.0]), torch.zeros(2), torch.zeros(2),
        torch.zeros(2, 4), torch.tensor([0, 3]), torch.tensor([1.0, -1.0]),
        score_value_scale=10.0, value_clip=0.0,
    )
    assert result.score_mse == pytest.approx(2.5)
    assert result.score_mae == pytest.approx(15.0)
    assert result.score_rmse == pytest.approx((250.0) ** .5)
    assert result.score_clip_fraction == 0


def test_score_clipping_is_raw_units_before_normalized_comparison():
    result = critic_loss(
        torch.tensor([20.0]), torch.tensor([0.0]), torch.tensor([10.0]),
        torch.zeros(1, 4), torch.tensor([1]), torch.tensor([0.0]),
        score_value_scale=10.0, value_clip=2.0,
    )
    # Conservative max compares errors from raw predictions 20 and 2 against 10.
    assert result.score_mse == pytest.approx(1.0)
    assert result.score_clip_fraction == 1.0


def test_rank_is_categorical_with_expected_utility_metrics_and_no_clipping():
    logits = torch.tensor([[8.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 8.0]])
    result = critic_loss(
        torch.zeros(2), torch.zeros(2), torch.zeros(2), logits,
        torch.tensor([0, 3]), torch.tensor([1.0, -1.0]),
    )
    assert result.rank_accuracy == 1.0
    assert result.rank_cross_entropy < 0.01
    assert result.rank_brier < 0.01
    assert result.rank_utility_explained_variance > 0.99


def test_invalid_scales_clips_and_placement_classes_are_rejected():
    args = (
        torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1, 4),
        torch.zeros(1, dtype=torch.long), torch.zeros(1),
    )
    with pytest.raises(ValueError, match="positive"):
        critic_loss(*args, score_value_scale=0)
    with pytest.raises(ValueError, match="non-negative"):
        critic_loss(*args, value_clip=-.1)
    with pytest.raises(ValueError, match="placement"):
        critic_loss(*args[:4], torch.tensor([4]), args[5])
