import pytest
import torch
from zenith_ppo.ppo.loss import ppo_loss


def test_clipped_huber_loss_is_finite_and_rejects_nan():
    value = ppo_loss(torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([1.0]),
                     torch.tensor([0.0]), torch.tensor([1.0]), torch.tensor([0.5]))
    assert torch.isfinite(value.total)
    with pytest.raises(FloatingPointError): ppo_loss(torch.tensor([float("nan")]), *(torch.zeros(1) for _ in range(5)))

