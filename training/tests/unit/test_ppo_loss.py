import pytest
import torch

from zenith_ppo.ppo.loss import (
    actor_loss,
    legal_action_entropy,
    segmented_forward_kl,
)


def test_actor_clipped_objective_is_finite_and_rejects_nan():
    value = actor_loss(
        torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([1.0]),
        torch.tensor([0.5]), torch.tensor([0.5]), torch.tensor([True]),
    )
    assert torch.isfinite(value.total)
    with pytest.raises(FloatingPointError):
        actor_loss(torch.tensor([float("nan")]), *(torch.zeros(1) for _ in range(5)))


def test_entropy_regularization_uses_raw_shannon_entropy():
    entropy = torch.tensor([
        torch.log(torch.tensor(2.0)),
        torch.log(torch.tensor(8.0)),
    ])
    result = actor_loss(
        torch.zeros(2), torch.zeros(2), torch.zeros(2), entropy,
        torch.ones(2),
        torch.ones(2, dtype=torch.bool),
        entropy_coefficient=.01,
    )
    assert result.entropy_efficiency == pytest.approx(1.0)
    assert result.entropy_loss == pytest.approx(-.01 * entropy.mean())


def test_inapplicable_rows_have_zero_entropy_and_remain_in_row_average():
    result = actor_loss(
        torch.zeros(3), torch.zeros(3), torch.zeros(3),
        torch.tensor([0.0, 1.0, 0.5]),
        torch.tensor([0.0, 1.0, 0.5]),
        torch.tensor([False, True, True]),
    )
    assert result.entropy_rows == 2
    assert result.entropy_efficiency == pytest.approx(.5)


def test_segmented_forward_kl_uses_reference_policy_direction():
    reference = torch.tensor([0.8, 0.2, 1.0]).log()
    current = torch.tensor([0.5, 0.5, 1.0]).log().requires_grad_()
    result = segmented_forward_kl(
        reference, current, torch.tensor([0, 2, 3])
    )
    expected = 0.8 * torch.log(torch.tensor(0.8 / 0.5)) \
        + 0.2 * torch.log(torch.tensor(0.2 / 0.5))
    assert result.tolist() == pytest.approx([expected.item(), 0.0])
    result.sum().backward()
    assert torch.isfinite(current.grad).all()


def test_legal_action_entropy_pushes_call_versus_pass_mass():
    # Pass and both calls share one legal-action distribution, so entropy must
    # directly affect the total probability assigned to calling.
    raw = torch.tensor([0.7, -0.2, 0.4], requires_grad=True)
    logp = raw.log_softmax(0)
    entropy, efficiency, applicable = legal_action_entropy(
        logp, torch.tensor([0, 3]), torch.tensor([3]),
    )
    assert applicable.tolist() == [True]
    assert 0 < efficiency.item() < 1
    entropy.sum().backward()
    assert raw.grad[0].abs() > 1e-4
    assert (raw.grad[1] + raw.grad[2]).abs() > 1e-4
    assert raw.grad.sum() == pytest.approx(0.0, abs=1e-7)


def test_legal_action_entropy_ignores_only_singleton_rows():
    uniform_three = torch.full((3,), 1 / 3).log()
    entropy, efficiency, applicable = legal_action_entropy(
        torch.cat((torch.tensor([0.0]), uniform_three)),
        torch.tensor([0, 1, 4]), torch.tensor([1, 3]),
    )
    assert applicable.tolist() == [False, True]
    assert entropy[0] == 0
    assert entropy[1] == pytest.approx(torch.log(torch.tensor(3.0)).item())
    assert efficiency.tolist() == pytest.approx([0.0, 1.0])


def test_legal_action_entropy_is_stable_for_negligible_action_mass():
    raw = torch.tensor([0.0, -1_000.0, -1_000.0], requires_grad=True)
    entropy, efficiency, applicable = legal_action_entropy(
        raw.log_softmax(0), torch.tensor([0, 3]), torch.tensor([3]),
    )
    assert applicable.tolist() == [True]
    assert torch.isfinite(entropy).all()
    assert entropy.item() == pytest.approx(0.0)
    assert efficiency.item() == pytest.approx(0.0)
    entropy.sum().backward()
    assert torch.isfinite(raw.grad).all()
