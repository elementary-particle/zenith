import pytest
import torch

from zenith_ppo.ppo.loss import (
    actor_loss,
    conditional_family_entropy,
    segmented_forward_kl,
)


def test_actor_clipped_objective_is_finite_and_rejects_nan():
    value = actor_loss(
        torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([1.0]),
        torch.tensor([0.5]), torch.tensor([1.0]),
    )
    assert torch.isfinite(value.total)
    with pytest.raises(FloatingPointError):
        actor_loss(torch.tensor([float("nan")]), *(torch.zeros(1) for _ in range(4)))


def test_entropy_regularization_is_decoupled_from_legal_action_count():
    entropy = torch.tensor([torch.log(torch.tensor(2.0)), torch.log(torch.tensor(8.0))])
    result = actor_loss(
        torch.zeros(2), torch.zeros(2), torch.zeros(2), entropy, entropy.clone(),
        entropy_coefficient=.01,
    )
    assert result.entropy_efficiency == pytest.approx(1.0)
    assert result.entropy_loss == pytest.approx(-.01)


def test_single_action_rows_are_excluded_from_entropy_efficiency():
    result = actor_loss(
        torch.zeros(3), torch.zeros(3), torch.zeros(3),
        torch.tensor([0.0, torch.log(torch.tensor(2.0)), torch.log(torch.tensor(2.0))]),
        torch.tensor([0.0, torch.log(torch.tensor(2.0)), torch.log(torch.tensor(4.0))]),
    )
    assert result.entropy_rows == 2
    assert result.entropy_efficiency == pytest.approx(.75)


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


def test_conditional_family_entropy_does_not_push_call_family_mass():
    # One pass and two call variants. Shifting both call logits together
    # changes total call probability but not the conditional call entropy.
    raw = torch.tensor([0.7, -0.2, 0.4], requires_grad=True)
    logp = raw.log_softmax(0)
    factors = torch.zeros(1, 3, 15, dtype=torch.long)
    factors[0, :, 0] = torch.tensor([0, 3, 4])
    entropy, applicable = conditional_family_entropy(
        logp, torch.tensor([0, 3]), factors, torch.tensor([3]),
    )
    assert applicable.tolist() == [True]
    entropy.sum().backward()
    assert raw.grad[0] == pytest.approx(0.0, abs=1e-7)
    assert (raw.grad[1] + raw.grad[2]) == pytest.approx(0.0, abs=1e-7)


def test_conditional_family_entropy_ignores_only_singleton_families():
    factors = torch.zeros(2, 3, 15, dtype=torch.long)
    factors[0, :2, 0] = torch.tensor([0, 4])
    factors[1, :, 0] = 1
    entropy, applicable = conditional_family_entropy(
        torch.tensor([-.2, -1.7, -1.0, -1.0, -1.0]),
        torch.tensor([0, 2, 5]), factors, torch.tensor([2, 3]),
    )
    assert applicable.tolist() == [False, True]
    assert entropy[0] == 0
    assert entropy[1] == pytest.approx(1.0)


def test_conditional_family_entropy_is_stable_for_negligible_family_mass():
    # The two calls are equally likely *within their family*, even though the
    # policy gives the family so little total mass that exp(logp) underflows.
    raw = torch.tensor([0.0, -1_000.0, -1_000.0], requires_grad=True)
    factors = torch.zeros(1, 3, 15, dtype=torch.long)
    factors[0, :, 0] = torch.tensor([0, 3, 4])
    entropy, applicable = conditional_family_entropy(
        raw.log_softmax(0), torch.tensor([0, 3]), factors, torch.tensor([3]),
    )
    assert applicable.tolist() == [True]
    assert torch.isfinite(entropy).all()
    assert entropy.item() == pytest.approx(1.0)
    entropy.sum().backward()
    assert torch.isfinite(raw.grad).all()
