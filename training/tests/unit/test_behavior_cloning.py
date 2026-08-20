from types import SimpleNamespace

import numpy as np
import pytest
import torch

from zenith_ppo.bc.data import BCExample
from zenith_ppo.cli.train_bc import (
    _Accumulator,
    _architecture,
    _regularized_policy_loss,
    _run_examples,
)
from zenith_ppo.encoding.packing import EncodedActionSpace
from zenith_ppo.model.factory import build_actor_critic
from zenith_ppo.types import ActionSpaceBinding


def test_regularized_policy_loss_is_legal_set_local_and_finite():
    logits = torch.tensor([1.0, 0.0, -1.0, 2.0, 1.0], requires_grad=True)
    logp = torch.cat((
        logits[:3].log_softmax(0), logits[3:].log_softmax(0)
    ))
    objective, selected_nll = _regularized_policy_loss(
        logp,
        torch.tensor([0, 4]),
        torch.tensor([0, 3, 5]),
        label_smoothing=0.05,
        confidence_penalty_coefficient=0.01,
    )

    assert torch.isfinite(objective)
    assert selected_nll.tolist() == pytest.approx(
        [-logp[0].item(), -logp[4].item()]
    )
    objective.backward()
    assert torch.isfinite(logits.grad).all()


def test_regularized_policy_loss_supports_normalized_family_weights():
    logits = torch.tensor([1.0, 0.0, 2.0, -1.0], requires_grad=True)
    logp = torch.cat((logits[:2].log_softmax(0), logits[2:].log_softmax(0)))
    objective, selected_nll = _regularized_policy_loss(
        logp,
        torch.tensor([0, 3]),
        torch.tensor([0, 2, 4]),
        weights=torch.tensor([1.0, 2.0]),
    )

    assert objective.item() == pytest.approx(
        ((selected_nll[0] + 2 * selected_nll[1]) / 3).item()
    )


def test_bc_accumulator_reports_policy_and_boundary_metrics():
    accumulator = _Accumulator("cpu")
    examples = (
        SimpleNamespace(family="discard"),
        SimpleNamespace(family="call"),
    )
    accumulator.add(
        examples,
        torch.tensor([1.0, 2.0]),
        torch.ones(2),
        torch.tensor([True, False]),
        torch.tensor([3.0, 2.0, 1.0, 0.5]),
    )
    metrics = accumulator.metrics()

    assert metrics["nll"] == pytest.approx(1.5)
    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["rank_boundary"] == pytest.approx({
        "order_nll": 1.5,
        "order_accuracy": 0.5,
        "brier": 0.25,
        "rows": 2,
    })


def test_bc_architecture_is_explicitly_checkpoint_breaking():
    model = SimpleNamespace(
        architecture_id="verified-public-state-value-ppo-v1"
    )
    assert _architecture(model) == "verified-public-state-value-ppo-v1"

    removed = SimpleNamespace(architecture_id="categorical-prospect-rank-v1")
    with pytest.raises(ValueError, match="verified production architecture"):
        _architecture(removed)


def test_bc_step_updates_policy_and_sparse_boundary_rank_critic():
    model = build_actor_critic({
        "architecture": "verified-public-state-value-ppo-v1",
        "layers": 1,
        "d_model": 16,
        "query_heads": 2,
        "kv_heads": 1,
        "head_dim": 8,
        "ffn_dim": 32,
        "context_tokens": 64,
            "action_memory_layers": 1,
            "action_memory_ffn_dim": 32,
            "concealed_shape_channels": 4,
        "concealed_shape_blocks": 1,
        "boundary_critic_width": 16,
        "ground_board_layers": 1,
        "structured_boundary_layers": 1,
    })
    token_factors = np.zeros((16, 10), dtype=np.uint8)
    action_factors = np.zeros((2, 15), dtype=np.uint8)
    action_factors[:, 0] = 1
    action_factors[:, 1] = (1, 2)
    encoded = EncodedActionSpace(
        binding=ActionSpaceBinding(0, 1, 1, 0),
        token_factors=token_factors,
        token_numeric=np.ones((16, 8), dtype=np.float32) * 0.1,
        actor_query_index=15,
        rank_boundary_features=np.zeros(28, dtype=np.float32),
        decision_seat=0,
        action_factors=action_factors,
        action_representatives=(),
        action_members=(),
        native_candidates=(),
    )
    example = BCExample(
        encoded,
        target=0,
        family="discard",
        critic=SimpleNamespace(
            rank_boundary_supervision=True,
            rank_order_target=0,
            hand_outcome_target=0,
            hand_score_delta=0.0,
            terminal_placement=0,
        ),
    )
    optimizer = torch.optim.AdamW((
        {"params": model.actor_parameters(), "lr": 1e-3},
        {"params": model.critic_parameters(), "lr": 2e-3},
    ))
    actor_before = {
        name: parameter.detach().clone()
        for name, parameter in model.actor_named_parameters()
    }
    critic_before = [
        parameter.detach().clone() for parameter in model.critic_parameters()
    ]

    _run_examples(
        model,
        optimizer,
        (example,),
        config={
            "token_budget": 256,
            "label_smoothing": 0.0,
            "confidence_penalty_coefficient": 0.0,
            "boundary_rank_coefficient": 1.0,
            "max_grad_norm": 10.0,
        },
        device="cpu",
        backend="eager",
        use_bf16=False,
        train=True,
        accumulator=_Accumulator("cpu"),
    )

    assert any(
        not torch.equal(actor_before[name], parameter)
        for name, parameter in model.actor_named_parameters()
    )
    assert any(
        not torch.equal(before, parameter)
        for before, parameter in zip(
            critic_before, model.critic_parameters(), strict=True
        )
    )
