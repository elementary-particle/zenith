from types import SimpleNamespace

import numpy as np
import torch

from experiments.gae_falls_short.privileged_critic import (
    PRIVILEGED_STATE_FEATURES,
    PrivilegedActionQ,
    encode_privileged_snapshot,
)


def _state():
    hidden = SimpleNamespace(
        concealed_counts=tuple(
            tuple(1 if tile == seat else 0 for tile in range(34))
            for seat in range(4)
        ),
        live_wall_counts=tuple(2 for _ in range(34)),
        wall=tuple(index for index in range(136)),
        wall_indices=(10, 70, 124, 2),
        ura_indicators=(8,),
        current_seat=3,
    )
    return SimpleNamespace(
        hidden=hidden,
        dealer=2,
        rivers=(),
        melds=(),
        dora_indicators=(4,),
        seat_flags=(1, 2, 4, 8),
        eligible_mask=0b1010,
        phase=3,
        live_wall_remaining=60,
        scores=(25000, 25000, 25000, 25000),
        round_wind=0,
        hand_number=0,
        honba=0,
        riichi_deposits=0,
    )


def test_privileged_snapshot_contract_and_dealer_rotation():
    result = encode_privileged_snapshot(_state())
    assert result.tile_features.shape == (34, 15)
    assert result.wall_types.shape == (136,)
    assert result.wall_status.shape == (136,)
    assert result.state_features.shape == (PRIVILEGED_STATE_FEATURES,)
    # Dealer 2 is relative seat zero.
    assert result.tile_features[2, 0] == 0.25
    assert result.wall_status[10] == 1
    assert result.wall_status[122] == 2
    assert result.wall_status[130] == 3


def test_privileged_q_returns_one_bounded_value_per_legal_action():
    model = PrivilegedActionQ(width=32, heads=4, layers=1)
    output = model(
        torch.zeros(2, 34, 15),
        torch.zeros(2, 136, dtype=torch.long),
        torch.zeros(2, 136, dtype=torch.long),
        torch.zeros(2, PRIVILEGED_STATE_FEATURES),
        torch.tensor([0, 3]),
        torch.zeros(2, 3, 15, dtype=torch.int32),
        torch.tensor([3, 1]),
    )
    assert output.shape == (2, 3)
    assert torch.isfinite(output).all()
    assert bool((output.abs() <= 1).all())
    assert np.allclose(output[1, 1:].detach().numpy(), 0.0)


def test_privileged_q_uses_detached_actor_hidden_states():
    model = PrivilegedActionQ(
        width=32, heads=4, layers=1, actor_hidden_width=24
    )
    actor_states = torch.randn(2, 24, requires_grad=True)
    actor_action_states = torch.randn(2, 3, 24, requires_grad=True)
    output = model(
        torch.zeros(2, 34, 15),
        torch.zeros(2, 136, dtype=torch.long),
        torch.zeros(2, 136, dtype=torch.long),
        torch.zeros(2, PRIVILEGED_STATE_FEATURES),
        torch.tensor([0, 3]),
        torch.zeros(2, 3, 15, dtype=torch.int32),
        torch.tensor([3, 1]),
        actor_states=actor_states,
        actor_action_states=actor_action_states,
    )
    output.sum().backward()
    assert output.shape == (2, 3)
    assert actor_states.grad is None
    assert actor_action_states.grad is None
    assert any(parameter.grad is not None for parameter in model.parameters())
