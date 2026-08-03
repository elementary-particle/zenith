import torch

from zenith_ppo.model.actor_critic import ActorCritic


def _config():
    return {
        "layers": 1,
        "d_model": 32,
        "query_heads": 2,
        "kv_heads": 1,
        "head_dim": 16,
        "ffn_dim": 64,
        "context_tokens": 64,
        "action_memory_layers": 2,
        "action_memory_ffn_dim": 64,
        "share_all_action_tiles": True,
        "concealed_shape_channels": 8,
        "concealed_shape_blocks": 1,
        "rank_critic_width": 16,
    }


def _inputs():
    torch.manual_seed(4)
    actions = torch.zeros(3, 5, 15, dtype=torch.long)
    actions[..., 0] = torch.randint(0, 11, (3, 5))
    actions[..., 1] = torch.randint(0, 34, (3, 5))
    return {
        "token_factors": torch.zeros(3, 8, 10, dtype=torch.long),
        "token_numeric": torch.zeros(3, 8, 8),
        "lengths": torch.tensor([8, 7, 6]),
        "actor_query_indices": torch.tensor([7, 6, 5]),
        "action_factors": actions,
        "action_lengths": torch.tensor([5, 4, 3]),
        "action_offsets": torch.tensor([0, 5, 9, 12]),
        "decision_seats": torch.tensor([0, 1, 3]),
        "rank_boundary_features": torch.zeros(3, 28),
        "backend": "eager",
    }


def test_policy_and_boundary_rank_shapes_are_normalized():
    model = ActorCritic(_config())
    output = model(**_inputs())

    assert output.logits.shape == (12,)
    assert output.rank_probabilities.shape == (3, 4)
    assert torch.allclose(output.rank_probabilities.sum(-1), torch.ones(3))
    for start, end in ((0, 5), (5, 9), (9, 12)):
        assert torch.allclose(
            output.log_probabilities[start:end].exp().sum(), torch.tensor(1.0)
        )


def test_candidate_order_is_permutation_equivariant():
    model = ActorCritic(_config()).eval()
    inputs = _inputs()
    baseline = model.forward_actor(**inputs).logits[:5]
    permutation = torch.tensor([2, 4, 0, 3, 1])
    inputs["action_factors"] = inputs["action_factors"].clone()
    inputs["action_factors"][0] = inputs["action_factors"][0, permutation]
    permuted = model.forward_actor(**inputs).logits[:5]

    assert torch.allclose(permuted, baseline[permutation], atol=1e-5)


def test_rollout_actor_can_skip_unused_entropy():
    model = ActorCritic(_config()).eval()
    inputs = _inputs()

    full = model.forward_actor(**inputs)
    rollout = model.forward_actor(**inputs, compute_entropy=False)

    assert rollout.entropy is None
    assert torch.equal(rollout.log_probabilities, full.log_probabilities)


def test_actor_and_boundary_critic_parameters_are_disjoint_and_complete():
    model = ActorCritic(_config())
    actor = set(map(id, model.actor_parameters()))
    critic = set(map(id, model.critic_parameters()))

    assert actor
    assert critic
    assert not actor & critic
    assert len(actor | critic) == sum(1 for _ in model.parameters())


def test_boundary_critic_backward_does_not_touch_actor():
    model = ActorCritic(_config())
    inputs = _inputs()
    model.forward_critic(**inputs).rank_order_logits.square().mean().backward()

    assert all(parameter.grad is None for parameter in model.actor_parameters())
    assert any(parameter.grad is not None for parameter in model.critic_parameters())
