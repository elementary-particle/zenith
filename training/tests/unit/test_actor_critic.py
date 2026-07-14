import pytest
import torch
from zenith_ppo.encoding.actions import segmented_log_softmax
from zenith_ppo.model.actor_critic import ActorCritic
from zenith_ppo.rollout.collector import _copy_inference_results


def test_finite_shapes_and_actor_causal_isolation():
    config = {"layers": 1, "d_model": 32, "query_heads": 2, "kv_heads": 1,
              "head_dim": 16, "ffn_dim": 64, "context_tokens": 32}
    model = ActorCritic(config)
    token = torch.ones(1, 8, 10, dtype=torch.long); actions = torch.ones(2, 15, dtype=torch.long)
    args = dict(token_factors=token, action_factors=actions,
        actor_query_indices=torch.tensor([3]), action_offsets=[0, 2],
        critic_factors=torch.zeros(1, 0, 10, dtype=torch.long),
        critic_lengths=torch.zeros(1, dtype=torch.long), backend="eager")
    output = model(**args)
    before = output.logits
    assert output.opponent_count_logits.shape == (1, 3, 34, 5)
    assert output.opponent_tenpai_logits.shape == (1, 3)
    token[:, 4:, 2] = 2
    after = model(**args).logits
    assert torch.equal(before, after)


def test_value_and_belief_gradient_boundaries():
    config = {"layers": 1, "critic_layers": 2, "d_model": 32, "query_heads": 2,
              "kv_heads": 1, "head_dim": 16, "ffn_dim": 64, "context_tokens": 32}
    model = ActorCritic(config)
    args = dict(
        token_factors=torch.ones(1, 4, 10, dtype=torch.long),
        action_factors=torch.ones(2, 15, dtype=torch.long),
        actor_query_indices=torch.tensor([3]), action_offsets=[0, 2],
        critic_factors=torch.ones(1, 2, 10, dtype=torch.long),
        critic_lengths=torch.tensor([2]), backend="eager",
    )
    model(**args).values.sum().backward()
    assert all(parameter.grad is None for name, parameter in model.named_parameters()
               if name.startswith(("token_embedding", "backbone", "actor_query",
                                   "action_embedding", "action_key", "opponent_")))
    assert any(parameter.grad is not None for name, parameter in model.named_parameters()
               if name.startswith(("critic_embedding", "critic_decoder", "value")))

    model.zero_grad(set_to_none=True)
    output = model(**args)
    (output.opponent_count_logits.sum() + output.opponent_tenpai_logits.sum()).backward()
    assert any(parameter.grad is not None for name, parameter in model.named_parameters()
               if name.startswith(("token_embedding", "backbone", "opponent_")))
    assert all(parameter.grad is None for name, parameter in model.named_parameters()
               if name.startswith(("critic_embedding", "critic_decoder", "value_query",
                                   "value", "action_embedding", "actor_query", "action_key")))


def test_private_state_cannot_change_policy_logits():
    config = {"layers": 1, "critic_layers": 2, "d_model": 32, "query_heads": 2,
              "kv_heads": 1, "head_dim": 16, "ffn_dim": 64, "context_tokens": 32}
    model = ActorCritic(config).eval()
    common = dict(
        token_factors=torch.ones(1, 4, 10, dtype=torch.long),
        action_factors=torch.ones(2, 15, dtype=torch.long),
        actor_query_indices=torch.tensor([3]), action_offsets=[0, 2],
        critic_lengths=torch.tensor([2]), backend="eager",
    )
    first = model(**common, critic_factors=torch.ones(1, 2, 10, dtype=torch.long))
    second = model(**common, critic_factors=torch.full((1, 2, 10), 2, dtype=torch.long))
    assert torch.equal(first.logits, second.logits)
    assert torch.equal(first.opponent_count_logits, second.opponent_count_logits)
    assert torch.equal(first.opponent_tenpai_logits, second.opponent_tenpai_logits)


def test_vectorized_ragged_normalization_entropy_and_sampling_match_reference():
    config = {"layers": 1, "d_model": 32, "query_heads": 2, "kv_heads": 1,
              "head_dim": 16, "ffn_dim": 64, "context_tokens": 32}
    model = ActorCritic(config)
    logits = torch.tensor([1.0, -2.0, 0.5, 2.0, -1.0, 3.0], requires_grad=True)
    offsets = [0, 2, 5, 6]

    actual = segmented_log_softmax(logits, offsets)
    expected = torch.cat([
        torch.log_softmax(logits[start:end], dim=0)
        for start, end in zip(offsets[:-1], offsets[1:])
    ])
    entropy = model.segmented_entropy(actual, offsets)
    expected_entropy = torch.stack([
        -(expected[start:end].exp() * expected[start:end]).sum()
        for start, end in zip(offsets[:-1], offsets[1:])
    ])
    assert torch.allclose(actual, expected)
    assert torch.allclose(entropy, expected_entropy)

    output = type("Output", (), {
        "log_probabilities": actual,
        "entropy": entropy,
        "values": torch.tensor([4.0, 5.0, 6.0]),
    })()
    deterministic = model.sample(output, offsets, deterministic=True)
    assert deterministic.tolist() == [0, 3, 5]
    first = model.sample(output, offsets, generator=torch.Generator().manual_seed(9))
    second = model.sample(output, offsets, generator=torch.Generator().manual_seed(9))
    assert torch.equal(first, second)
    assert all(start <= value < end for value, start, end in zip(first, offsets[:-1], offsets[1:]))

    actual.sum().backward()
    assert torch.isfinite(logits.grad).all()


def test_production_model_uses_compact_embeddings():
    from zenith_ppo.config import load

    config = load("training/configs/default.toml")
    model_config = dict(config.values["model"])
    model_config["context_tokens"] = config.values["encoding"]["context_tokens"]
    model = ActorCritic(model_config)
    assert sum(parameter.numel() for parameter in model.parameters()) < 8_000_000


def test_rollout_result_copy_converts_a_whole_shard_at_once():
    output = type("Output", (), {
        "log_probabilities": torch.tensor([-0.1, -2.0, -3.0, -0.2, 0.0]),
        "entropy": torch.tensor([0.4, 0.5, 0.0]),
        "values": torch.tensor([1.0, 2.0, 3.0]),
    })()
    groups, logp, entropy, values = _copy_inference_results(
        output, torch.tensor([0, 3, 4]), [0, 2, 4, 5]
    )
    assert groups == [0, 1, 0]
    assert logp == pytest.approx([-0.1, -0.2, 0.0])
    assert entropy == pytest.approx([0.4, 0.5, 0.0])
    assert values == pytest.approx([1.0, 2.0, 3.0])
