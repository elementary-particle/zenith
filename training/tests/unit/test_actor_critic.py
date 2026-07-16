import pytest
import torch

from zenith_ppo.encoding.actions import segmented_log_softmax
from zenith_ppo.model.actor_critic import ActorCritic
from zenith_ppo.rollout.collector import _copy_inference_results


def _config():
    return {"layers": 1, "critic_layers": 2, "d_model": 32,
            "query_heads": 2, "kv_heads": 1, "head_dim": 16,
            "ffn_dim": 64, "context_tokens": 64}


def _inputs():
    torch.manual_seed(4)
    return dict(
        token_factors=torch.randint(0, 2, (3, 8, 10)),
        token_numeric=torch.zeros(3, 8, 8),
        lengths=torch.tensor([8, 7, 6]),
        actor_query_indices=torch.tensor([7, 6, 5]),
        action_factors=torch.randint(0, 2, (3, 5, 15)),
        action_lengths=torch.tensor([4, 2, 3]),
        action_offsets=torch.tensor([0, 4, 6, 9]),
        oracle_factors=torch.randint(0, 2, (2, 12, 10)),
        oracle_numeric=torch.zeros(2, 12, 8),
        oracle_lengths=torch.tensor([12, 10]),
        decision_oracle_indices=torch.tensor([0, 0, 1]),
        decision_seats=torch.tensor([0, 1, 3]),
        backend="eager",
    )


def test_finite_shapes_probabilities_and_causal_actor_isolation():
    model = ActorCritic(_config()).eval()
    args = _inputs()
    output = model(**args)
    assert output.logits.shape == (9,)
    assert output.score_values.shape == (3,)
    assert output.rank_logits.shape == (3, 4)
    assert torch.allclose(output.rank_probabilities.sum(-1), torch.ones(3))
    assert output.opponent_count_logits.shape == (3, 3, 34, 5)
    assert output.opponent_tenpai_logits.shape == (3, 3)

    suffix_args = _inputs()
    suffix_args["actor_query_indices"] = torch.tensor([4, 4, 4])
    before = model.forward_actor(**suffix_args)
    suffix_args["token_factors"][:, 5:] = 1 - suffix_args["token_factors"][:, 5:]
    after = model.forward_actor(**suffix_args)
    assert torch.equal(before.logits, after.logits)
    assert torch.equal(before.opponent_count_logits, after.opponent_count_logits)


def test_action_permutation_equivariance_padding_isolation_and_sampling():
    model = ActorCritic(_config()).eval()
    args = _inputs()
    args = {key: value[:1] if torch.is_tensor(value) and value.ndim and value.shape[0] == 3
            else value for key, value in args.items()}
    args.update(action_lengths=torch.tensor([4]), action_offsets=torch.tensor([0, 4]),
                decision_oracle_indices=torch.tensor([0]), decision_seats=torch.tensor([0]))
    baseline = model.forward_actor(**args)
    permutation = torch.tensor([2, 0, 3, 1])
    changed = dict(args)
    changed["action_factors"] = args["action_factors"].clone()
    changed["action_factors"][0, :4] = args["action_factors"][0, permutation]
    changed["action_factors"][0, 4] = 7
    permuted = model.forward_actor(**changed)
    assert torch.allclose(permuted.logits, baseline.logits[permutation], atol=1e-6)

    first = model.sample(
        baseline, args["action_offsets"], generator=torch.Generator().manual_seed(9)
    )
    second = model.sample(
        baseline, args["action_offsets"], generator=torch.Generator().manual_seed(9)
    )
    assert torch.equal(first, second)


def test_candidate_logits_respond_to_public_state_and_competing_actions():
    model = ActorCritic(_config()).eval()
    args = _inputs()
    baseline = model.forward_actor(**args).logits
    public = dict(args, token_factors=args["token_factors"].clone())
    public["token_factors"][:, 1, 2] += 1
    assert not torch.equal(model.forward_actor(**public).logits, baseline)
    competition = dict(args, action_factors=args["action_factors"].clone())
    competition["action_factors"][0, 3, 1] += 1
    changed = model.forward_actor(**competition).logits
    assert changed[0] != baseline[0]


def test_oracle_never_changes_actor_or_belief_and_has_disjoint_gradients():
    model = ActorCritic(_config()).eval()
    args = _inputs()
    first_actor = model.forward_actor(**args)
    first_critic = model.forward_critic(**args)
    changed = dict(args, oracle_factors=1 - args["oracle_factors"])
    second_actor = model.forward_actor(**changed)
    second_critic = model.forward_critic(**changed)
    assert torch.equal(first_actor.logits, second_actor.logits)
    assert torch.equal(
        first_actor.opponent_count_logits, second_actor.opponent_count_logits
    )
    assert not torch.equal(first_critic.score_values, second_critic.score_values)

    (first_critic.score_values.sum() + first_critic.rank_logits.sum()).backward()
    assert any(parameter.grad is not None for parameter in model.critic_parameters())
    assert all(parameter.grad is None for parameter in model.actor_parameters())
    model.zero_grad(set_to_none=True)
    actor = model.forward_actor(**args)
    (actor.logits.sum() + actor.opponent_count_logits.sum()).backward()
    assert any(parameter.grad is not None for parameter in model.actor_parameters())
    assert all(parameter.grad is None for parameter in model.critic_parameters())


def test_factorized_belief_uses_shared_tile_embeddings_and_three_queries():
    model = ActorCritic(_config())
    args = _inputs()
    output = model.forward_actor(**args)
    targets = torch.randint(0, 5, (3, 3, 34))
    torch.nn.functional.cross_entropy(
        output.opponent_count_logits.reshape(-1, 5), targets.reshape(-1)
    ).backward()
    assert model.opponent_queries.shape == (3, 32)
    assert model.belief_tile_embedding.weight.shape == (34, 32)
    assert torch.count_nonzero(model.belief_tile_embedding.weight.grad)


def test_segmented_normalization_and_result_copy():
    logits = torch.tensor([1.0, -2.0, 0.5, 2.0, -1.0, 3.0], requires_grad=True)
    offsets = [0, 2, 5, 6]
    actual = segmented_log_softmax(logits, offsets)
    expected = torch.cat([
        torch.log_softmax(logits[start:end], 0)
        for start, end in zip(offsets[:-1], offsets[1:])
    ])
    assert torch.allclose(actual, expected)
    output = type("Output", (), {
        "log_probabilities": actual,
        "entropy": torch.tensor([.4, .5, 0.]),
        "score_values": torch.tensor([1., 2., 3.]),
        "rank_values": torch.tensor([-1., -2., -3.]),
    })()
    copied = _copy_inference_results(output, torch.tensor([0, 3, 5]), offsets)
    assert copied[0] == [0, 1, 0]
    assert copied[3] == [1., 2., 3.]


def test_production_model_keeps_actor_and_oracle_compact():
    from zenith_ppo.config import load
    config = load("training/configs/default.toml")
    model_config = dict(config.values["model"])
    model_config["context_tokens"] = config.values["encoding"]["context_tokens"]
    model = ActorCritic(model_config)
    assert sum(parameter.numel() for parameter in model.parameters()) < 16_000_000
