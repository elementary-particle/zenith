import torch

from zenith_ppo.model.factory import build_actor_critic
from zenith_ppo.model.structured_boundary import StructuredMatchBoundaryCritic
from zenith_ppo.model.verified_components import tile_player_relations


def _config():
    return {
        "architecture": "verified-public-state-value-ppo-v1",
        "layers": 1,
        "ground_board_layers": 1,
        "structured_boundary_layers": 1,
        "d_model": 32,
        "query_heads": 2,
        "kv_heads": 1,
        "head_dim": 16,
        "ffn_dim": 64,
        "context_tokens": 64,
        "action_memory_layers": 2,
        "action_memory_ffn_dim": 64,
        "concealed_shape_channels": 8,
        "concealed_shape_blocks": 1,
        "boundary_critic_width": 16,
    }


def _inputs():
    factors = torch.zeros(2, 18, 10, dtype=torch.long)
    factors[:, :10, 0] = 1
    factors[:, :4, 1:4] = torch.tensor((2, 1, 1))
    factors[0, :4, 3] = torch.tensor((1, 2, 3, 4))
    factors[1, :4, 3] = torch.tensor((3, 4, 1, 2))
    factors[:, 4:8, 1] = 3
    factors[:, 4:8, 2] = torch.tensor((1, 2, 3, 4))
    factors[:, 8:10, 1] = 3
    factors[:, 8:10, 2] = torch.tensor((6, 7))
    factors[:, 10, 0], factors[:, 10, 1] = 2, 8
    factors[:, 11:14, 0] = 3
    factors[:, 11:14, 1] = 1
    factors[:, 11:14, 2] = torch.tensor((2, 4, 11))
    factors[:, 14:16, 0] = 3
    factors[:, 14:16, 1] = 4
    factors[:, 14, 2] = 5
    factors[:, 14, 4:6] = torch.tensor((1, 5))
    factors[:, 14, 7] = 2
    factors[:, 15, 2] = 8
    factors[:, 16, 0], factors[:, 16, 1] = 4, 8
    factors[:, 17, 0], factors[:, 17, 1] = 5, 8
    actions = torch.zeros(2, 5, 15, dtype=torch.long)
    actions[..., 0] = 1
    actions[..., 1] = torch.tensor((0, 4, 8, 12, 27))
    boundary = torch.zeros(2, 28)
    boundary[:, :4] = torch.tensor((1.0, 0.8, 1.2, 1.0))
    boundary[:, 4] = 1
    boundary[:, 9] = 1
    return {
        "token_factors": factors,
        "token_numeric": torch.zeros(2, 18, 8),
        "lengths": torch.tensor((18, 18)),
        "actor_query_indices": torch.tensor((17, 17)),
        "action_factors": actions,
        "action_lengths": torch.tensor((5, 4)),
        "action_offsets": torch.tensor((0, 5, 9)),
        "rank_boundary_features": boundary,
        "decision_seats": torch.tensor((0, 2)),
        "backend": "eager",
    }


def _permute_suits(inputs, permutation):
    result = dict(inputs)
    lookup = torch.tensor((0, *(value + 1 for value in permutation), 4))
    tokens = inputs["token_factors"].clone()
    suit = tokens[..., 4]
    tokens[..., 4] = torch.where(
        suit.ge(1) & suit.le(3), lookup[suit.clamp(0, 4)], suit,
    )
    actions = inputs["action_factors"].clone()
    suit = actions[..., 3]
    actions[..., 3] = torch.where(
        suit.ge(1) & suit.le(3), lookup[suit.clamp(0, 4)], suit,
    )
    primary = actions[..., 1]
    valid = primary.ge(0) & primary.lt(27)
    suit_lookup = torch.tensor(permutation)
    changed = suit_lookup[primary.clamp(0, 26).div(
        9, rounding_mode="floor",
    )] * 9 + primary.clamp(0, 26).remainder(9)
    actions[..., 1] = torch.where(valid, changed, primary)
    semantic = actions[..., 7:11]
    tile = semantic.div(4, rounding_mode="floor")
    valid = semantic.gt(0) & tile.lt(27)
    changed = suit_lookup[tile.clamp(0, 26).div(
        9, rounding_mode="floor",
    )] * 9 + tile.clamp(0, 26).remainder(9)
    actions[..., 7:11] = torch.where(
        valid, changed * 4 + semantic.remainder(4), semantic,
    )
    result["token_factors"] = tokens
    result["action_factors"] = actions
    return result


def test_verified_policy_outputs_policy_prospects_and_state_value():
    model = build_actor_critic(_config()).eval()
    output = model.forward_actor(**_inputs())

    assert output.logits.shape == (9,)
    assert output.hand_outcome_logits.shape == (9, 4)
    assert output.score_delta_logits.shape == (9, 11)
    assert output.placement_logits.shape == (9, 4)
    assert output.state_values.shape == (2,)
    torch.testing.assert_close(
        output.log_probabilities[:5].exp().sum(), torch.tensor(1.0),
    )
    torch.testing.assert_close(
        output.log_probabilities[5:].exp().sum(), torch.tensor(1.0),
    )


def test_lightweight_policy_forward_skips_auxiliary_and_value_heads_exactly():
    model = build_actor_critic(_config()).eval()
    inputs = _inputs()
    expected = model.forward_actor(**inputs)

    actual = model.forward_actor(
        **inputs,
        compute_entropy=False,
        compute_value=False,
        compute_auxiliary=False,
    )

    torch.testing.assert_close(actual.logits, expected.logits, rtol=0, atol=0)
    torch.testing.assert_close(
        actual.log_probabilities, expected.log_probabilities, rtol=0, atol=0,
    )
    assert actual.entropy is None
    assert actual.state_values is None
    assert actual.hand_outcome_logits is None
    assert actual.score_delta_logits is None
    assert actual.placement_logits is None


def test_cached_decision_features_match_direct_critic_forward():
    model = build_actor_critic(_config()).eval()
    inputs = _inputs()
    with torch.no_grad():
        output = model.forward_actor(
            **inputs,
            compute_entropy=False,
            compute_value=False,
            compute_auxiliary=False,
        )
        features = model.decision_features(output, inputs["action_lengths"])
        boundary = model.forward_critic(**inputs).rank_values

    cached = model.forward_decision_head(features, boundary)
    direct = model.forward_decision_critic(**inputs)

    torch.testing.assert_close(cached, direct, rtol=0, atol=0)


def test_verified_policy_is_candidate_permutation_equivariant():
    model = build_actor_critic(_config()).eval()
    inputs = _inputs()
    baseline = model.forward_actor(**inputs).logits[:5]
    permutation = torch.tensor((2, 4, 0, 3, 1))
    inputs["action_factors"] = inputs["action_factors"].clone()
    inputs["action_factors"][0] = inputs["action_factors"][0, permutation]

    changed = model.forward_actor(**inputs).logits[:5]

    torch.testing.assert_close(
        changed, baseline[permutation], atol=1e-5, rtol=1e-5,
    )


def test_verified_policy_is_whole_suit_equivariant():
    model = build_actor_critic(_config()).eval()
    inputs = _inputs()
    baseline = model.forward_actor(**inputs).logits
    changed = model.forward_actor(
        **_permute_suits(inputs, (1, 2, 0)),
    ).logits

    torch.testing.assert_close(changed, baseline, atol=1e-5, rtol=1e-5)


def test_tile_relations_are_whole_suit_equivariant():
    relations = tile_player_relations()
    permutation = torch.tensor((*range(9, 18), *range(0, 9), *range(18, 34)))
    torch.testing.assert_close(
        relations,
        relations.index_select(0, permutation).index_select(1, permutation),
    )


def test_zero_initialized_state_residual_equals_boundary_value():
    model = build_actor_critic(_config()).eval()
    inputs = _inputs()
    actor = model.forward_actor(**inputs)
    boundary = model.forward_critic(**inputs)

    torch.testing.assert_close(actor.state_values, boundary.rank_values)
    assert set(map(id, model.actor_parameters())).isdisjoint(
        set(map(id, model.critic_parameters()))
    )


def test_verified_bc_checkpoint_migration_preserves_policy_exactly():
    torch.manual_seed(11)
    source = build_actor_critic(_config()).eval()
    legacy = {
        name: value for name, value in source.state_dict().items()
        if not name.startswith("decision_value.")
    }
    torch.manual_seed(17)
    restored = build_actor_critic(_config()).eval()

    restored.load_state_dict(legacy)

    expected = source.forward_actor(**_inputs(), compute_value=False)
    actual = restored.forward_actor(**_inputs(), compute_value=False)
    torch.testing.assert_close(actual.logits, expected.logits, rtol=0, atol=0)


def test_structured_boundary_is_absolute_dealer_invariant():
    critic = StructuredMatchBoundaryCritic(16, layers=1).eval()
    features = _inputs()["rank_boundary_features"]
    seats = _inputs()["decision_seats"]
    expected = critic(features, seats).rank_order_logits
    for dealer in range(4):
        changed = features.clone()
        changed[:, 4:8] = 0
        changed[:, 4 + dealer] = 1
        torch.testing.assert_close(
            critic(changed, seats).rank_order_logits,
            expected,
            rtol=0,
            atol=0,
        )
