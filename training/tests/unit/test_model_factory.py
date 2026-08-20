import pytest

from zenith_ppo.model.factory import (
    build_actor_critic,
    checkpoint_architecture,
)
from zenith_ppo.model.structured_boundary import VerifiedActorCritic


def _config():
    return {
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
    }


def test_factory_builds_only_verified_production_model():
    model = build_actor_critic(_config())

    assert isinstance(model, VerifiedActorCritic)
    assert checkpoint_architecture(model) == \
        "verified-public-state-value-ppo-v1"


def test_factory_rejects_experimental_architectures():
    config = _config() | {"architecture": "categorical-prospect-rank-v1"}

    with pytest.raises(ValueError, match="production requires"):
        build_actor_critic(config)
