import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from audit import _load_initial, _model  # noqa: E402


VALUES = {
    "model": {
        "layers": 1, "d_model": 32, "query_heads": 2, "kv_heads": 1,
        "head_dim": 16, "ffn_dim": 64, "action_memory_layers": 2,
        "action_memory_ffn_dim": 64, "share_all_action_tiles": True,
        "concealed_shape_channels": 4, "concealed_shape_blocks": 1,
        "rank_critic_width": 8,
    },
    "encoding": {"context_tokens": 64},
}


def test_variants_are_behavior_identical_at_initialization():
    baseline = _model(VALUES, "baseline", "cpu", 1)
    state = baseline.state_dict()
    role = _model(VALUES, "role_aware_tiles", "cpu", 1)
    deep = _model(VALUES, "deeper_action_head", "cpu", 1)
    post = _model(VALUES, "post_action_shape", "cpu", 1)
    _load_initial(role, state, "role_aware_tiles")
    _load_initial(deep, state, "deeper_action_head")
    _load_initial(post, state, "post_action_shape")

    factors = torch.zeros(2, 3, 15, dtype=torch.long)
    factors[..., 7] = torch.tensor(((1, 5, 9), (13, 17, 21)))
    with torch.no_grad():
        expected = baseline.canonical_tile_embedding.action_tiles(factors)
        actual = role.canonical_tile_embedding.action_tiles(factors)
    torch.testing.assert_close(actual, expected)
    assert len(deep.action_memory.blocks) == 4
    for block in deep.action_memory.blocks[2:]:
        assert torch.count_nonzero(block.memory_attention.out_proj.weight) == 0
        assert torch.count_nonzero(block.candidate_attention.out_proj.weight) == 0
        assert torch.count_nonzero(block.ffn.down.weight) == 0
    assert torch.count_nonzero(post.post_action_projection.weight) == 0
