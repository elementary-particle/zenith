import torch

import riichi
from zenith_ppo.model.actor_critic import ActorCritic
from zenith_ppo.rollout.native import NativeInferenceRunner


def _model():
    return ActorCritic({
        "layers": 1,
        "d_model": 16,
        "query_heads": 2,
        "kv_heads": 1,
        "head_dim": 8,
        "ffn_dim": 32,
        "context_tokens": 256,
        "action_memory_layers": 1,
        "action_memory_ffn_dim": 32,
        "share_all_action_tiles": True,
        "concealed_shape_channels": 4,
        "concealed_shape_blocks": 1,
        "rank_critic_width": 16,
    })


def test_cpu_native_inference_samples_and_submits_one_request():
    torch.manual_seed(3)
    engine = riichi.RolloutEngine(
        2,
        master_seed=5,
        num_threads=1,
        context_tokens=256,
        token_budget=4096,
    )
    matches = engine.reset_chunk(2)
    engine.register_lineups(matches, [(0, 0, 0, 0)] * 2, [15, 15])
    request = engine.next_request()
    runner = NativeInferenceRunner(
        {0: _model()},
        backend="eager",
        generator=torch.Generator().manual_seed(19),
    )

    selected, old_logp = runner.infer(request)

    assert len(selected) == request.row_count
    assert all(
        0 <= group < int(length)
        for group, length in zip(selected, request.action_lengths, strict=True)
    )
    assert all(value <= 0 for value in old_logp)
    engine.submit(request.request_id, selected, old_logp)
    stats = runner.stats(engine)
    assert stats.requests == 1
    assert stats.rows == request.row_count
    assert stats.padded_tokens >= stats.useful_tokens
    assert stats.compile_fallbacks == 0
