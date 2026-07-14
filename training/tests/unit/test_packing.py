from dataclasses import replace
from types import SimpleNamespace

import torch

from zenith_ppo.encoding.event_cache import EventPrefixCache
from zenith_ppo.encoding.packing import (
    encode_native_batch,
    mean_token_length,
    model_batch,
    opponent_count_targets,
    opponent_tenpai_targets,
    pack,
    padding_fraction,
)


def test_opponent_targets_rotate_cover_counts_and_use_open_melds():
    counts = [[0] * 34 for _ in range(4)]
    for seat, count in enumerate((1, 2, 3, 4)):
        counts[seat][seat] = count
    counts[3][:4] = [1, 2, 3, 4]
    targets = opponent_count_targets(counts, observer=2)
    assert targets[:, [3, 0, 1]].diagonal().tolist() == [4, 1, 2]
    assert set(targets.reshape(-1)) >= {0, 1, 2, 3, 4}

    closed = [0] * 34
    for tile in (0, 1, 2, 9, 10, 11, 18, 19, 20, 27, 27, 27, 28):
        closed[tile] += 1
    opened = [0] * 34
    for tile in (0, 1, 2, 9, 10, 11, 18, 19, 20, 28):
        opened[tile] += 1
    labels = opponent_tenpai_targets([closed, opened, opened], [0, 1, 1])
    assert labels.tolist() == [1.0, 1.0, 1.0]


def test_token_budget_packing_covers_each_decision_once():
    packed = pack([3, 8, 2, 5], 12)
    assert sorted(index for batch in packed.batches for index in batch) == [0, 1, 2, 3]
    assert packed.padded_tokens >= packed.total_tokens
    assert mean_token_length(packed, 4) == 4.5
    assert padding_fraction(packed) == (packed.padded_tokens - 18) / packed.padded_tokens
    assert 0.0 <= padding_fraction(packed) <= 1.0


def test_packing_can_enforce_a_padding_fraction_limit():
    packed = pack([10, 1], 100, max_padding_fraction=0.10)
    assert packed.batches == ((0,), (1,))
    assert padding_fraction(packed) <= 0.10


def test_model_batch_bulk_materialization_preserves_ragged_rows():
    rows = (
        SimpleNamespace(
            token_factors=((1,) * 10, (2,) * 10),
            token_numeric=((0.5,) * 8, (1.5,) * 8),
            action_factors=((3,) * 15,),
            actor_query_index=0,
            value_query_index=1,
        ),
        SimpleNamespace(
            token_factors=((4,) * 10,),
            token_numeric=((2.5,) * 8,),
            action_factors=((5,) * 15, (6,) * 15),
            actor_query_index=0,
            value_query_index=0,
        ),
    )
    batch = model_batch(rows)
    assert batch["token_factors"].shape == (2, 2, 10)
    assert batch["token_factors"][1, 1].eq(0).all()
    assert batch["token_numeric"].dtype == torch.float32
    assert batch["action_factors"].shape == (3, 15)
    assert batch["action_offsets"].tolist() == [0, 1, 3]
    assert batch["lengths"].tolist() == [2, 1]


def test_cached_and_uncached_native_encoding_are_identical():
    import riichi

    from zenith_ppo.env.adapter import EnvAdapter

    env = riichi.Env(2, master_seed=17, num_threads=1, privileged=True)
    try:
        adapter = EnvAdapter(env)
        batch = adapter.reset([0, 1])
        uncached = encode_native_batch(batch, adapter.histories)
        cache = EventPrefixCache()
        cached = encode_native_batch(batch, adapter.histories, event_cache=cache)
        assert tuple(replace(row, native_actions=()) for row in cached) == tuple(
            replace(row, native_actions=()) for row in uncached
        )
        encoded_events = cache.stats.events_encoded
        repeated = encode_native_batch(batch, adapter.histories, event_cache=cache)
        assert tuple(replace(row, native_actions=()) for row in repeated) == tuple(
            replace(row, native_actions=()) for row in uncached
        )
        assert cache.stats.events_encoded == encoded_events
    finally:
        env.close()


def test_privileged_critic_is_separate_from_invariant_ordinary_actor():
    import riichi

    from zenith_ppo.env.adapter import EnvAdapter
    from zenith_ppo.encoding.schema import Segment

    env = riichi.Env(1, master_seed=19, num_threads=1, privileged=True)
    try:
        adapter = EnvAdapter(env)
        batch = adapter.reset([0])
        ordinary = encode_native_batch(batch, adapter.histories, critic_mode="privileged")
        assert ordinary
        assert all(
            all(factor[0] != Segment.CRITIC_PRIVATE for factor in row.token_factors)
            for row in ordinary
        )
        assert all(
            row.critic_factors.size and
            all(factor[0] == Segment.CRITIC_PRIVATE for factor in row.critic_factors)
            for row in ordinary
        )
        public_only = encode_native_batch(batch, adapter.histories, critic_mode="ordinary")
        assert all((left.token_factors == right.token_factors).all()
                   for left, right in zip(ordinary, public_only, strict=True))
        assert all(not row.critic_factors.size for row in public_only)
    finally:
        env.close()
