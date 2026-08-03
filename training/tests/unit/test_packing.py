from dataclasses import replace
from types import SimpleNamespace

import torch

from zenith_ppo.encoding.event_cache import EventPrefixCache
from zenith_ppo.encoding.packing import (
    encode_native_batch,
    mean_token_length,
    model_batch,
    pack,
    padding_fraction,
)


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
            rank_boundary_features=(0.0,) * 28,
            decision_seat=0,
        ),
        SimpleNamespace(
            token_factors=((4,) * 10,),
            token_numeric=((2.5,) * 8,),
            action_factors=((5,) * 15, (6,) * 15),
            actor_query_index=0,
            rank_boundary_features=(1.0,) * 28,
            decision_seat=1,
        ),
    )
    batch = model_batch(rows)
    assert batch["token_factors"].shape == (2, 2, 10)
    assert batch["token_factors"].dtype == torch.int32
    assert batch["token_factors"][1, 1].eq(0).all()
    assert batch["token_numeric"].dtype == torch.float32
    assert batch["action_factors"].shape == (2, 2, 15)
    assert batch["action_factors"].dtype == torch.int32
    assert batch["action_lengths"].tolist() == [1, 2]
    assert batch["action_offsets"].tolist() == [0, 1, 3]
    assert batch["lengths"].tolist() == [2, 1]
    assert batch["decision_seats"].tolist() == [0, 1]
    assert batch["rank_boundary_features"].shape == (2, 28)


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
        assert tuple(replace(row, native_candidates=()) for row in cached) == tuple(
            replace(row, native_candidates=()) for row in uncached
        )
        encoded_events = cache.stats.events_encoded
        repeated = encode_native_batch(batch, adapter.histories, event_cache=cache)
        assert tuple(replace(row, native_candidates=()) for row in repeated) == tuple(
            replace(row, native_candidates=()) for row in uncached
        )
        assert cache.stats.events_encoded == encoded_events
    finally:
        env.close()


def test_native_encoding_contains_only_public_model_tensors():
    import riichi

    from zenith_ppo.env.adapter import EnvAdapter
    env = riichi.Env(1, master_seed=19, num_threads=1, privileged=True)
    try:
        adapter = EnvAdapter(env)
        batch = adapter.reset([0])
        encoded = encode_native_batch(batch, adapter.histories)
        assert encoded
        assert all(any(
            int(factor[1]) == 3 and int(factor[2]) == 9
            for factor in row.token_factors
        ) for row in encoded)
        assert all(not hasattr(row, "oracle_factors") for row in encoded)
        inputs = model_batch(encoded)
        assert not any("oracle" in name for name in inputs)
    finally:
        env.close()
