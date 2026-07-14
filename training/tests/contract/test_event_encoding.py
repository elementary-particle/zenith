import pytest
from zenith_ppo.encoding.event_cache import EventPrefixCache
from zenith_ppo.encoding.events import encode_event, encode_history
from zenith_ppo.encoding.schema import numeric_features
from zenith_ppo.encoding.state import encode_state
from zenith_ppo.env.history import EventStore


def test_one_token_per_gap_free_event_and_hidden_value_mask():
    rows = [{"episode_generation": 1, "sequence": i, "kind": 4, "actor_seat": 1,
             "visibility_mask": 2, "arg0": 12} for i in range(2)]
    assert len(encode_history(rows, observer=0, generation=1)) == 2
    assert encode_history(rows, observer=0, generation=1)[0].tile_rank == 0
    rows[1]["sequence"] = 3
    with pytest.raises(ValueError, match="gap"): encode_history(rows, observer=0, generation=1)


def _row(sequence, *, visibility=0b1111, tile=12, kind=4, arg1=0):
    return {
        "environment_id": 3,
        "episode_generation": 7,
        "sequence": sequence,
        "kind": kind,
        "actor_seat": 1,
        "visibility_mask": visibility,
        "arg0": tile,
        "arg1": arg1,
    }


def test_factorized_event_prefix_cache_encodes_only_appended_rows():
    calls = []

    def counting_encoder(row, **kwargs):
        from zenith_ppo.encoding.events import encode_event

        calls.append(int(row["sequence"]))
        return encode_event(row, **kwargs)

    store = EventStore(3, 7)
    store.append([_row(0), _row(1)])
    cache = EventPrefixCache(encoder=counting_encoder)

    first = cache.encode(store, observer=0)
    assert calls == [0, 1]
    assert cache.encode(store, observer=0) is first
    assert calls == [0, 1]

    store.append([_row(2)])
    extended = cache.encode(store, observer=0)
    assert calls == [0, 1, 2]
    expected = encode_history(store.rows, observer=0, generation=7)
    assert extended.token_factors == tuple(token.categorical() for token in expected)
    assert extended.token_numeric == tuple(numeric_features(token) for token in expected)
    assert cache.stats.events_encoded == 3
    assert cache.stats.tokens_reused == 4


def test_event_prefix_cache_isolates_observers_and_store_instances():
    store = EventStore(3, 7)
    store.append([_row(0, visibility=0b0010, tile=20)])
    cache = EventPrefixCache()
    hidden = cache.encode(store, observer=0)
    visible = cache.encode(store, observer=1)
    assert hidden.token_factors != visible.token_factors

    replacement = EventStore(3, 7)
    replacement.append([_row(0, visibility=0b1111, tile=40)])
    replaced = cache.encode(replacement, observer=0)
    assert replaced.token_factors != hidden.token_factors

    cache.retain({(9, 9)})
    assert cache.entry_count == 0


def test_tsumo_is_validated_but_omitted_and_event_details_are_compact():
    rows = [
        _row(0, kind=3, visibility=0b0001),
        _row(1, kind=4, arg1=1),
        _row(2, kind=12, tile=25_000, arg1=70_000),
    ]
    tokens = encode_history(rows, observer=0, generation=7)
    assert [token.field for token in tokens] == [4, 12]
    assert tokens[0].flag != 0  # tsumogiri detail
    assert all(0 <= token.flag < 256 for token in tokens)


def test_model_history_starts_at_latest_kyoku_without_mutating_store():
    store = EventStore(3, 7)
    store.append([
        _row(0, kind=1),
        _row(1, kind=2),
        _row(2, kind=4, tile=4),
        _row(3, kind=15),
        _row(4, kind=2),
        _row(5, kind=4, tile=40),
    ])
    expected = encode_history(store.rows, observer=0, generation=7)
    cached = EventPrefixCache().encode(store, observer=0)
    assert [token.field for token in expected] == [2, 4]
    assert cached.token_factors == tuple(token.categorical() for token in expected)
    assert store.next_sequence == 6


def test_semantically_identical_tile_copies_share_event_and_state_factors():
    first = encode_event(_row(0, tile=17), observer=0)
    second = encode_event(_row(0, tile=18), observer=0)
    assert first is not None and second is not None
    assert first.categorical() == second.categorical()
    assert numeric_features(first) == numeric_features(second) == (0.0,) * 8

    counts = [0] * 34
    counts[4] = 1
    state = encode_state(
        {"scores": [25_000] * 4, "priv_concealed_tile_ids": [[17], [], [], []]},
        {"concealed_counts": counts, "flags": 0},
        observer=0,
    )
    concealed = next(token for token in state if token.kind == 4 and token.field == 1)
    assert (first.tile_suit, first.tile_rank, first.tile_red) == (
        concealed.tile_suit,
        concealed.tile_rank,
        concealed.tile_red,
    )
