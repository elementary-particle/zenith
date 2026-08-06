import pytest
from zenith_ppo.encoding.packing import pack
from zenith_ppo.env.history import EventStore
from zenith_ppo.env.projection import project_event, validate_event_payload


def test_context_and_event_gap_fail_actionably():
    with pytest.raises(ValueError):
        pack([9], 8)
    store = EventStore(0, 1)
    with pytest.raises(ValueError, match="expected 0"):
        store.append([{"environment_id": 0, "episode_generation": 1, "sequence": 1}])


def test_empty_model_batch_hidden_payload_and_closed_env_fail_cleanly():
    import riichi
    from zenith_ppo.encoding.packing import model_batch

    with pytest.raises(ValueError, match="empty"):
        model_batch(())
    hidden = project_event({
        "visibility_mask": 0b0010,
        "args": (135, 2, 3, 4),
        "payload": b"secret",
    }, observer=0)
    assert hidden["args"] == (0, 0, 0, 0) and hidden["payload"] == b""
    env = riichi.Env(1, master_seed=1, num_threads=1)
    env.close()
    with pytest.raises(Exception, match="closed"):
        env.inspect([0])


def test_unknown_and_truncated_event_payload_versions_are_rejected():
    with pytest.raises(ValueError, match="version"):
        validate_event_payload({"kind": 2, "payload": bytes([9]) + bytes(23)})
    truncated = bytearray(bytes([2]) + bytes(23))
    truncated[20] = 1
    with pytest.raises(ValueError, match="length"):
        validate_event_payload({"kind": 2, "payload": truncated})
    with pytest.raises(ValueError, match="end_game"):
        validate_event_payload({"kind": 16, "payload": bytes(19)})


def test_native_engine_snapshot_replays_the_next_match_exactly():
    import riichi

    def engine(seed):
        return riichi.RolloutEngine(
            1, master_seed=seed, num_threads=1,
            context_tokens=2048, token_budget=4096,
        )

    source = engine(73)
    matches = source.reset_chunk(1)
    source.register_lineups(matches, [(9, 9, 9, 9)], [0], bot_policy_slots=[9])
    source.take_chunk()
    snapshots = source.snapshot()
    replay = engine(73)
    replay.restore(snapshots)

    chunks = []
    for current in (source, replay):
        matches = current.reset_chunk(1)
        current.register_lineups(
            matches, [(9, 9, 9, 9)], [0], bot_policy_slots=[9]
        )
        chunks.append(current.take_chunk().as_numpy())
    for name in chunks[0]:
        if name == "row_ids":
            continue
        import numpy as np
        np.testing.assert_array_equal(chunks[0][name], chunks[1][name])
