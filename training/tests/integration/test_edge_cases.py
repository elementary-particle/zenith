import json
from pathlib import Path

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


def test_native_stall_diagnostic_contains_a_replayable_snapshot(tmp_path):
    import riichi
    from zenith_ppo.env.adapter import EnvAdapter
    from zenith_ppo.rollout.collector import Collector

    source = riichi.Env(1, master_seed=73, num_threads=1, privileged=True)
    adapter = EnvAdapter(source)
    batch = adapter.reset([0])
    path = Collector(
        adapter, None, None, diagnostic_dir=tmp_path
    )._write_stall_diagnostic(batch, {(0, 1)}, ())
    payload = json.loads(Path(path).read_text(encoding="utf-8"))

    assert payload["format"] == "zenith-native-env-stall-v1"
    assert payload["master_seed"] == 73
    assert payload["states"][0]["environment_id"] == 0
    assert payload["recent_events"]["0:1"]
    snapshot = bytes.fromhex(payload["snapshots_hex"]["0"])
    replay = riichi.Env(1, master_seed=0, num_threads=1, privileged=True)
    restored = replay.restore({0: snapshot})
    assert restored.states[0].episode_generation == 1

    source.close()
    replay.close()
