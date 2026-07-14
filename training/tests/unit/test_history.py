from zenith_ppo.env import history
from zenith_ppo.env.history import EventStore, HistoryRegistry


def event(environment_id=3, generation=7, sequence=0, *, kind=2):
    return {
        "environment_id": environment_id,
        "episode_generation": generation,
        "sequence": sequence,
        "kind": kind,
        "actor_seat": 0,
        "target_seat": 1,
        "visibility_mask": 0b1111,
        "args": (0, 1, 2, 3),
        "payload": b"payload",
    }


def test_store_id_hashes_each_event_once_and_caches_reads(monkeypatch):
    calls = 0
    original = history._canonical_row

    def counted(row):
        nonlocal calls
        calls += 1
        return original(row)

    monkeypatch.setattr(history, "_canonical_row", counted)
    store = EventStore(3, 7)
    empty_id = store.store_id
    assert store.store_id == empty_id
    assert calls == 0

    store.append([event()])
    populated_id = store.store_id
    assert populated_id != empty_id
    assert store.store_id == populated_id
    assert calls == 1


def test_store_id_round_trips_and_is_sensitive_to_ordered_content():
    rows = [event(sequence=0), event(sequence=1, kind=4)]
    original = EventStore(3, 7, rows)
    restored = HistoryRegistry.from_state_dict(
        {
            "stores": [
                {
                    "environment_id": 3,
                    "episode_generation": 7,
                    "rows": rows,
                    "sealed": False,
                }
            ]
        }
    ).get(3, 7)
    changed = EventStore(3, 7, [event(sequence=0), event(sequence=1, kind=5)])
    assert restored.store_id == original.store_id
    assert changed.store_id != original.store_id


def test_registry_seals_only_the_previous_active_generation():
    registry = HistoryRegistry()
    first = registry.get(3, 7)
    other = registry.get(4, 1)
    assert registry.get(3, 7) is first
    replacement = registry.get(3, 8)
    assert first.sealed
    assert not replacement.sealed
    assert not other.sealed


def test_registry_rejects_multiple_restored_active_generations():
    values = [
        {
            "environment_id": 3,
            "episode_generation": generation,
            "rows": [],
            "sealed": False,
        }
        for generation in (7, 8)
    ]
    try:
        HistoryRegistry.from_state_dict({"stores": values})
    except ValueError as exc:
        assert "multiple active" in str(exc)
    else:
        raise AssertionError("multiple active generations were accepted")
