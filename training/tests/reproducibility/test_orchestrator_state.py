from zenith_ppo.env.history import HistoryRegistry
from zenith_ppo.evaluation.ratings import Rating, RatingTable
from zenith_ppo.population.registry import CheckpointPool, PoolEntry


def test_history_registry_round_trip_preserves_open_and_sealed_stores():
    histories = HistoryRegistry()
    histories.get(0, 1).append(({
        "environment_id": 0,
        "episode_generation": 1,
        "sequence": 0,
        "kind": 1,
        "args": (3, 4),
        "payload": b"payload",
    },))
    histories.get(0, 2)

    restored = HistoryRegistry.from_state_dict(histories.state_dict())

    assert restored.get(0, 1).sealed
    assert restored.get(0, 1).rows[0]["payload"] == b"payload"
    assert restored.get(0, 2).next_sequence == 0
    restored.retain({(0, 2)})
    assert len(restored.state_dict()["stores"]) == 1


def test_pool_and_rating_round_trip_preserve_immutable_identity():
    pool = CheckpointPool()
    pool.admit(PoolEntry("a", "/a", "run", 3, bytes=17))
    pool.pin("a", "active-match")
    restored_pool = CheckpointPool.from_state_dict(pool.state_dict())

    table = RatingTable({"a": Rating(27.0, 6.0, 8, (3, 2, 2, 1), "s")})
    restored_table = RatingTable.from_state_dict(table.state_dict())

    assert restored_pool.snapshot() == pool.snapshot()
    assert restored_table.ratings == table.ratings
