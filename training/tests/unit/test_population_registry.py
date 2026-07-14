from zenith_ppo.compatibility import CompatibilitySet
from zenith_ppo.population.registry import CheckpointPool, PoolEntry


def test_entries_are_immutable_retirable_and_pinnable():
    pool = CheckpointPool(); entry = PoolEntry("a", "/a", CompatibilitySet(), "run", 1)
    pool.admit(entry); pool.pin("a", "match"); pool.retire("a")
    snapshot = pool.snapshot()
    assert snapshot.entries[0].pins == ("match",) and snapshot.eligible == ()

