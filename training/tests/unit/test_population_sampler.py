from zenith_ppo.compatibility import CompatibilitySet
from zenith_ppo.population.registry import CheckpointPool, PoolEntry
from zenith_ppo.population.sampler import UniformSampler
from zenith_ppo.seeds import SeedStreams


def test_uniform_distinct_history_and_bootstrap():
    empty = CheckpointPool().snapshot(); sampler = UniformSampler(SeedStreams(1))
    assert sampler.sample(empty, environment_id=0, generation=1, current_id="c", policy_version=1).learner_mask == 15
    pool = CheckpointPool()
    for key in "ab": pool.admit(PoolEntry(key, f"/{key}", CompatibilitySet(), "r", 1))
    lineup = sampler.sample(pool.snapshot(), environment_id=0, generation=1, current_id="c", policy_version=1)
    assert set(lineup.seat_policy_ids) == {"a", "b", "c"}


def test_rollout_cohort_is_small_deterministic_and_shared_by_lineups():
    pool = CheckpointPool()
    for key in "abcd":
        pool.admit(PoolEntry(key, f"/{key}", CompatibilitySet(), "r", 1))
    first = UniformSampler(SeedStreams(9))
    second = UniformSampler(SeedStreams(9))

    cohort, trace = first.select_cohort(pool.snapshot(), 2)
    repeated, repeated_trace = second.select_cohort(pool.snapshot(), 2)

    assert cohort == repeated
    assert trace == repeated_trace
    assert len(cohort.eligible) == len(trace) == 2
    allowed = {entry.checkpoint_id for entry in cohort.eligible}
    lineups = [
        first.sample(
            cohort,
            environment_id=environment_id,
            generation=1,
            current_id="current",
            policy_version=3,
        )
        for environment_id in range(8)
    ]
    assert all(
        set(lineup.seat_policy_ids) - {"current"} == allowed
        for lineup in lineups
    )

    sticky, sticky_trace = first.select_cohort(
        pool.snapshot(), 2, preferred=sorted(allowed)
    )
    assert {entry.checkpoint_id for entry in sticky.eligible} == allowed
    assert sticky_trace == ()
