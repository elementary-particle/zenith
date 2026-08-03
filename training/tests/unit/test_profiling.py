from zenith_ppo.profiling import StageProfiler


def test_stage_profiler_is_opt_in_and_sorts_largest_first():
    disabled = StageProfiler()
    with disabled.measure("ignored"):
        pass
    assert disabled.snapshot()["stages"] == []

    profiler = StageProfiler(enabled=True)
    with profiler.measure("small"):
        with profiler.measure("nested"):
            pass
    profiler.observe("rows_per_launch", 2)
    profiler.observe("rows_per_launch", 6)
    with profiler.measure("large"):
        sum(range(1000))
    snapshot = profiler.snapshot()
    assert {row["stage"] for row in snapshot["stages"]} == {
        "small", "nested", "large"
    }
    assert all(
        row["seconds"] >= row["exclusive_seconds"] >= 0
        and row["calls"] == 1
        for row in snapshot["stages"]
    )
    assert snapshot["observations"]["rows_per_launch"]["mean"] == 4
    assert snapshot["peak_memory_bytes"] > 0
