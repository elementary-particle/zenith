import pytest

from zenith_ppo.evaluation.runner import cyclic_lineups, paired_bootstrap, run_series


def test_cyclic_block_balances_every_initial_seat():
    blocks = cyclic_lineups(("a", "b", "c", "d"))
    assert all(sorted(block[seat] for block in blocks) == ["a", "b", "c", "d"] for seat in range(4))


def test_each_held_out_seed_runs_a_complete_cyclic_seat_block():
    calls = []

    def play(lineup, seed, **contract):
        calls.append((lineup, seed, contract))
        return {"ranks": (0, 1, 2, 3), "scores": (40, 30, 20, 10), "valid": True}

    outcomes = run_series(("a", "b", "c", "d"), (101, 202), play)

    assert len(outcomes) == 8
    assert [seed for _, seed, _ in calls] == [101] * 4 + [202] * 4
    assert [lineup for lineup, _, _ in calls[:4]] == list(cyclic_lineups(("a", "b", "c", "d")))
    assert all(outcome["game_id"] == index for index, outcome in enumerate(outcomes))
    assert all(call[2] == {
        "ordinary_view": True, "rank_only": True, "gradients": False
    } for call in calls)


def test_repeated_bot_identities_and_paired_bootstrap():
    assert len(cyclic_lineups(("candidate", "bot", "baseline", "bot"))) == 4
    outcomes = [{
        "valid": True, "seed": seed, "game_id": seed,
        "checkpoint_ids": ("candidate", "bot", "baseline", "bot"),
        "scores": (26_000 + seed, 25_000, 24_000, 25_000),
        "ranks": (0, 2, 1, 3),
    } for seed in range(8)]
    report = paired_bootstrap(outcomes, "candidate", "baseline", resamples=100)
    assert report["score_difference"]["mean"] == pytest.approx(2.0035)
    assert report["placement_difference"]["mean"] == -1.0
