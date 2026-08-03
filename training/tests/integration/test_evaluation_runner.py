import pytest

from zenith_ppo.evaluation.runner import (
    cyclic_lineups,
    paired_bootstrap,
    run_series,
    run_series_batched,
    seat_balanced_lineups,
    series_requests,
)
from zenith_ppo.seeds import derive_seed


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
    for rotation, call in enumerate(calls[:4]):
        assert call[2] == {
            "ordinary_view": True,
            "rank_only": True,
            "gradients": False,
            "action_seed": derive_seed(101, f"rank_evaluation_lineup_{rotation}"),
        }
    for rotation, call in enumerate(calls[4:]):
        assert call[2] == {
            "ordinary_view": True,
            "rank_only": True,
            "gradients": False,
            "action_seed": derive_seed(202, f"rank_evaluation_lineup_{rotation}"),
        }


def test_batched_series_preserves_schedule_and_result_order():
    identities = ("candidate", "bc", "candidate", "bc")
    requests = series_requests(identities, (101, 202))
    batches = []

    def play_games(batch):
        batches.append(tuple(request["game_id"] for request in batch))
        return tuple({
            "ranks": (0, 1, 2, 3),
            "scores": (40_000 + request["game_id"], 30_000, 20_000, 10_000),
            "valid": True,
        } for request in batch)

    outcomes = run_series_batched(
        identities, (101, 202), play_games,
        batch_size=5, series_id="batched",
    )

    assert batches == [(0, 1, 2, 3, 4), (5, 6, 7, 8, 9), (10, 11)]
    assert len(requests) == len(outcomes) == 12
    assert [outcome["game_id"] for outcome in outcomes] == list(range(12))
    assert all(outcome["valid"] for outcome in outcomes)
    assert [outcome["seed"] for outcome in outcomes] == [101] * 6 + [202] * 6
    assert [outcome["rotation"] for outcome in outcomes] == list(range(6)) * 2


def test_batched_series_isolates_returned_game_errors():
    def play_games(batch):
        return tuple(
            RuntimeError("game failed") if request["game_id"] == 1 else {
                "ranks": (0, 1, 2, 3),
                "scores": (40_000, 30_000, 20_000, 10_000),
                "valid": True,
            }
            for request in batch
        )

    outcomes = run_series_batched(
        ("a", "b", "c", "d"), (7,), play_games, batch_size=4,
    )

    assert [outcome["valid"] for outcome in outcomes] == [True, False, True, True]
    assert outcomes[1]["error"] == "RuntimeError: game failed"


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
    assert report["pairwise_win_rate"]["mean"] == 1.0


def test_repeated_identities_cover_six_unique_seat_allocations():
    identities = ("candidate", "bc", "candidate", "bc")
    lineups = seat_balanced_lineups(identities)

    assert len(lineups) == 6
    assert len(set(lineups)) == 6
    assert all(lineup.count("candidate") == 2 for lineup in lineups)
    assert all(
        sum(lineup[seat] == "candidate" for lineup in lineups) == 3
        for seat in range(4)
    )

    calls = []
    run_series(
        identities,
        (17,),
        lambda lineup, seed, **_: calls.append((lineup, seed)) or {
            "ranks": (0, 1, 2, 3),
            "scores": (40, 30, 20, 10),
            "valid": True,
        },
    )
    assert calls == [(lineup, 17) for lineup in lineups]


def test_bootstrap_drops_an_entire_invalid_seed_block():
    outcomes = []
    for seed in (1, 2):
        for rotation in range(6):
            outcomes.append({
                "valid": not (seed == 2 and rotation == 5),
                "seed": seed,
                "game_id": len(outcomes),
                "rotation": rotation,
                "checkpoint_ids": ("candidate", "bc", "candidate", "bc"),
                "scores": (30_000, 20_000, 30_000, 20_000),
                "ranks": (0, 2, 1, 3),
            })

    report = paired_bootstrap(outcomes, "candidate", "bc", resamples=10)

    assert report["placement_difference"]["paired_seeds"] == 1
    assert report["score_difference"]["paired_seeds"] == 1
    assert report["pairwise_win_rate"]["paired_seeds"] == 1
