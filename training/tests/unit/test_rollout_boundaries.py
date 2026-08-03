import struct
from types import SimpleNamespace

import pytest

from zenith_ppo.rollout.collector import (
    _GameMetricAccumulator, _apply_boundaries, _mark_boundary_rows,
)


def test_boundary_counts_are_per_environment_not_per_seat():
    samples = [
        SimpleNamespace(
            match_boundary=False, terminal=False,
        )
        for _ in range(4)
    ]
    tails = {(9, 3, seat): seat for seat in range(4)}
    payload = b"".join(score.to_bytes(4, "little", signed=True) for score in (30_000, 25_000, 24_000, 21_000))
    payload += bytes((1, 2, 3, 4))
    batch = SimpleNamespace(transition=SimpleNamespace(events=(
        SimpleNamespace(environment_id=9, episode_generation=3, kind=15, payload=b""),
        SimpleNamespace(
            environment_id=9, episode_generation=3, kind=16,
            args=(11, 0, 0, 0), payload=payload,
        ),
    )))

    kyoku, matches = _apply_boundaries(batch, samples, tails)

    assert (kyoku, matches) == (1, 1)
    assert sum(sample.match_boundary for sample in samples) == 4


def test_game_event_metrics_use_player_kyoku_and_conditional_denominators():
    def event(kind, *, actor=255, target=255, args=(0, 0, 0, 0), payload=b""):
        return SimpleNamespace(
            environment_id=7, episode_generation=2, kind=kind,
            actor_seat=actor, target_seat=target, args=args, payload=payload,
        )

    def settlement(*deltas):
        return struct.pack("<4i", *deltas)

    metrics = _GameMetricAccumulator(lambda _key: 0b1111)
    metrics.observe((
        event(2),
        event(11, actor=0),
        event(5, actor=1),
        event(4, actor=2),
        event(4, actor=2),
        event(11, actor=2),
        event(13, actor=2, target=2, payload=settlement(-1000, -2000, 5000, -2000)),
        event(15),
        event(2),
        event(13, actor=0, target=3, payload=settlement(8000, 4000, 0, -12000)),
        event(13, actor=1, target=3, payload=settlement(8000, 4000, 0, -12000)),
        event(15),
        event(2),
        event(11, actor=0),
        event(6, actor=1),
        event(14, args=(3, 0b1111, 0, 0), payload=settlement(0, 0, 0, 0)),
        event(15),
        event(16, payload=settlement(31_000, 29_000, 40_000, -1000) + bytes((2, 3, 1, 4))),
    ))

    assert metrics.metrics() == pytest.approx({
        "game/player_win_rate": 3 / 12,
        "game/player_deal_in_rate": 1 / 12,
        "game/player_riichi_rate": 3 / 12,
        "game/player_calling_rate": 2 / 12,
        "game/player_average_winning_points": 17_000 / 3,
        "game/player_average_deal_in_points": 12_000,
        "game/exhaustive_ryukyoku_rate": 1 / 3,
        "game/player_bankrupt_rate": 1 / 4,
        "game/player_tsumo_rate": 1 / 3,
        "game/player_dama_rate": 2 / 3,
        "game/player_average_turns_before_winning": 5 / 3,
    })


def test_boundary_rank_supervision_is_sparse_per_policy_and_kyoku():
    samples = [
        SimpleNamespace(
            binding=SimpleNamespace(seat=seat),
            ppo_eligible=True,
            checkpoint_id="learner",
            rank_boundary_supervision=False,
        )
        for seat in (0, 1)
    ]

    _mark_boundary_rows(samples, (0, 1))

    assert [sample.rank_boundary_supervision for sample in samples] == [True, False]


def test_kyoku_boundary_marks_only_seats_that_acted_in_current_kyoku():
    samples = [
        SimpleNamespace(
            kyoku_boundary=False, match_boundary=False, terminal=False,
        )
        for _ in range(4)
    ]
    tails = {(7, 5, seat): seat for seat in range(4)}
    kyoku_tails = {(7, 5, 0): 0, (7, 5, 2): 2}
    settlement = (8_000, -8_000, 1_000, -1_000)
    payload = b"".join(
        score.to_bytes(4, "little", signed=True) for score in settlement
    )
    batch = SimpleNamespace(transition=SimpleNamespace(events=(
        SimpleNamespace(
            environment_id=7, episode_generation=5, kind=13,
            actor_seat=0, payload=payload,
        ),
        SimpleNamespace(
            environment_id=7, episode_generation=5, kind=15,
            actor_seat=255, payload=b"",
        ),
    )))

    _apply_boundaries(batch, samples, tails, kyoku_tails=kyoku_tails)

    assert tuple(sample.kyoku_boundary for sample in samples) == (True, False, True, False)
    assert kyoku_tails == {}


def test_match_outcome_binds_terminal_ranks_scores_and_policy_ids():
    samples = [
        SimpleNamespace(
            checkpoint_id=checkpoint_id,
            match_boundary=False, terminal=False,
        )
        for checkpoint_id in ("current", "a", "current", "b")
    ]
    tails = {(3, 8, seat): seat for seat in range(4)}
    scores = (31_000, 27_000, 24_000, 18_000)
    payload = b"".join(score.to_bytes(4, "little", signed=True) for score in scores)
    payload += bytes((1, 2, 3, 4))
    batch = SimpleNamespace(transition=SimpleNamespace(events=(
        SimpleNamespace(
            environment_id=3, episode_generation=8, kind=15,
            actor_seat=255, payload=b"",
        ),
        SimpleNamespace(
            environment_id=3, episode_generation=8, kind=16,
            actor_seat=255, args=(9, 0, 0, 0), payload=payload,
        ),
    )))
    outcomes = []

    kyoku, matches = _apply_boundaries(
        batch, samples, tails, match_outcomes=outcomes
    )

    assert (kyoku, matches) == (1, 1)
    assert len(outcomes) == 1
    assert outcomes[0].checkpoint_ids == ("current", "a", "current", "b")
    assert outcomes[0].ranks == (0, 1, 2, 3)
    assert outcomes[0].scores == scores
    assert outcomes[0].completed_kyoku == 9


def test_terminal_placements_use_trajectory_local_indices_without_scanning_samples():
    class IndexedOnly(list):
        def __iter__(self):
            raise AssertionError("terminal placement scanned the full rollout")

    samples = IndexedOnly([
        SimpleNamespace(
            match_boundary=False, terminal=False,
            terminal_placement=-1,
        )
        for _ in range(6)
    ])
    tails = {(5, 4, seat): seat + 1 for seat in range(4)}
    trajectory_indices = {
        (5, 4, 0): [0, 1],
        (5, 4, 1): [2],
        (5, 4, 2): [3],
        (5, 4, 3): [4],
    }
    payload = b"".join(
        score.to_bytes(4, "little", signed=True)
        for score in (30_000, 25_000, 24_000, 21_000)
    ) + bytes((1, 2, 3, 4))
    batch = SimpleNamespace(transition=SimpleNamespace(events=(
        SimpleNamespace(
            environment_id=5, episode_generation=4, kind=16,
            args=(8, 0, 0, 0), payload=payload,
        ),
    )))

    _, matches = _apply_boundaries(
        batch, samples, tails, trajectory_indices=trajectory_indices
    )

    assert matches == 1
    assert [samples[index].terminal_placement for index in range(6)] == [
        0, 0, 1, 2, 3, -1,
    ]
    assert trajectory_indices == {}
