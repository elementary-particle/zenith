from types import SimpleNamespace

from zenith_ppo.rewards.ranking import rewards as ranking_rewards
from zenith_ppo.rollout.collector import _apply_boundaries, _game_metrics_per_kyoku
from zenith_ppo.types import RewardRecord


def test_boundary_counts_are_per_environment_not_per_seat():
    samples = [
        SimpleNamespace(
            reward=RewardRecord(), match_boundary=False,
            terminal=False,
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
    assert tuple(sample.reward.rank_reward for sample in samples) == ranking_rewards((0, 1, 2, 3))


def test_game_event_metrics_are_normalized_per_completed_kyoku():
    counts = {
        "open_wins": 2,
        "closed_wins": 3,
        "deal_ins_after_opponent_riichi": 1,
        "exhaustive_ryukyoku": 4,
        "exhaustive_ryukyoku_tenpai_score": 6,
    }

    assert _game_metrics_per_kyoku(counts, 0) == {}
    assert _game_metrics_per_kyoku(counts, 10) == {
        "game/open_wins_per_kyoku": 0.2,
        "game/closed_wins_per_kyoku": 0.3,
        "game/deal_ins_after_opponent_riichi_per_kyoku": 0.1,
        "game/exhaustive_ryukyoku_rate": 0.4,
        "game/exhaustive_ryukyoku_tenpai_score_per_kyoku": 0.6,
    }


def test_accepted_riichi_is_charged_to_the_declaring_seat():
    samples = [
        SimpleNamespace(
            reward=RewardRecord(), match_boundary=False,
            terminal=False,
        )
        for _ in range(4)
    ]
    tails = {(4, 2, seat): seat for seat in range(4)}
    batch = SimpleNamespace(transition=SimpleNamespace(events=(
        SimpleNamespace(
            environment_id=4, episode_generation=2, kind=12,
            actor_seat=1, payload=b"",
        ),
    )))

    kyoku, matches = _apply_boundaries(batch, samples, tails)

    assert (kyoku, matches) == (0, 0)
    assert tuple(sample.reward.kyoku_delta for sample in samples) == (0, -1, 0, 0)


def test_terminal_settlement_is_added_to_prior_score_changes():
    samples = [
        SimpleNamespace(
            reward=RewardRecord(kyoku_delta=-1 if seat == 0 else 0),
            match_boundary=False, terminal=False,
        )
        for seat in range(4)
    ]
    tails = {(7, 5, seat): seat for seat in range(4)}
    settlement = (8_000, -8_000, 0, 0)
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

    kyoku, matches = _apply_boundaries(batch, samples, tails)

    assert (kyoku, matches) == (1, 0)
    assert tuple(sample.reward.kyoku_delta for sample in samples) == (7, -8, 0, 0)


def test_kyoku_boundary_marks_only_seats_that_acted_in_current_kyoku():
    samples = [
        SimpleNamespace(
            reward=RewardRecord(), kyoku_boundary=False,
            match_boundary=False, terminal=False,
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

    assert tuple(sample.reward.kyoku_delta for sample in samples) == (8, 0, 1, 0)
    assert tuple(sample.kyoku_boundary for sample in samples) == (True, False, True, False)
    assert kyoku_tails == {}


def test_match_outcome_binds_terminal_ranks_scores_and_policy_ids():
    samples = [
        SimpleNamespace(
            checkpoint_id=checkpoint_id, reward=RewardRecord(),
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
