from types import SimpleNamespace

from zenith_ppo.population.league import (
    AdversarialLeague,
    CheckpointLeague,
    EMASelfPlayLeague,
)
from zenith_ppo.seeds import SeedStreams


def _outcome(left, right, ranks=(0, 2, 1, 3)):
    return SimpleNamespace(
        checkpoint_ids=(left, right, left, right),
        ranks=ranks,
    )


def test_league_explores_every_pair_then_selects_the_strongest_adversary():
    league = AdversarialLeague()
    rng = SeedStreams(5).python_rng("opponent")
    focal = league.policy_ids[0]
    for opponent in league.policy_ids[1:]:
        league.record((_outcome(focal, opponent),))
    # Make league-2 the only opponent that beats the focal policy.
    league.record((_outcome(focal, "league-2", ranks=(2, 0, 3, 1)),))

    selected_focal, opponent, trace = league.select_pair(rng)

    assert selected_focal == focal
    assert opponent == "league-2"
    assert trace["mode"] == "adversarial-perturbation"


def test_league_state_round_trip_preserves_matchmaking_cursor_and_payoffs():
    first = AdversarialLeague()
    first.record((_outcome("league-0", "league-1"),))
    rng = SeedStreams(8).python_rng("opponent")
    first.select_pair(rng)
    restored = AdversarialLeague(state=first.state_dict())

    assert restored.state_dict() == first.state_dict()
    assert restored.estimate("league-1", "league-0").mean \
        == -first.estimate("league-0", "league-1").mean


def test_checkpoint_league_rates_frozen_opponents_and_round_trips():
    league = CheckpointLeague(
        ("old-a", "old-b"), minimum_games=1, uniform_fraction=0.0
    )
    rng = SeedStreams(8).python_rng("opponent")
    learner, opponent, trace = league.select_pair(rng)
    outcome = SimpleNamespace(
        match_id=(3, 4),
        checkpoint_ids=(learner, opponent, learner, opponent),
        ranks=(0, 2, 1, 3),
    )
    league.record((outcome,))

    assert trace["mode"] == "rating-calibration"
    assert league.ratings.ratings[learner].ordinal \
        > league.ratings.ratings[opponent].ordinal
    restored = CheckpointLeague(
        ("old-a", "old-b"), minimum_games=1, uniform_fraction=0.0,
        state=league.state_dict(),
    )
    assert restored.state_dict() == league.state_dict()


def test_ema_self_play_league_round_trips_completed_games():
    league = EMASelfPlayLeague()
    learner, opponent, trace = league.select_pair(None)
    outcome = SimpleNamespace(
        checkpoint_ids=(learner, opponent, learner, opponent),
        ranks=(0, 2, 1, 3),
    )

    assert trace["kind"] == "ema_self_play"
    assert league.record((outcome,)) == 1
    restored = EMASelfPlayLeague(state=league.state_dict())
    assert restored.state_dict() == league.state_dict()
