from zenith_ppo.inference import CONSERVATIVE_BOT_ID
from zenith_ppo.population.league import (
    CheckpointLeague,
    EMASelfPlayLeague,
    PureSelfPlayLeague,
)
from zenith_ppo.population.sampler import SelfPlaySampler
from zenith_ppo.seeds import SeedStreams


def test_normal_rollouts_select_two_of_four_league_models():
    sampler = SelfPlaySampler(SeedStreams(1))
    lineup = sampler.sample(
        environment_id=0, generation=1, current_id="current", policy_version=3
    )

    assert len(set(lineup.seat_policy_ids)) == 2
    assert all(lineup.seat_policy_ids.count(policy_id) == 2
               for policy_id in set(lineup.seat_policy_ids))
    assert lineup.learner_mask == 0b1111
    assert CONSERVATIVE_BOT_ID not in lineup.seat_policy_ids


def test_self_play_assignment_is_seeded_and_reproducible():
    first = SelfPlaySampler(SeedStreams(9))
    second = SelfPlaySampler(SeedStreams(9))

    left = [first.sample(
        environment_id=index, generation=1, current_id="current", policy_version=3
    ) for index in range(8)]
    right = [second.sample(
        environment_id=index, generation=1, current_id="current", policy_version=3
    ) for index in range(8)]

    assert left == right


def test_conservative_bot_is_never_sampled_for_training():
    sampler = SelfPlaySampler(SeedStreams(3))
    lineups = [sampler.sample(
        environment_id=environment_id,
        generation=1,
        current_id="current",
        policy_version=8,
    ) for environment_id in range(4)]

    assert all(CONSERVATIVE_BOT_ID not in lineup.seat_policy_ids for lineup in lineups)
    assert all(lineup.learner_mask == 0b1111 for lineup in lineups)


def test_pure_self_play_assigns_one_live_policy_to_every_seat():
    sampler = SelfPlaySampler(SeedStreams(3), PureSelfPlayLeague())
    lineup = sampler.sample(
        environment_id=0, generation=1,
        current_id="self-play", policy_version=8,
    )

    assert lineup.seat_policy_ids == ("self-play",) * 4
    assert lineup.learner_mask == 0b1111
    assert lineup.pool_snapshot_id == "pure-self-play"


def test_checkpoint_league_marks_only_alternating_learner_seats():
    league = CheckpointLeague(("old-a", "old-b"), minimum_games=1)
    sampler = SelfPlaySampler(SeedStreams(3), league)
    lineup = sampler.sample(
        environment_id=0, generation=1,
        current_id="learner", policy_version=8,
    )

    assert set(lineup.seat_policy_ids) in ({"learner", "old-a"}, {"learner", "old-b"})
    assert lineup.seat_policy_ids.count("learner") == 2
    assert lineup.learner_mask in (0b0101, 0b1010)
    assert lineup.pool_snapshot_id == "checkpoint-league"


def test_checkpoint_response_lineup_rotates_one_learner_seat():
    league = CheckpointLeague(
        ("target",), minimum_games=1, learner_seats=1,
    )
    sampler = SelfPlaySampler(SeedStreams(3), league)
    lineups = [sampler.sample(
        environment_id=index, generation=1,
        current_id="learner", policy_version=8,
    ) for index in range(16)]

    assert all(row.seat_policy_ids.count("learner") == 1 for row in lineups)
    assert all(row.seat_policy_ids.count("target") == 3 for row in lineups)
    assert all(row.learner_mask.bit_count() == 1 for row in lineups)
    assert {row.learner_mask for row in lineups} == {1, 2, 4, 8}


def test_ema_self_play_marks_only_alternating_learner_seats():
    league = EMASelfPlayLeague()
    sampler = SelfPlaySampler(SeedStreams(3), league)
    lineup = sampler.sample(
        environment_id=0, generation=1, current_id="learner", policy_version=8
    )

    assert set(lineup.seat_policy_ids) == {"learner", "ema-opponent"}
    assert lineup.seat_policy_ids.count("learner") == 2
    assert lineup.learner_mask in (0b0101, 0b1010)
    assert lineup.pool_snapshot_id == "checkpoint-league"
