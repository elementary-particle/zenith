from zenith_ppo.inference import CONSERVATIVE_BOT_ID
from zenith_ppo.population.sampler import SelfPlaySampler
from zenith_ppo.seeds import SeedStreams


def test_normal_rollouts_are_four_seat_self_play():
    sampler = SelfPlaySampler(
        SeedStreams(1), conservative_bot_match_fraction=0
    )
    lineup = sampler.sample(
        environment_id=0, generation=1, current_id="current", policy_version=3
    )

    assert lineup.seat_policy_ids == ("current",) * 4
    assert lineup.learner_mask == 0b1111


def test_self_play_assignment_is_seeded_and_reproducible():
    first = SelfPlaySampler(SeedStreams(9), conservative_bot_match_fraction=.5)
    second = SelfPlaySampler(SeedStreams(9), conservative_bot_match_fraction=.5)

    left = [first.sample(
        environment_id=index, generation=1, current_id="current", policy_version=3
    ) for index in range(8)]
    right = [second.sample(
        environment_id=index, generation=1, current_id="current", policy_version=3
    ) for index in range(8)]

    assert left == right


def test_bot_probes_use_two_rotating_live_policy_seats():
    sampler = SelfPlaySampler(
        SeedStreams(3), conservative_bot_match_fraction=1
    )
    lineups = [sampler.sample(
        environment_id=environment_id,
        generation=1,
        current_id="current",
        policy_version=8,
    ) for environment_id in range(4)]

    assert all(lineup.seat_policy_ids.count(CONSERVATIVE_BOT_ID) == 2 for lineup in lineups)
    assert {lineup.learner_mask for lineup in lineups} == {0b0101, 0b1010}
