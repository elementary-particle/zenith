from types import SimpleNamespace

from zenith_ppo.rewards.ranking import rewards as ranking_rewards
from zenith_ppo.rollout.collector import _apply_boundaries, _public_tiles
from zenith_ppo.types import RewardRecord


def test_boundary_counts_are_per_environment_not_per_seat():
    samples = [
        SimpleNamespace(
            reward=RewardRecord(), kyoku_boundary=False, match_boundary=False,
            terminal=False,
        )
        for _ in range(4)
    ]
    tails = {(9, 3, seat): seat for seat in range(4)}
    payload = b"".join(score.to_bytes(4, "little", signed=True) for score in (30_000, 25_000, 24_000, 21_000))
    payload += bytes((1, 2, 3, 4))
    batch = SimpleNamespace(transition=SimpleNamespace(events=(
        SimpleNamespace(environment_id=9, episode_generation=3, kind=15, payload=b""),
        SimpleNamespace(environment_id=9, episode_generation=3, kind=16, payload=payload),
    )))

    kyoku, matches = _apply_boundaries(batch, samples, tails)

    assert (kyoku, matches) == (1, 1)
    assert sum(sample.kyoku_boundary for sample in samples) == 4
    assert sum(sample.match_boundary for sample in samples) == 4
    assert tuple(sample.reward.rank_reward for sample in samples) == ranking_rewards((0, 1, 2, 3))


def test_public_tiles_are_bounded_to_latest_kyoku():
    events = [
        {"kind": 2, "args": (0, 0, 0, 0)},
        {"kind": 4, "args": (4, 0, 0, 0)},
        {"kind": 15, "args": (0, 0, 0, 0)},
        {"kind": 2, "args": (1, 0, 0, 0)},
        {"kind": 4, "args": (40, 0, 0, 0)},
        {"kind": 5, "args": (44, 45, 46, 255)},
    ]
    assert _public_tiles(events) == (40, 44, 45, 46)
