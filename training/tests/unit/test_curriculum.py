import pytest

from zenith_ppo.rewards.curriculum import Curriculum


def test_training_progress_is_completed_match_fraction():
    schedule = Curriculum({"total_matches": 100})

    assert schedule.snapshot(25, 7).progress == pytest.approx(0.25)
    assert schedule.snapshot(100, 8).progress == 1.0
    assert schedule.snapshot(200, 9).progress == 1.0


def test_training_progress_has_empty_deterministic_state():
    first = Curriculum({"total_matches": 100})
    second = Curriculum({"total_matches": 100}, first.state_dict())

    assert second.snapshot(25, 7) == first.snapshot(25, 7)
