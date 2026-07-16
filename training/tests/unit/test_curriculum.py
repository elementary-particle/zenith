import pytest

from zenith_ppo.rewards.curriculum import Curriculum


CONFIG = {
    "total_matches": 46_000,
    "rank_start_fraction": .60, "rank_ramp_fraction": .15,
    "minimum_discard_rows": 4096, "competence_threshold": .15,
    "pause_threshold": .18, "competence_batches": 5,
    "regression_batches": 2, "taper_matches": 10_000,
}


def test_gate_requires_five_valid_batches_and_ignores_small_batches():
    schedule = Curriculum(CONFIG)
    assert not schedule.observe(applicable_rows=4095, worse_shanten_rate=.01,
                                completed_matches=128)
    for _ in range(4):
        schedule.observe(applicable_rows=4096, worse_shanten_rate=.15,
                         completed_matches=128)
    assert schedule.snapshot(640, 0).guidance_scale == 1
    schedule.observe(applicable_rows=4096, worse_shanten_rate=.14,
                     completed_matches=128)
    assert schedule.snapshot(768, 0).guidance_phase == "taper"


def test_pause_regression_restore_and_post_zero_reactivation():
    schedule = Curriculum(CONFIG, {"phase": "taper", "guidance_scale": .5,
                                   "taper_matches": 5000})
    schedule.observe(applicable_rows=4096, worse_shanten_rate=.17,
                     completed_matches=128)
    assert schedule.snapshot(10_000, 0).guidance_scale == .5
    schedule.observe(applicable_rows=4096, worse_shanten_rate=.19,
                     completed_matches=128)
    assert schedule.snapshot(10_128, 0).guidance_scale == .5
    schedule.observe(applicable_rows=4096, worse_shanten_rate=.19,
                     completed_matches=128)
    restored = schedule.snapshot(10_256, 0)
    assert restored.guidance_phase == "full"
    assert restored.guidance_scale == 1
    assert restored.bot_fraction == pytest.approx(.05)


def test_rank_waits_for_time_and_zero_guidance_then_freezes_without_rewind():
    schedule = Curriculum(CONFIG, {"phase": "zero", "guidance_scale": 0})
    assert schedule.snapshot(27_599, 0).weights == (1, 0)
    assert schedule.snapshot(31_050, 0).weights == pytest.approx((.5, .5))
    schedule.observe(applicable_rows=4096, worse_shanten_rate=.19,
                     completed_matches=128)
    schedule.observe(applicable_rows=4096, worse_shanten_rate=.19,
                     completed_matches=128)
    frozen = schedule.snapshot(40_000, 0).weights
    assert frozen == pytest.approx((.5, .5))
    assert schedule.rank_frozen


def test_state_round_trip_is_deterministic():
    first = Curriculum(CONFIG)
    first.observe(applicable_rows=4096, worse_shanten_rate=.14, completed_matches=128)
    second = Curriculum(CONFIG, first.state_dict())
    assert second.snapshot(128, 7) == first.snapshot(128, 7)
