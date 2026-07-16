from zenith_ppo.rewards.curriculum import Curriculum


def test_rank_blend_requires_both_elapsed_budget_and_zero_guidance():
    config = {
        "total_matches": 100, "rank_start_fraction": .6,
        "rank_ramp_fraction": .15, "minimum_discard_rows": 1,
        "competence_threshold": .15, "pause_threshold": .18,
        "competence_batches": 1, "regression_batches": 2,
        "taper_matches": 1,
    }
    guided = Curriculum(config)
    assert guided.snapshot(100, 0).weights == (1, 0)
    competent = Curriculum(config, {"phase": "zero", "guidance_scale": 0})
    values = [competent.snapshot(matches, matches) for matches in range(60, 76)]
    assert any(0 < value.weights[0] < 1 for value in values)
    assert values[-1].weights == (0, 1)
