from zenith_ppo.evaluation.runner import convergence


def test_equal_budget_curves_report_crossing_and_area():
    curves = {"curriculum": [[(0, 0), (10, 1)]] * 3, "rank": [[(0, 0), (10, .5)]] * 3}
    result = convergence(curves, .5)
    assert set(result["areas"]) == {"curriculum", "rank"}

