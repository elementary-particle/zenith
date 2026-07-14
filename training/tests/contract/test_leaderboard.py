from zenith_ppo.evaluation.ratings import Rating, RatingTable


def test_leaderboard_contains_identity_uncertainty_and_ordinal():
    board = RatingTable({"checkpoint": Rating(mu=30, sigma=2, games=10)}).leaderboard()
    assert board[0][0] == "checkpoint" and board[0][1].ordinal == 24

