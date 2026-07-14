import pytest
from zenith_ppo.evaluation.ratings import RatingTable


def test_invalid_games_change_no_rating():
    pytest.importorskip("openskill")
    table = RatingTable().update([{"series_id": "s", "game_id": 0,
        "checkpoint_ids": ("a", "b", "c", "d"), "ranks": (), "valid": False}])
    assert table.ratings == {}

