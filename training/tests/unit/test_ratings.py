import pytest
from zenith_ppo.evaluation.ratings import RatingTable


def test_all_players_receive_rating_and_order_is_canonical():
    pytest.importorskip("openskill")
    outcome = {"series_id": "s", "game_id": 0, "checkpoint_ids": ("a", "b", "c", "d"),
               "ranks": (0, 1, 2, 3), "valid": True}
    table = RatingTable().update([outcome])
    assert len(table.ratings) == 4 and table.ratings["a"].mu > table.ratings["d"].mu

