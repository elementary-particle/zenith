import pytest
from types import SimpleNamespace

from zenith_ppo.evaluation.ratings import RatingTable
from zenith_ppo.inference import CONSERVATIVE_BOT_ID
from zenith_ppo.rollout.rating import ConservativeBotRating


def test_all_players_receive_rating_and_order_is_canonical():
    pytest.importorskip("openskill")
    outcome = {"series_id": "s", "game_id": 0, "checkpoint_ids": ("a", "b", "c", "d"),
               "ranks": (0, 1, 2, 3), "valid": True}
    table = RatingTable().update([outcome])
    assert len(table.ratings) == 4 and table.ratings["a"].mu > table.ratings["d"].mu


def rollout(checkpoint_ids, ranks):
    return SimpleNamespace(
        checkpoint_ids=checkpoint_ids, ranks=ranks,
        scores=(40_000, 30_000, 20_000, 10_000),
    )


def test_rollout_rating_is_direct_policy_vs_bot_placement_rate():
    table = ConservativeBotRating().update((
        rollout(("current", CONSERVATIVE_BOT_ID, "current", CONSERVATIVE_BOT_ID),
                (0, 2, 1, 3)),
    ))
    metrics = table.metrics()
    assert table.bot_matches == 1
    assert table.comparisons == table.wins == 4
    assert metrics["win_rate_vs_conservative_bot"] == 1.0
    assert 0.0 < metrics["lower_win_rate_vs_conservative_bot"] < 1.0


def test_rollout_rating_ignores_pure_self_play_and_round_trips():
    table = ConservativeBotRating().update((SimpleNamespace(
        checkpoint_ids=("current",) * 4,
        ranks=(0, 1, 2, 3),
    ),))
    assert table.state_dict() == {"bot_matches": 0, "wins": 0, "comparisons": 0}
    assert ConservativeBotRating.from_state_dict(table.state_dict()).metrics() == table.metrics()
