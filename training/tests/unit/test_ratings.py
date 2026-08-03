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
    assert metrics["rolling_bot_matches"] == 1
    assert metrics["rolling_head_to_head_comparisons"] == 4
    assert metrics["rolling_window_updates"] == 1
    assert metrics["rolling_win_rate_vs_conservative_bot"] == 1.0
    assert 0.0 < metrics["rolling_lower_win_rate_vs_conservative_bot"] < 1.0


def test_rollout_rating_zero_probe_update_omits_rate():
    table = ConservativeBotRating().update((SimpleNamespace(
        checkpoint_ids=("current",) * 4,
        ranks=(0, 1, 2, 3),
    ),))
    metrics = table.metrics()
    assert metrics == {
        "rolling_bot_matches": 0.0,
        "rolling_head_to_head_comparisons": 0.0,
        "rolling_window_updates": 1.0,
    }


def test_rollout_rating_evicts_the_eleventh_update():
    win = rollout(
        ("current", CONSERVATIVE_BOT_ID, "current", CONSERVATIVE_BOT_ID),
        (0, 2, 1, 3),
    )
    loss = rollout(
        ("current", CONSERVATIVE_BOT_ID, "current", CONSERVATIVE_BOT_ID),
        (2, 0, 3, 1),
    )
    table = ConservativeBotRating().update((win,))
    for _ in range(10):
        table.update((loss,))

    metrics = table.metrics()
    assert metrics["rolling_window_updates"] == 10
    assert metrics["rolling_bot_matches"] == 10
    assert metrics["rolling_head_to_head_comparisons"] == 40
    assert metrics["rolling_win_rate_vs_conservative_bot"] == 0.0


def test_rollout_rating_round_trip_preserves_window():
    table = ConservativeBotRating()
    table.update(())
    table.update((rollout(
        ("current", CONSERVATIVE_BOT_ID, "current", CONSERVATIVE_BOT_ID),
        (0, 2, 1, 3),
    ),))

    state = table.state_dict()
    restored = ConservativeBotRating.from_state_dict(state)
    assert state["version"] == 2
    assert state["window_updates"] == 10
    assert len(state["updates"]) == 2
    assert restored.state_dict() == state
    assert restored.metrics() == table.metrics()


def test_rollout_rating_legacy_lifetime_state_starts_fresh():
    restored = ConservativeBotRating.from_state_dict({
        "bot_matches": 50,
        "wins": 100,
        "comparisons": 200,
    })
    assert restored.state_dict()["updates"] == []
    assert restored.metrics()["rolling_window_updates"] == 0


@pytest.mark.parametrize("state", [
    {"bot_matches": 1, "wins": 1, "comparisons": 1, "mystery": 1},
    {"version": 3, "window_updates": 10, "updates": []},
    {"version": 2, "window_updates": 9, "updates": []},
    {"version": 2, "window_updates": 10, "updates": [
        {"bot_matches": 1, "wins": 5, "comparisons": 4},
    ]},
])
def test_rollout_rating_rejects_incompatible_or_malformed_state(state):
    with pytest.raises(ValueError):
        ConservativeBotRating.from_state_dict(state)
