import numpy as np
import pytest

from zenith_ppo.rollout.game_metrics import metric_values, native_match_counts


def test_native_gameplay_columns_aggregate_by_policy_seats():
    columns = {
        "terminal_completed_kyoku": np.array([3]),
        "terminal_exhaustive_ryukyoku": np.array([1]),
        "terminal_scores": np.array([[31_000, 29_000, 40_000, -1_000]]),
        "terminal_wins": np.array([[1, 1, 1, 0]]),
        "terminal_deal_ins": np.array([[0, 0, 0, 1]]),
        "terminal_riichi_hands": np.array([[2, 0, 1, 0]]),
        "terminal_calling_hands": np.array([[0, 2, 0, 0]]),
        "terminal_tsumo_wins": np.array([[0, 0, 1, 0]]),
        "terminal_dama_wins": np.array([[1, 1, 0, 0]]),
        "terminal_winning_points": np.array([[8_000, 4_000, 5_000, 0]]),
        "terminal_winning_point_events": np.array([[1, 1, 1, 0]]),
        "terminal_deal_in_points": np.array([[0, 0, 0, 12_000]]),
        "terminal_deal_in_point_events": np.array([[0, 0, 0, 1]]),
        "terminal_winning_turns": np.array([[1, 1, 3, 0]]),
        "terminal_winning_turn_events": np.array([[1, 1, 1, 0]]),
    }
    counts = native_match_counts(columns, 0, ("learner",) * 4)["learner"]
    assert metric_values(counts) == pytest.approx({
        "game/player_win_rate": 3 / 12,
        "game/player_deal_in_rate": 1 / 12,
        "game/player_riichi_rate": 3 / 12,
        "game/player_calling_rate": 2 / 12,
        "game/player_average_winning_points": 17_000 / 3,
        "game/player_average_deal_in_points": 12_000,
        "game/exhaustive_ryukyoku_rate": 1 / 3,
        "game/player_bankrupt_rate": 1 / 4,
        "game/player_tsumo_rate": 1 / 3,
        "game/player_dama_rate": 2 / 3,
        "game/player_average_turns_before_winning": 5 / 3,
    })


def test_shared_policy_counts_kyoku_once_but_player_opportunities_per_seat():
    columns = {
        "terminal_completed_kyoku": np.array([2]),
        "terminal_exhaustive_ryukyoku": np.array([0]),
        "terminal_scores": np.array([[25_000] * 4]),
        **{
            name: np.zeros((1, 4), dtype=np.int64)
            for name in (
                "terminal_wins", "terminal_deal_ins", "terminal_riichi_hands",
                "terminal_calling_hands", "terminal_tsumo_wins",
                "terminal_dama_wins", "terminal_winning_points",
                "terminal_winning_point_events", "terminal_deal_in_points",
                "terminal_deal_in_point_events", "terminal_winning_turns",
                "terminal_winning_turn_events",
            )
        },
    }
    counts = native_match_counts(columns, 0, ("a", "a", "b", "b"))
    assert counts["a"]["kyoku"] == counts["b"]["kyoku"] == 2
    assert counts["a"]["player_kyoku"] == counts["b"]["player_kyoku"] == 4
