"""Aggregate native per-match gameplay counters into public metrics."""

COUNT_NAMES = (
    "kyoku", "player_kyoku", "wins", "deal_ins", "riichi_hands",
    "calling_hands", "tsumo_wins", "dama_wins", "winning_points",
    "winning_point_events", "deal_in_points", "deal_in_point_events",
    "winning_turns", "winning_turn_events", "exhaustive_ryukyoku",
    "player_matches", "bankrupt_matches",
)

SEAT_COLUMNS = {
    "wins": "terminal_wins",
    "deal_ins": "terminal_deal_ins",
    "riichi_hands": "terminal_riichi_hands",
    "calling_hands": "terminal_calling_hands",
    "tsumo_wins": "terminal_tsumo_wins",
    "dama_wins": "terminal_dama_wins",
    "winning_points": "terminal_winning_points",
    "winning_point_events": "terminal_winning_point_events",
    "deal_in_points": "terminal_deal_in_points",
    "deal_in_point_events": "terminal_deal_in_point_events",
    "winning_turns": "terminal_winning_turns",
    "winning_turn_events": "terminal_winning_turn_events",
}


def empty_counts():
    return {name: 0.0 for name in COUNT_NAMES}


def add_counts(target, source):
    for name in COUNT_NAMES:
        target[name] += float(source[name])
    return target


def native_owned_counts(columns, terminal, owned):
    """Return exact counters for a set of seats in one native match."""
    completed_kyoku = int(columns["terminal_completed_kyoku"][terminal])
    owned = tuple(owned)
    counts = empty_counts()
    counts["kyoku"] = float(completed_kyoku)
    counts["player_kyoku"] = float(completed_kyoku * len(owned))
    counts["exhaustive_ryukyoku"] = float(
        columns["terminal_exhaustive_ryukyoku"][terminal]
    )
    counts["player_matches"] = float(len(owned))
    counts["bankrupt_matches"] = float(sum(
        int(columns["terminal_scores"][terminal][seat]) < 0
        for seat in owned
    ))
    for name, column in SEAT_COLUMNS.items():
        counts[name] = sum(
            float(columns[column][terminal][seat]) for seat in owned
        )
    return counts


def native_match_counts(columns, terminal, lineup):
    """Return exact counters per policy from one native terminal row."""
    return {
        policy_id: native_owned_counts(
            columns,
            terminal,
            (seat for seat, owner in enumerate(lineup) if owner == policy_id),
        )
        for policy_id in dict.fromkeys(lineup)
    }


def metric_values(counts):
    """Finalize player-opportunity and conditional gameplay statistics."""
    player_kyoku = float(counts["player_kyoku"])
    kyoku = float(counts["kyoku"])
    player_matches = float(counts["player_matches"])
    if player_kyoku <= 0 or kyoku <= 0 or player_matches <= 0:
        return {}
    return {
        "game/player_win_rate": counts["wins"] / player_kyoku,
        "game/player_deal_in_rate": counts["deal_ins"] / player_kyoku,
        "game/player_riichi_rate": counts["riichi_hands"] / player_kyoku,
        "game/player_calling_rate": counts["calling_hands"] / player_kyoku,
        "game/player_average_winning_points": counts["winning_points"] / max(
            1.0, counts["winning_point_events"]
        ),
        "game/player_average_deal_in_points": counts["deal_in_points"] / max(
            1.0, counts["deal_in_point_events"]
        ),
        "game/exhaustive_ryukyoku_rate": counts["exhaustive_ryukyoku"] / kyoku,
        "game/player_bankrupt_rate": counts["bankrupt_matches"] / player_matches,
        "game/player_tsumo_rate": counts["tsumo_wins"] / max(1.0, counts["wins"]),
        "game/player_dama_rate": counts["dama_wins"] / max(1.0, counts["wins"]),
        "game/player_average_turns_before_winning": counts["winning_turns"] / max(
            1.0, counts["winning_turn_events"]
        ),
    }
