"""Common deterministic/neural inference-policy adapters."""

from __future__ import annotations

CONSERVATIVE_BOT_ID = "conservative_bot"


class ConservativeBot:
    """Stable public-information bot shared by rollout and evaluation."""

    policy_id = CONSERVATIVE_BOT_ID

    @staticmethod
    def _kind(encoded, group):
        representative = encoded.action_representatives[group]
        return int(encoded.native_candidates[representative].kind)

    @staticmethod
    def _tile_type(encoded, group):
        representative = encoded.action_representatives[group]
        action = encoded.native_candidates[representative]
        return int(action.tiles[0]) // 4 if action.tiles else None

    def select_group(self, encoded, *, state=None) -> int:
        groups = range(len(encoded.action_representatives))
        wins = [group for group in groups if self._kind(encoded, group) in (8, 9)]
        if wins:
            return min(wins)

        discards = [group for group in groups if self._kind(encoded, group) in (1, 2)]
        if discards:
            options = discards
            safe_types = self._genbutsu_types(state, encoded.binding.seat)
            if safe_types is not None:
                safe = [
                    group for group in options
                    if self._tile_type(encoded, group) in safe_types
                ]
                if safe:
                    options = safe
            return min(options, key=lambda group: (
                self._kind(encoded, group) == 2,
                self._tile_type(encoded, group),
                group,
            ))
        passes = [group for group in groups if self._kind(encoded, group) == 0]
        return min(passes or groups)

    @staticmethod
    def _genbutsu_types(state, seat: int):
        if state is None:
            return None
        flags = tuple(int(value) for value in state.seat_flags)
        declared = [target for target in range(4) if target != int(seat) and flags[target] & 0b11]
        if not declared:
            return None
        by_seat = {
            target: {int(row.tile) // 4 for row in state.rivers if int(row.seat) == target}
            for target in declared
        }
        return set.intersection(*(by_seat[target] for target in declared))
