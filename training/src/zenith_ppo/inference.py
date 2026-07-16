"""Common deterministic/neural inference-policy adapters."""

from __future__ import annotations

from typing import Protocol


CONSERVATIVE_BOT_ID = "conservative_bot"


class InferencePolicy(Protocol):
    policy_id: str

    def select_group(self, encoded, *, state=None) -> int: ...


class NeuralCheckpointPolicy:
    """Single-row adapter used by evaluation tests."""

    def __init__(self, policy_id, model, *, device="cpu", backend="sdpa"):
        self.policy_id, self.model = str(policy_id), model
        self.device, self.backend = device, backend

    def select_group(self, encoded, *, state=None) -> int:
        import torch
        from .encoding.packing import model_batch

        inputs = model_batch([encoded], device=self.device, backend=self.backend)
        self.model.eval()
        with torch.no_grad():
            output = self.model.forward_actor(**inputs)
        return int(output.log_probabilities.argmax())


class ConservativeBot:
    """Stable public-information bot shared by rollout and evaluation."""

    policy_id = CONSERVATIVE_BOT_ID

    @staticmethod
    def _kind(encoded, group):
        representative = encoded.action_representatives[group]
        return int(encoded.native_actions[representative].kind)

    @staticmethod
    def _tile_type(encoded, group):
        representative = encoded.action_representatives[group]
        action = encoded.native_actions[representative]
        return int(action.tiles[0]) // 4 if action.tiles else None

    def select_group(self, encoded, *, state=None) -> int:
        groups = range(len(encoded.action_representatives))
        wins = [group for group in groups if self._kind(encoded, group) in (8, 9)]
        if wins:
            return min(wins)

        reaction = encoded.teachers.reaction
        if reaction is not None:
            return max(
                range(len(reaction.probabilities)),
                key=lambda group: (reaction.probabilities[group], -group),
            )

        discard = encoded.teachers.discard
        if discard is not None:
            options = list(range(len(discard.group_sets)))
            safe_types = self._genbutsu_types(state, encoded.binding.seat)
            if safe_types is not None:
                safe = [
                    option for option in options
                    if self._tile_type(encoded, discard.group_sets[option][0]) in safe_types
                ]
                if safe:
                    options = safe
            option = min(options, key=lambda value: (discard.costs[value], value))
            option_groups = discard.group_sets[option]
            riichi = [group for group in option_groups if self._kind(encoded, group) == 2]
            return min(riichi or option_groups)
        return 0

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
