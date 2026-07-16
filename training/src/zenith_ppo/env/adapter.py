"""One-call ingestion adapter for the immutable native batched env."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from ..types import DecisionBinding
from .history import HistoryRegistry
from .projection import validate_event_payload


@dataclass(frozen=True, slots=True)
class EnvBatch:
    transition: object
    arrays: Mapping[str, object]
    decisions: tuple[object, ...]
    bindings: tuple[DecisionBinding, ...]


@dataclass(frozen=True, slots=True)
class _MergedTransition:
    """Minimal transition facade used after resetting completed environments."""

    states: tuple[object, ...]
    events: tuple[object, ...]


class EnvAdapter:
    def __init__(self, env):
        self.env = env
        self.histories = HistoryRegistry()

    def reset(self, environment_ids: Sequence[int]) -> EnvBatch:
        return self._consume(self.env.reset(list(map(int, environment_ids))))

    def step(self, actions: Sequence[object]) -> EnvBatch:
        if not actions:
            raise ValueError("joint action submission must not be empty")
        return self._consume(self.env.step(tuple(actions)))

    def restore(self, snapshots: Mapping[int, bytes]) -> EnvBatch:
        return self._consume(self.env.restore(dict(snapshots)))

    def snapshot(self, environment_ids: Sequence[int]) -> dict[int, bytes]:
        return {
            int(environment_id): bytes(payload)
            for environment_id, payload in self.env.snapshot(
                list(map(int, environment_ids))
            ).items()
        }

    def inspect(self, environment_ids: Sequence[int], *, privileged=None) -> EnvBatch:
        return self._consume(
            self.env.inspect(list(map(int, environment_ids)), privileged=privileged)
        )

    def reset_completed(self, batch: EnvBatch) -> EnvBatch:
        completed = tuple(
            state.environment_id
            for state in batch.transition.states
            if int(state.lifecycle) == 3
        )
        failed = tuple(
            state.environment_id
            for state in batch.transition.states
            if int(state.lifecycle) == 4
        )
        if failed:
            raise RuntimeError(f"native env failed for environments {failed}")
        if not completed:
            return batch
        restarted = self.reset(completed)
        live_states = tuple(
            state for state in batch.transition.states if int(state.lifecycle) != 3
        )
        states = tuple(sorted(
            (*live_states, *restarted.transition.states),
            key=lambda state: state.environment_id,
        ))
        decisions = tuple(decision for state in states for decision in state.decisions)
        bindings = tuple(
            DecisionBinding(
                decision.environment_id,
                decision.episode_generation,
                decision.frame_id,
                decision.seat,
            )
            for decision in decisions
        )
        transition = _MergedTransition(
            states,
            tuple((*batch.transition.events, *restarted.transition.events)),
        )
        return EnvBatch(transition, {}, decisions, bindings)

    @staticmethod
    def select(batch: EnvBatch, action_indices: Sequence[int]) -> tuple[object, ...]:
        if len(action_indices) != len(batch.decisions):
            raise ValueError("one action index is required for every eligible decision")
        selected = []
        for decision, index in zip(batch.decisions, action_indices, strict=True):
            index = int(index)
            if not 0 <= index < len(decision.actions):
                raise IndexError(f"action index {index} is out of range")
            selected.append(decision.actions[index])
        return tuple(selected)

    def _consume(self, transition) -> EnvBatch:
        arrays = transition.as_numpy()
        payload = arrays["event_payload"]
        rows_by_store: dict[tuple[int, int], list[dict]] = {}
        for index in range(len(arrays["event_kind"])):
            start = int(arrays["event_payload_offsets"][index])
            end = int(arrays["event_payload_offsets"][index + 1])
            row = {
                "environment_id": int(arrays["event_environment_id"][index]),
                "episode_generation": int(arrays["event_episode_generation"][index]),
                "sequence": int(arrays["event_sequence"][index]),
                "kind": int(arrays["event_kind"][index]),
                "actor_seat": int(arrays["event_actor_seat"][index]),
                "target_seat": int(arrays["event_target_seat"][index]),
                "visibility_mask": int(arrays["event_visibility_mask"][index]),
                "args": tuple(int(value) for value in arrays["event_args"][index]),
                "payload": bytes(payload[start:end]),
            }
            validate_event_payload(row)
            key = (row["environment_id"], row["episode_generation"])
            rows_by_store.setdefault(key, []).append(row)
        for key, rows in rows_by_store.items():
            self.histories.get(*key).append(rows)

        decisions = tuple(
            decision for state in transition.states for decision in state.decisions
        )
        bindings = tuple(
            DecisionBinding(
                decision.environment_id,
                decision.episode_generation,
                decision.frame_id,
                decision.seat,
            )
            for decision in decisions
        )
        return EnvBatch(transition, arrays, decisions, bindings)
