"""One-call ingestion adapter for the immutable native batched env."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Mapping, Sequence

from ..types import ActionSpaceBinding
from .history import HistoryRegistry
from .projection import validate_event_payload


@dataclass(frozen=True, slots=True)
class EnvBatch:
    transition: object
    arrays: Mapping[str, object]
    action_spaces: tuple[object, ...]
    bindings: tuple[ActionSpaceBinding, ...]

    @staticmethod
    def merge(*batches: "EnvBatch") -> "EnvBatch":
        """Combine disjoint native results into one inference-ready frame.

        ``step`` and ``advance`` must remain separate native calls, but their
        returned environments are independent.  Keeping them in one Python
        batch prevents permanently fragmenting the rollout inference queue as
        environments alternate between decision and automatic frames.
        """
        batches = tuple(batch for batch in batches if batch is not None)
        if not batches:
            raise ValueError("cannot merge an empty environment batch")
        if len(batches) == 1:
            return batches[0]
        states = tuple(
            state for batch in batches for state in batch.transition.states
        )
        environment_ids = [int(state.environment_id) for state in states]
        if len(environment_ids) != len(set(environment_ids)):
            raise ValueError("merged environment batches must be disjoint")
        states = tuple(sorted(states, key=lambda state: int(state.environment_id)))
        events = tuple(sorted(
            (event for batch in batches for event in batch.transition.events),
            key=lambda event: (
                int(event.environment_id),
                int(event.episode_generation),
                int(event.sequence),
            ),
        ))
        transition = SimpleNamespace(
            states=states,
            events=events,
            transition_id=max(
                int(batch.transition.transition_id) for batch in batches
            ),
        )
        action_spaces = tuple(
            space for state in states for space in state.action_spaces
        )
        bindings = tuple(
            ActionSpaceBinding(
                int(space.environment_id),
                int(space.episode_generation),
                int(space.frame_id),
                int(space.seat),
            )
            for space in action_spaces
        )
        return EnvBatch(transition, {}, action_spaces, bindings)


class EnvAdapter:
    def __init__(self, env):
        self.env = env
        self.histories = HistoryRegistry()

    def reset(self, environment_ids: Sequence[int]) -> EnvBatch:
        return self._consume(self.env.reset(list(map(int, environment_ids))))

    def step(self, selections: Sequence[object]) -> EnvBatch:
        if not selections:
            raise ValueError("joint selection submission must not be empty")
        return self._consume(self.env.step(tuple(selections)))

    def advance(self, environment_ids: Sequence[int]) -> EnvBatch:
        if not environment_ids:
            raise ValueError("automatic advancement must not be empty")
        return self._consume(self.env.advance(list(map(int, environment_ids))))

    def load_hanchan(self, values: Sequence[object]) -> EnvBatch:
        if not values:
            raise ValueError("hanchan load must not be empty")
        return self._consume(self.env.load_hanchan(tuple(values)))

    def apply_events(self, values: Sequence[object]) -> EnvBatch:
        if not values:
            raise ValueError("replay event application must not be empty")
        return self._consume(self.env.apply_events(tuple(values)))

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

    def fork_privileged_wall(
        self,
        source_environment_id: int,
        branches: Sequence[tuple[int, int]],
    ) -> EnvBatch:
        """Fork one root into paired search branches and clone public history."""
        branches = tuple((int(target), int(key)) for target, key in branches)
        if not branches:
            raise ValueError("search fork requires at least one branch")
        transition = self.env.fork_privileged_wall(
            int(source_environment_id), branches
        )
        generations = {int(state.episode_generation) for state in transition.states}
        if len(generations) != 1:
            raise AssertionError("search branches changed episode generation")
        self.histories.fork(
            int(source_environment_id),
            generations.pop(),
            (target for target, _ in branches),
        )
        return self._consume(transition)

    def fork_public_information(
        self,
        source_environment_id: int,
        observer_seat: int,
        branches: Sequence[tuple[int, int]],
    ) -> EnvBatch:
        """Fork a self-turn root with public-hidden hands and wall resampled."""
        branches = tuple((int(target), int(key)) for target, key in branches)
        if not branches:
            raise ValueError("public search fork requires at least one branch")
        transition = self.env.fork_public_information(
            int(source_environment_id), int(observer_seat), branches
        )
        generations = {int(state.episode_generation) for state in transition.states}
        if len(generations) != 1:
            raise AssertionError("search branches changed episode generation")
        self.histories.fork(
            int(source_environment_id),
            generations.pop(),
            (target for target, _ in branches),
        )
        return self._consume(transition)

    def fork_search_state(
        self,
        source_environment_id: int,
        target_environment_ids: Sequence[int],
    ) -> EnvBatch:
        """Clone an already sampled search branch and its public history."""
        targets = tuple(map(int, target_environment_ids))
        if not targets:
            raise ValueError("search state fork requires at least one branch")
        transition = self.env.fork_search_state(
            int(source_environment_id), targets
        )
        generations = {int(state.episode_generation) for state in transition.states}
        if len(generations) != 1:
            raise AssertionError("search branches changed episode generation")
        self.histories.fork(
            int(source_environment_id), generations.pop(), targets,
        )
        return self._consume(transition)

    @staticmethod
    def select(batch: EnvBatch, candidate_indices: Sequence[int]) -> tuple[object, ...]:
        if len(candidate_indices) != len(batch.action_spaces):
            raise ValueError("one candidate index is required for every action space")
        selected = []
        for space, index in zip(batch.action_spaces, candidate_indices, strict=True):
            index = int(index)
            if not 0 <= index < len(space.candidates):
                raise IndexError(f"candidate index {index} is out of range")
            selected.append(space.candidates[index].select())
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

        action_spaces = tuple(
            space for state in transition.states for space in state.action_spaces
        )
        bindings = tuple(
            ActionSpaceBinding(
                space.environment_id,
                space.episode_generation,
                space.frame_id,
                space.seat,
            )
            for space in action_spaces
        )
        return EnvBatch(transition, arrays, action_spaces, bindings)
