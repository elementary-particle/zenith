"""Gap-detecting immutable event stores."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
from typing import Any


def _canonical_row(row: dict) -> bytes:
    """Return the stable wire representation used by the rolling store digest."""
    return json.dumps(
        row, sort_keys=True, separators=(",", ":"), default=list
    ).encode()


@dataclass
class EventStore:
    environment_id: int
    episode_generation: int
    rows: list[dict] = field(default_factory=list)
    sealed: bool = False
    _digest: Any = field(init=False, repr=False, compare=False)
    _store_id: str | None = field(init=False, default=None, repr=False, compare=False)

    def __post_init__(self):
        # A store digest represents its identity and every ordered row. Hashing
        # each row once on ingestion makes later observation materialization
        # independent of match age. Restored stores rebuild this state once.
        initial_rows = self.rows
        initially_sealed = self.sealed
        self.rows = []
        self.sealed = False
        self._digest = sha256()
        self._digest.update(b"zenith-event-store-v2\0")
        self._digest.update(
            f"{int(self.environment_id)}:{int(self.episode_generation)}\0".encode()
        )
        self.append(initial_rows)
        self.sealed = initially_sealed

    @property
    def next_sequence(self): return len(self.rows)

    @property
    def store_id(self):
        if self._store_id is None:
            self._store_id = self._digest.hexdigest()
        return self._store_id

    def append(self, rows):
        if self.sealed:
            raise RuntimeError("event store is sealed")
        for row in rows:
            if int(row["environment_id"]) != self.environment_id \
                    or int(row["episode_generation"]) != self.episode_generation:
                raise ValueError("event identity does not match store")
            if int(row["sequence"]) != self.next_sequence:
                raise ValueError(f"event gap: expected {self.next_sequence}, got {row['sequence']}")
            stored = dict(row)
            payload = _canonical_row(stored)
            self._digest.update(len(payload).to_bytes(8, "little"))
            self._digest.update(payload)
            self._store_id = None
            self.rows.append(stored)


class HistoryRegistry:
    def __init__(self):
        self._stores = {}
        self._active_generations = {}

    def get(self, environment_id, generation):
        key = (int(environment_id), int(generation))
        existing = self._stores.get(key)
        if existing is not None:
            return existing
        previous_generation = self._active_generations.get(key[0])
        if previous_generation is not None:
            self._stores[(key[0], previous_generation)].sealed = True
        store = EventStore(*key)
        self._stores[key] = store
        self._active_generations[key[0]] = key[1]
        return store

    def state_dict(self):
        return {
            "stores": [
                {
                    "environment_id": store.environment_id,
                    "episode_generation": store.episode_generation,
                    "rows": tuple(dict(row) for row in store.rows),
                    "sealed": store.sealed,
                }
                for _, store in sorted(self._stores.items())
            ]
        }

    @classmethod
    def from_state_dict(cls, state):
        registry = cls()
        for value in state.get("stores", ()):
            store = EventStore(
                int(value["environment_id"]),
                int(value["episode_generation"]),
                [dict(row) for row in value.get("rows", ())],
                bool(value.get("sealed", False)),
            )
            registry._stores[(store.environment_id, store.episode_generation)] = store
            if not store.sealed:
                previous = registry._active_generations.setdefault(
                    store.environment_id, store.episode_generation
                )
                if previous != store.episode_generation:
                    raise ValueError(
                        "multiple active event generations for one environment"
                    )
        return registry

    def retain(self, identities):
        keep = {
            (int(environment_id), int(generation))
            for environment_id, generation in identities
        }
        self._stores = {
            key: store for key, store in self._stores.items() if key in keep
        }
        self._active_generations = {
            environment_id: generation
            for environment_id, generation in self._active_generations.items()
            if (environment_id, generation) in keep
        }
