"""Incremental factorization of immutable per-observer event prefixes."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

from .events import encode_event
from .schema import numeric_features


@dataclass(frozen=True, slots=True)
class FactorizedEventPrefix:
    token_factors: tuple[tuple[int, ...], ...]
    token_numeric: tuple[tuple[float, ...], ...]
    categorical_array: object
    numeric_array: object
    event_end: int


@dataclass(slots=True)
class EventPrefixCacheStats:
    requests: int = 0
    rebuilds: int = 0
    extensions: int = 0
    events_encoded: int = 0
    tokens_reused: int = 0
    evictions: int = 0


class EventPrefixCache:
    """Bounded, rebuildable cache for append-only native event stores.

    Entries include the concrete store identity because a restored registry may
    replace a store while retaining its environment/generation coordinates.
    """

    def __init__(self, max_entries: int = 4096, *, encoder=encode_event):
        if int(max_entries) <= 0:
            raise ValueError("event prefix cache must allow at least one entry")
        self.max_entries = int(max_entries)
        self._encoder = encoder
        self._entries: OrderedDict[tuple, tuple[object, FactorizedEventPrefix]] = (
            OrderedDict()
        )
        self.stats = EventPrefixCacheStats()

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    def encode(self, store, *, observer: int):
        import numpy as np

        observer = int(observer)
        key = (
            int(store.environment_id),
            int(store.episode_generation),
            observer,
            id(store),
        )
        event_end = len(store.rows)
        cached = self._entries.pop(key, None)
        previous = cached[1] if cached is not None and cached[0] is store else None
        self.stats.requests += 1
        if previous is None or previous.event_end > event_end:
            previous = FactorizedEventPrefix(
                (),
                (),
                np.empty((0, 10), dtype=np.uint8),
                np.empty((0, 8), dtype=np.float32),
                0,
            )
            self.stats.rebuilds += 1
        reused = len(previous.token_factors)
        if previous.event_end < event_end:
            factors = list(previous.token_factors)
            numeric = list(previous.token_numeric)
            encoded_count = 0
            for expected, row in enumerate(
                store.rows[previous.event_end:event_end], previous.event_end
            ):
                if int(row["episode_generation"]) != int(store.episode_generation):
                    raise ValueError("event generation changed within history")
                sequence = int(row["sequence"])
                if sequence != expected:
                    raise ValueError(
                        f"event sequence gap: expected {expected}, got {sequence}"
                    )
                if int(row["kind"]) == 2:
                    factors.clear()
                    numeric.clear()
                    reused = 0
                token = self._encoder(row, observer=observer)
                if token is not None:
                    factors.append(token.categorical())
                    numeric.append(numeric_features(token))
                    encoded_count += 1
            current = FactorizedEventPrefix(
                tuple(factors),
                tuple(numeric),
                np.asarray(factors, dtype=np.uint8).reshape(-1, 10),
                np.asarray(numeric, dtype=np.float32).reshape(-1, 8),
                event_end,
            )
            self.stats.extensions += int(previous.event_end > 0)
            self.stats.events_encoded += encoded_count
        else:
            current = previous
        self.stats.tokens_reused += reused

        # Retaining the store makes Python object-id reuse impossible while an
        # entry is live and lets us verify identity on every lookup.
        self._entries[key] = (store, current)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)
            self.stats.evictions += 1
        return current

    def retain(self, identities) -> None:
        keep = {
            (int(environment_id), int(generation))
            for environment_id, generation in identities
        }
        self._entries = OrderedDict(
            (key, entry)
            for key, entry in self._entries.items()
            if key[:2] in keep
        )
