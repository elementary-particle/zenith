"""Incremental factorization of immutable per-observer event prefixes."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

from .events import factorize_event
from .schema import numeric_features


@dataclass(frozen=True, slots=True)
class FactorizedEventPrefix:
    categorical_array: object
    numeric_array: object
    event_end: int

    @property
    def token_factors(self):
        """Compatibility tuple; rollout encoding consumes the array directly."""
        return tuple(map(tuple, self.categorical_array.tolist()))

    @property
    def token_numeric(self):
        """Compatibility tuple; rollout encoding consumes the array directly."""
        return tuple(map(tuple, self.numeric_array.tolist()))


@dataclass(slots=True)
class _EventPrefixStorage:
    categorical: object
    numeric: object
    length: int = 0

    @classmethod
    def empty(cls, capacity=16):
        import numpy as np

        return cls(
            np.empty((int(capacity), 10), dtype=np.uint8),
            np.empty((int(capacity), 8), dtype=np.float32),
        )

    def append(self, categorical, numeric):
        if self.length == len(self.categorical):
            import numpy as np

            capacity = max(16, 2 * self.length)
            grown_categorical = np.empty((capacity, 10), dtype=np.uint8)
            grown_numeric = np.empty((capacity, 8), dtype=np.float32)
            grown_categorical[:self.length] = self.categorical[:self.length]
            grown_numeric[:self.length] = self.numeric[:self.length]
            self.categorical = grown_categorical
            self.numeric = grown_numeric
        self.categorical[self.length] = categorical
        self.numeric[self.length] = numeric
        self.length += 1

    def snapshot(self, event_end):
        categorical = self.categorical[:self.length].view()
        numeric = self.numeric[:self.length].view()
        categorical.flags.writeable = False
        numeric.flags.writeable = False
        return FactorizedEventPrefix(categorical, numeric, int(event_end))


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

    def __init__(self, max_entries: int = 4096, *, encoder=None):
        if int(max_entries) <= 0:
            raise ValueError("event prefix cache must allow at least one entry")
        self.max_entries = int(max_entries)
        self._encoder = factorize_event if encoder is None else encoder
        self._encoder_is_factorized = encoder is None
        self._entries: OrderedDict[
            tuple, tuple[object, _EventPrefixStorage, FactorizedEventPrefix]
        ] = (
            OrderedDict()
        )
        self.stats = EventPrefixCacheStats()

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    def encode(self, store, *, observer: int):
        observer = int(observer)
        key = (
            int(store.environment_id),
            int(store.episode_generation),
            observer,
            id(store),
        )
        event_end = len(store.rows)
        cached = self._entries.pop(key, None)
        valid = cached is not None and cached[0] is store
        storage = cached[1] if valid else None
        previous = cached[2] if valid else None
        self.stats.requests += 1
        if previous is None or previous.event_end > event_end:
            storage = _EventPrefixStorage.empty()
            previous = storage.snapshot(0)
            self.stats.rebuilds += 1
        reused = len(previous.categorical_array)
        if previous.event_end < event_end:
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
                    # Allocate new backing storage so previously returned
                    # immutable snapshots cannot change at a kyoku reset.
                    storage = _EventPrefixStorage.empty()
                    reused = 0
                token = self._encoder(row, observer=observer)
                if token is not None:
                    if self._encoder_is_factorized:
                        categorical, numeric = token
                    else:
                        categorical = token.categorical()
                        numeric = numeric_features(token)
                    storage.append(categorical, numeric)
                    encoded_count += 1
            current = storage.snapshot(event_end)
            self.stats.extensions += int(previous.event_end > 0)
            self.stats.events_encoded += encoded_count
        else:
            current = previous
        self.stats.tokens_reused += reused

        # Retaining the store makes Python object-id reuse impossible while an
        # entry is live and lets us verify identity on every lookup.
        self._entries[key] = (store, storage, current)
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
