"""Immutable compatible checkpoint-pool snapshots."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json

from ..compatibility import CompatibilitySet


@dataclass(frozen=True, slots=True)
class PoolEntry:
    checkpoint_id: str
    artifact: str
    compatibility: CompatibilitySet
    source_run: str
    source_update: int
    bytes: int = 0
    admitted: bool = True
    retired: bool = False
    pins: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PoolSnapshot:
    entries: tuple[PoolEntry, ...]
    rating_snapshot_id: str | None = None

    @property
    def snapshot_id(self):
        payload = {
            "entries": [
                (entry.checkpoint_id, entry.admitted, entry.retired)
                for entry in self.entries
            ],
            "rating_snapshot_id": self.rating_snapshot_id,
        }
        return sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()

    @property
    def eligible(self): return tuple(entry for entry in self.entries if entry.admitted and not entry.retired)


class CheckpointPool:
    def __init__(self, compatibility=CompatibilitySet()):
        self.compatibility, self._entries = compatibility, {}

    def admit(self, entry):
        self.compatibility.require(entry.compatibility, context=f"pool entry {entry.checkpoint_id}")
        if entry.checkpoint_id in self._entries and self._entries[entry.checkpoint_id] != entry:
            raise ValueError("checkpoint IDs are immutable")
        self._entries[entry.checkpoint_id] = entry

    def retire(self, checkpoint_id):
        entry = self._entries[checkpoint_id]
        self._entries[checkpoint_id] = replace(entry, retired=True)

    def pin(self, checkpoint_id, owner):
        entry = self._entries[checkpoint_id]
        if owner not in entry.pins: self._entries[checkpoint_id] = replace(entry, pins=entry.pins + (owner,))

    def unpin(self, checkpoint_id, owner):
        entry = self._entries[checkpoint_id]
        self._entries[checkpoint_id] = replace(entry, pins=tuple(pin for pin in entry.pins if pin != owner))

    def snapshot(self, rating_snapshot_id=None):
        return PoolSnapshot(tuple(self._entries[key] for key in sorted(self._entries)), rating_snapshot_id)

    def enforce_retention(self, maximum):
        maximum = int(maximum)
        if maximum < 1:
            raise ValueError("checkpoint retention must be positive")
        live = sorted(
            (entry for entry in self._entries.values() if not entry.retired),
            key=lambda entry: (entry.source_update, entry.checkpoint_id),
        )
        for entry in live[:-maximum]:
            if not entry.pins:
                self.retire(entry.checkpoint_id)

    def evaluation_schedule(self, checkpoint_ids, *, anchors=()):
        ids = tuple(dict.fromkeys((*anchors, *checkpoint_ids)))
        if len(ids) < 4: raise ValueError("connected four-player evaluation needs four checkpoints")
        anchor = tuple(ids[:3])
        return tuple(tuple((*anchor, checkpoint)) if checkpoint not in anchor else tuple(ids[:4])
                     for checkpoint in ids)

    def publish_rating_snapshot(self, rating_table):
        payload = json.dumps({key: asdict(value) for key, value in sorted(rating_table.ratings.items())},
                             sort_keys=True, separators=(",", ":"))
        return sha256(payload.encode()).hexdigest()

    def state_dict(self):
        return {
            "compatibility": asdict(self.compatibility),
            "entries": [asdict(entry) for entry in self.snapshot().entries],
        }

    @classmethod
    def from_state_dict(cls, state):
        compatibility = CompatibilitySet.from_mapping(state.get("compatibility", {}))
        pool = cls(compatibility)
        for value in state.get("entries", ()):
            value = dict(value)
            value["compatibility"] = CompatibilitySet.from_mapping(value["compatibility"])
            value["pins"] = tuple(value.get("pins", ()))
            pool.admit(PoolEntry(**value))
        return pool
