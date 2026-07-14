"""Count/byte bounded pin-aware LRU model residency."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass


@dataclass
class ResidentModel:
    checkpoint_id: str
    model: object
    bytes: int
    generation: int
    pins: int = 0


class ModelResidency:
    def __init__(self, *, max_models=4, max_bytes=1 << 32):
        self.max_models, self.max_bytes, self.bytes = max_models, max_bytes, 0
        self.models, self.next_generation = OrderedDict(), 1
        self.loads = self.evictions = 0

    def get(self, checkpoint_id, loader):
        if checkpoint_id in self.models:
            self.models.move_to_end(checkpoint_id); return self.models[checkpoint_id]
        model, size = loader(checkpoint_id)
        entry = ResidentModel(checkpoint_id, model, int(size), self.next_generation)
        self.next_generation += 1; self.loads += 1
        self.models[checkpoint_id] = entry; self.bytes += entry.bytes
        self._evict(); return entry

    def _evict(self):
        while len(self.models) > self.max_models or self.bytes > self.max_bytes:
            victim = next((key for key, value in self.models.items() if value.pins == 0), None)
            if victim is None: raise RuntimeError("residency limits exceeded by pinned models")
            removed = self.models.pop(victim); self.bytes -= removed.bytes; self.evictions += 1

    def pin(self, checkpoint_id): self.models[checkpoint_id].pins += 1
    def unpin(self, checkpoint_id):
        entry = self.models[checkpoint_id]
        if entry.pins <= 0:
            raise RuntimeError(f"model {checkpoint_id!r} is not pinned")
        entry.pins -= 1

    def grouped(self, checkpoint_ids, loader):
        """Resolve each distinct model once and return stable checkpoint groups."""
        groups = {}
        for index, checkpoint_id in enumerate(checkpoint_ids):
            groups.setdefault(checkpoint_id, []).append(index)
        return tuple(
            (self.get(checkpoint_id, loader), tuple(indices))
            for checkpoint_id, indices in groups.items()
        )
