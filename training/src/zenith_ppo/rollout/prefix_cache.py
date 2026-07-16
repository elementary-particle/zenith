"""Bounded inference-only event-prefix cache metadata."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass


@dataclass
class PrefixEntry:
    key: tuple
    event_end: int
    value: object
    bytes: int


class PrefixCache:
    def __init__(self, max_bytes=1 << 30):
        self.max_bytes, self.bytes = int(max_bytes), 0
        self.entries = OrderedDict()

    def get(self, key, event_end):
        entry = self.entries.get(key)
        if entry is None or entry.event_end > event_end: return None
        self.entries.move_to_end(key); return entry

    def put(self, key, event_end, value, size):
        if getattr(value, "requires_grad", False): raise ValueError("prefix cache is inference-only")
        old = self.entries.pop(key, None)
        if old: self.bytes -= old.bytes
        entry = PrefixEntry(key, int(event_end), value, int(size))
        self.entries[key] = entry; self.bytes += entry.bytes
        while self.bytes > self.max_bytes and self.entries:
            _, removed = self.entries.popitem(last=False); self.bytes -= removed.bytes

    def invalidate(self, predicate=None):
        keys = list(self.entries) if predicate is None else [key for key in self.entries if predicate(key)]
        for key in keys: self.bytes -= self.entries.pop(key).bytes
