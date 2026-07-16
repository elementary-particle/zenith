"""Opt-in synchronized wall-time attribution for one training update."""

from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter


class StageProfiler:
    def __init__(self, *, enabled=False, device="cpu"):
        self.enabled = bool(enabled)
        self.device = str(device)
        self._seconds: dict[str, float] = {}
        self._calls: dict[str, int] = {}

    def _synchronize(self):
        if self.enabled and self.device == "cuda":
            import torch

            torch.cuda.synchronize()

    @contextmanager
    def measure(self, name):
        if not self.enabled:
            yield
            return
        self._synchronize()
        started = perf_counter()
        try:
            yield
        finally:
            self._synchronize()
            self._seconds[name] = self._seconds.get(name, 0.0) + perf_counter() - started
            self._calls[name] = self._calls.get(name, 0) + 1

    def snapshot(self):
        rows = [
            {"stage": name, "seconds": seconds, "calls": self._calls[name]}
            for name, seconds in self._seconds.items()
        ]
        rows.sort(key=lambda row: (-row["seconds"], row["stage"]))
        attributed_total = sum(row["seconds"] for row in rows)
        for row in rows:
            row["percent_attributed"] = (
                100.0 * row["seconds"] / attributed_total if attributed_total else 0.0
            )
        result = {
            "synchronized": self.enabled and self.device == "cuda",
            "device": self.device,
            "attributed_seconds": attributed_total,
            "stages": rows,
        }
        if self.enabled and self.device == "cuda":
            import torch
            result["peak_memory_bytes"] = int(torch.cuda.max_memory_allocated())
        elif self.enabled:
            import resource
            # Linux reports KiB for ru_maxrss.
            result["peak_memory_bytes"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
        return result
