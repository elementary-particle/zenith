"""Low-perturbation CPU and CUDA attribution for one training update."""

from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter


class StageProfiler:
    def __init__(self, *, enabled=False, device="cpu"):
        self.enabled = bool(enabled)
        self.device = str(device)
        self._seconds: dict[str, float] = {}
        self._exclusive_seconds: dict[str, float] = {}
        self._gpu_seconds: dict[str, float] = {}
        self._calls: dict[str, int] = {}
        self._stack: list[list[object]] = []
        self._pending_cuda: list[tuple[str, object, object]] = []
        self._observations: dict[str, list[float]] = {}

    def observe(self, name, value):
        if self.enabled:
            self._observations.setdefault(str(name), []).append(float(value))

    @contextmanager
    def measure(self, name):
        if not self.enabled:
            yield
            return
        cuda_events = None
        if self.device == "cuda":
            import torch
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            cuda_events = start_event, end_event
        started = perf_counter()
        frame = [name, started, 0.0]
        self._stack.append(frame)
        try:
            yield
        finally:
            ended = perf_counter()
            popped = self._stack.pop()
            if popped is not frame:
                raise RuntimeError("stage profiler nesting was corrupted")
            elapsed = ended - started
            exclusive = max(0.0, elapsed - float(frame[2]))
            if self._stack:
                self._stack[-1][2] = float(self._stack[-1][2]) + elapsed
            self._seconds[name] = self._seconds.get(name, 0.0) + elapsed
            self._exclusive_seconds[name] = (
                self._exclusive_seconds.get(name, 0.0) + exclusive
            )
            self._calls[name] = self._calls.get(name, 0) + 1
            if cuda_events is not None:
                cuda_events[1].record()
                self._pending_cuda.append((name, *cuda_events))

    def _resolve_cuda_events(self):
        if not self._pending_cuda:
            return
        import torch

        # Synchronize once per report instead of twice around every measured
        # region.  This keeps profiling from turning an asynchronous update
        # into thousands of serialized launches.
        torch.cuda.synchronize()
        for name, start, end in self._pending_cuda:
            self._gpu_seconds[name] = self._gpu_seconds.get(name, 0.0) + (
                start.elapsed_time(end) / 1000.0
            )
        self._pending_cuda.clear()

    def snapshot(self):
        if self.enabled and self.device == "cuda":
            self._resolve_cuda_events()
        rows = [
            {
                "stage": name,
                "seconds": seconds,
                "exclusive_seconds": self._exclusive_seconds[name],
                "gpu_seconds": self._gpu_seconds.get(name, 0.0),
                "calls": self._calls[name],
            }
            for name, seconds in self._seconds.items()
        ]
        rows.sort(key=lambda row: (-row["exclusive_seconds"], row["stage"]))
        attributed_total = sum(row["exclusive_seconds"] for row in rows)
        for row in rows:
            row["percent_attributed"] = (
                100.0 * row["exclusive_seconds"] / attributed_total
                if attributed_total else 0.0
            )
        result = {
            "synchronized_at_snapshot": self.enabled and self.device == "cuda",
            "device": self.device,
            "attributed_seconds": attributed_total,
            "gpu_attributed_seconds": sum(row["gpu_seconds"] for row in rows),
            "stages": rows,
            "observations": {
                name: {
                    "count": len(values),
                    "mean": sum(values) / len(values),
                    "minimum": min(values),
                    "maximum": max(values),
                    "p50": sorted(values)[len(values) // 2],
                    "p95": sorted(values)[min(
                        len(values) - 1, int(0.95 * len(values))
                    )],
                }
                for name, values in sorted(self._observations.items()) if values
            },
        }
        if self.enabled and self.device == "cuda":
            import torch
            result["peak_memory_bytes"] = int(torch.cuda.max_memory_allocated())
            result["peak_reserved_memory_bytes"] = int(
                torch.cuda.max_memory_reserved()
            )
        elif self.enabled:
            import resource
            # Linux reports KiB for ru_maxrss.
            result["peak_memory_bytes"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
        return result
