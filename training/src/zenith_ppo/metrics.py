"""Canonical JSONL authority and asynchronous TensorBoard projection."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from queue import Queue
from threading import Thread

from .metric_registry import validate


def completed_match_metric_values(outcomes) -> dict[str, float]:
    """Aggregate terminal match telemetry without inventing zero-game samples."""
    outcomes = tuple(outcomes)
    if not outcomes:
        return {}
    first_place_scores = []
    fourth_place_scores = []
    completed_kyoku = []
    for outcome in outcomes:
        ranks = tuple(int(rank) for rank in outcome.ranks)
        if sorted(ranks) != [0, 1, 2, 3]:
            raise ValueError("completed match ranks must be a permutation of 0..3")
        kyoku = int(outcome.completed_kyoku)
        if kyoku <= 0:
            raise ValueError("completed match must contain at least one kyoku")
        first_place_scores.append(int(outcome.scores[ranks.index(0)]))
        fourth_place_scores.append(int(outcome.scores[ranks.index(3)]))
        completed_kyoku.append(kyoku)
    count = len(outcomes)
    return {
        "game/first_place_score_mean": sum(first_place_scores) / (1000.0 * count),
        "game/fourth_place_score_mean": sum(fourth_place_scores) / (1000.0 * count),
        "game/kyoku_per_match_mean": sum(completed_kyoku) / count,
    }


class CanonicalMetrics:
    def __init__(self, path, *, run_id, writer_session):
        self.path = Path(path); self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id, self.writer_session, self.sequence = run_id, writer_session, 0
        self.keys = set()
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                row = json.loads(line); self.sequence = max(self.sequence, row["sequence"] + 1)
                self.keys.add((row["name"], row["axis"], row["step"], row["source"]))

    def commit(self, points):
        records, local = [], set()
        for point in points:
            validate(point.name, point.axis, point.unit, point.window, point.reduction)
            if not math.isfinite(float(point.value)): raise ValueError("canonical metrics must be finite")
            key = (point.name, point.axis, int(point.step), point.source)
            if key in self.keys or key in local: raise ValueError(f"duplicate metric key {key}")
            local.add(key)
            records.append({"run_id": self.run_id, "writer_session": self.writer_session,
                "sequence": self.sequence + len(records), "name": point.name, "axis": point.axis,
                "step": int(point.step), "value": float(point.value), "unit": point.unit,
                "window": point.window, "reduction": point.reduction, "source": point.source,
                "committed_at": datetime.now(timezone.utc).isoformat()})
        with self.path.open("a", encoding="utf-8") as output:
            for record in records: output.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
            output.flush()
            os.fsync(output.fileno())
        self.keys |= local; self.sequence += len(records)
        return records


class TensorBoardProjector:
    def __init__(self, log_dir, *, run_id=None, writer_session=None,
                 max_queue=1024, flush_seconds=30, purge_step=None):
        from torch.utils.tensorboard import SummaryWriter
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        owner = log_dir / "owner.json"
        # A resumed process owns a new writer session but contributes event files
        # to the same logical TensorBoard run.
        identity = {"run_id": run_id}
        if owner.exists() and json.loads(owner.read_text()) != identity:
            raise RuntimeError(f"TensorBoard directory {log_dir} belongs to another session")
        owner.write_text(json.dumps(identity, sort_keys=True), encoding="utf-8")
        self.writer = SummaryWriter(
            str(log_dir), max_queue=max_queue, flush_secs=flush_seconds,
            filename_suffix=".zenith", purge_step=purge_step,
        )
        self.writer.add_custom_scalars({
            "Rollout progress": {
                "win rate vs conservative bot": ["Margin", [
                    "rollout_rating/win_rate_vs_conservative_bot",
                    "rollout_rating/lower_win_rate_vs_conservative_bot",
                    "rollout_rating/upper_win_rate_vs_conservative_bot",
                ]],
            },
            "Match outcomes": {
                "final scores (thousand points)": ["Multiline", [
                    "game/first_place_score_mean",
                    "game/fourth_place_score_mean",
                ]],
                "match length": ["Multiline", [
                    "game/kyoku_per_match_mean",
                ]],
            },
        })
        self.queue, self.failure = Queue(max_queue), None
        self.thread = Thread(target=self._run, daemon=True); self.thread.start()

    def enqueue(self, records):
        if self.failure is not None: raise RuntimeError("TensorBoard writer failed") from self.failure
        self.queue.put(tuple(records))

    def _run(self):
        try:
            while True:
                records = self.queue.get()
                if records is None: return
                for row in records:
                    if row.get("kind", "scalar") == "histogram":
                        self.writer.add_histogram(row["name"], row["values"], row["step"])
                    elif row.get("kind") == "text":
                        self.writer.add_text(row["name"], row["text"], row["step"])
                    else:
                        self.writer.add_scalar(row["name"], row["value"], row["step"],
                                               new_style=True, double_precision=True)
        except Exception as exc: self.failure = exc

    def close(self):
        self.queue.put(None); self.thread.join(); self.writer.flush(); self.writer.close()
        if self.failure: raise RuntimeError("TensorBoard writer failed") from self.failure

    def enqueue_histogram(self, name, values, step, *, max_elements, max_bytes, rng):
        sampled = histogram_sample(
            values,
            max_elements=max_elements,
            max_bytes=max_bytes,
            rng=rng,
        )
        self.enqueue(({
            "kind": "histogram", "name": name, "values": sampled, "step": int(step)
        },))
        return {"sampled": int(sampled.size), "total": int(len(values))}

    def enqueue_text(self, name, text, step):
        self.enqueue(({"kind": "text", "name": str(name), "text": str(text),
                       "step": int(step)},))


def histogram_sample(values, *, max_elements, max_bytes, rng):
    import numpy as np
    flat = np.asarray(values).reshape(-1)
    count = min(len(flat), int(max_elements), int(max_bytes) // max(1, flat.dtype.itemsize))
    if count == len(flat): return flat.copy()
    indices = np.sort(rng.choice(len(flat), size=count, replace=False))
    return flat[indices]
