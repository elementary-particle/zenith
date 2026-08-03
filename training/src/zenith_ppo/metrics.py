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


# TensorBoard is an operational dashboard, not a mirror of the canonical audit
# log. Keep this list deliberately small: detailed ablation diagnostics remain
# available in metrics/canonical.jsonl without creating hundreds of mostly-zero
# or redundant TensorBoard series.
TENSORBOARD_SCALARS = frozenset(
    {
        "ppo/policy_loss",
        "ppo/critic_loss",
        "ppo/entropy",
        "ppo/entropy_efficiency",
        "ppo/pre_update_approximate_kl",
        "ppo/post_update_approximate_kl",
        "ppo/kl_coefficient",
        "ppo/magnet_kl",
        "ppo/magnet_kl_coefficient",
        "ppo/magnet_ema_tau",
        "ppo/magnet_parameter_rms_distance",
        "ppo/magnet_relative_parameter_rms_distance",
        "ppo/clip_fraction",
        "ppo/actor_optimization_fraction",
        "ppo/critic_optimization_fraction",
        "ppo/actor_gradient_norm",
        "ppo/critic_gradient_norm",
        "ppo/actor_gradient_clip_fraction",
        "ppo/critic_gradient_clip_fraction",
        "critic/boundary_order_cross_entropy",
        "critic/boundary_order_accuracy",
        "critic/boundary_rank_brier",
        "critic/match_rank_explained_variance",
        "rollout/current_kyoku_advantage_std",
        "rollout/policy_advantage_std",
        "rollout/call_opportunity_selected_call_rate",
        "rollout/riichi_opportunity_selected_riichi_rate",
        "curriculum/progress",
        "curriculum/actor_learning_rate",
        "curriculum/critic_learning_rate",
        "performance/model_queries_per_second",
        "performance/automatic_resolution_fraction",
        "game/player_win_rate",
        "game/player_deal_in_rate",
        "game/player_riichi_rate",
        "game/player_calling_rate",
        "game/player_tsumo_rate",
        "game/player_dama_rate",
        "game/player_average_winning_points",
        "game/player_average_deal_in_points",
        "game/exhaustive_ryukyoku_rate",
        "game/player_bankrupt_rate",
        "game/player_average_turns_before_winning",
    }
)


def tensorboard_records(records):
    """Select stable dashboard scalars from canonical metric records."""
    return tuple(row for row in records if row.get("name") in TENSORBOARD_SCALARS)


class CanonicalMetrics:
    def __init__(self, path, *, run_id, writer_session, resume_sequence=None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id, self.writer_session, self.sequence = run_id, writer_session, 0
        self.keys = set()
        if self.path.exists():
            lines = self.path.read_text(encoding="utf-8").splitlines(keepends=True)
            rows = [json.loads(line) for line in lines]
            for expected, row in enumerate(rows):
                if int(row["sequence"]) != expected:
                    raise ValueError("canonical metric sequence is not contiguous")
            if resume_sequence is not None:
                cursor = int(resume_sequence)
                if cursor < 0 or cursor > len(rows):
                    raise ValueError(
                        f"checkpoint metric cursor {cursor} is outside canonical log "
                        f"with {len(rows)} records"
                    )
                if cursor < len(rows):
                    self._truncate(lines, cursor)
                    rows = rows[:cursor]
            for row in rows:
                self.sequence = row["sequence"] + 1
                self.keys.add((row["name"], row["axis"], row["step"], row["source"]))

    def _truncate(self, lines, count):
        temporary = self.path.with_name(f".{self.path.name}.resume.tmp")
        with temporary.open("w", encoding="utf-8") as output:
            output.writelines(lines[:count])
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, self.path)
        descriptor = os.open(self.path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def records(self):
        if not self.path.exists():
            return
        with self.path.open(encoding="utf-8") as source:
            for line in source:
                yield json.loads(line)

    def commit(self, points):
        records, local = [], set()
        for point in points:
            validate(point.name, point.axis, point.unit, point.window, point.reduction)
            if not math.isfinite(float(point.value)):
                raise ValueError("canonical metrics must be finite")
            key = (point.name, point.axis, int(point.step), point.source)
            if key in self.keys or key in local:
                raise ValueError(f"duplicate metric key {key}")
            local.add(key)
            records.append(
                {
                    "run_id": self.run_id,
                    "writer_session": self.writer_session,
                    "sequence": self.sequence + len(records),
                    "name": point.name,
                    "axis": point.axis,
                    "step": int(point.step),
                    "value": float(point.value),
                    "unit": point.unit,
                    "window": point.window,
                    "reduction": point.reduction,
                    "source": point.source,
                    "committed_at": datetime.now(timezone.utc).isoformat(),
                }
            )
        with self.path.open("a", encoding="utf-8") as output:
            for record in records:
                output.write(
                    json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
                )
            output.flush()
            os.fsync(output.fileno())
        self.keys |= local
        self.sequence += len(records)
        return records


class TensorBoardProjector:
    def __init__(
        self,
        log_dir,
        *,
        run_id=None,
        writer_session=None,
        max_queue=1024,
        flush_seconds=30,
        purge_step=None,
        reset=False,
    ):
        from torch.utils.tensorboard import SummaryWriter

        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        owner = log_dir / "owner.json"
        # A resumed process owns a new writer session but contributes event files
        # to the same logical TensorBoard run.
        identity = {"run_id": run_id}
        if owner.exists() and json.loads(owner.read_text()) != identity:
            raise RuntimeError(
                f"TensorBoard directory {log_dir} belongs to another session"
            )
        owner.write_text(json.dumps(identity, sort_keys=True), encoding="utf-8")
        if reset:
            for event_file in log_dir.glob("events.out.tfevents.*"):
                event_file.unlink()
        self.writer = SummaryWriter(
            str(log_dir),
            max_queue=max_queue,
            flush_secs=flush_seconds,
            filename_suffix=".zenith",
            purge_step=purge_step,
        )
        self.writer.add_custom_scalars(
            {
                "Policy optimization": {
                    "policy KL": [
                        "Multiline",
                        [
                            "ppo/pre_update_approximate_kl",
                            "ppo/post_update_approximate_kl",
                        ],
                    ],
                    "optimizer activity": [
                        "Multiline",
                        [
                            "ppo/actor_optimization_fraction",
                            "ppo/critic_optimization_fraction",
                        ],
                    ],
                    "gradient norms": [
                        "Multiline",
                        [
                            "ppo/actor_gradient_norm",
                            "ppo/critic_gradient_norm",
                        ],
                    ],
                },
                "Policy regularization": {
                    "magnet KL": [
                        "Multiline",
                        [
                            "ppo/magnet_kl",
                        ],
                    ],
                    "magnet lag": [
                        "Multiline",
                        [
                            "ppo/magnet_parameter_rms_distance",
                            "ppo/magnet_relative_parameter_rms_distance",
                        ],
                    ],
                    "magnet schedule": [
                        "Multiline",
                        [
                            "ppo/magnet_kl_coefficient",
                            "ppo/magnet_ema_tau",
                        ],
                    ],
                },
                "Critic quality": {
                    "fresh-rollout explained variance": [
                        "Multiline",
                        [
                            "critic/match_rank_explained_variance",
                        ],
                    ],
                    "boundary rank": [
                        "Multiline",
                        [
                            "critic/boundary_order_accuracy",
                            "critic/boundary_rank_brier",
                        ],
                    ],
                },
                "Gameplay": {
                    "player rates per kyoku": [
                        "Multiline",
                        [
                            "game/player_win_rate",
                            "game/player_deal_in_rate",
                            "game/player_riichi_rate",
                            "game/player_calling_rate",
                        ],
                    ],
                    "win composition": [
                        "Multiline",
                        [
                            "game/player_tsumo_rate",
                            "game/player_dama_rate",
                        ],
                    ],
                    "point outcomes": [
                        "Multiline",
                        [
                            "game/player_average_winning_points",
                            "game/player_average_deal_in_points",
                        ],
                    ],
                    "terminal and timing rates": [
                        "Multiline",
                        [
                            "game/exhaustive_ryukyoku_rate",
                            "game/player_bankrupt_rate",
                            "game/player_average_turns_before_winning",
                        ],
                    ],
                },
            }
        )
        self.queue, self.failure = Queue(max_queue), None
        self.thread = Thread(target=self._run, daemon=True)
        self.thread.start()

    def enqueue(self, records):
        if self.failure is not None:
            raise RuntimeError("TensorBoard writer failed") from self.failure
        self.queue.put(tuple(records))

    def enqueue_all(self, records, *, chunk_size=1024):
        chunk = []
        for record in records:
            chunk.append(record)
            if len(chunk) == chunk_size:
                self.enqueue(chunk)
                chunk = []
        if chunk:
            self.enqueue(chunk)

    def _run(self):
        try:
            while True:
                records = self.queue.get()
                if records is None:
                    return
                for row in records:
                    if row.get("kind", "scalar") == "histogram":
                        self.writer.add_histogram(
                            row["name"], row["values"], row["step"]
                        )
                    elif row.get("kind") == "text":
                        self.writer.add_text(row["name"], row["text"], row["step"])
                    else:
                        self.writer.add_scalar(
                            row["name"],
                            row["value"],
                            row["step"],
                            new_style=True,
                            double_precision=True,
                        )
        except Exception as exc:
            self.failure = exc

    def close(self):
        self.queue.put(None)
        self.thread.join()
        self.writer.flush()
        self.writer.close()
        if self.failure:
            raise RuntimeError("TensorBoard writer failed") from self.failure

    def enqueue_histogram(self, name, values, step, *, max_elements, max_bytes, rng):
        sampled = histogram_sample(
            values,
            max_elements=max_elements,
            max_bytes=max_bytes,
            rng=rng,
        )
        self.enqueue(
            ({"kind": "histogram", "name": name, "values": sampled, "step": int(step)},)
        )
        return {"sampled": int(sampled.size), "total": int(len(values))}

    def enqueue_text(self, name, text, step):
        self.enqueue(
            ({"kind": "text", "name": str(name), "text": str(text), "step": int(step)},)
        )


def histogram_sample(values, *, max_elements, max_bytes, rng):
    import numpy as np

    flat = np.asarray(values).reshape(-1)
    count = min(
        len(flat), int(max_elements), int(max_bytes) // max(1, flat.dtype.itemsize)
    )
    if count == len(flat):
        return flat.copy()
    indices = np.sort(rng.choice(len(flat), size=count, replace=False))
    return flat[indices]
