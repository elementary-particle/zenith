"""Durable multi-update PPO training orchestration."""

from __future__ import annotations

from dataclasses import asdict
from contextlib import contextmanager
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import signal
import shutil
import subprocess
from time import perf_counter

from .capabilities import configure


_INITIAL_POLICY_PHASES = {
    "behavior_cloning_complete": "behavior_cloning",
}


@contextmanager
def _preserve_training_rng_state():
    """Keep synchronous evaluation from changing subsequent training draws."""
    import random
    import numpy as np
    import torch

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    )
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def parameter_digest(model) -> str:
    tensors = list(model.named_parameters()) + list(model.named_buffers())
    return sha256(
        b"".join(
            name.encode() + value.detach().cpu().contiguous().numpy().tobytes()
            for name, value in sorted(tensors)
        )
    ).hexdigest()


def population_digest(models) -> str:
    digest = sha256()
    for policy_id, model in sorted(models.items()):
        digest.update(policy_id.encode())
        digest.update(parameter_digest(model).encode())
    return digest.hexdigest()


def _merge_update_results(results):
    from .ppo.trainer import UpdateResult, _finalize_statistic_metrics

    rows = tuple(results.values())
    if not rows:
        return UpdateResult(False, 0, 0, 0, {}, "no league model received rollout rows")
    names = set().union(*(row.metrics for row in rows))
    metrics = {
        name: sum(float(row.metrics.get(name, 0.0)) for row in rows) / len(rows)
        for name in names
    }
    metrics["entropy_applicable_rows"] = sum(
        float(row.metrics.get("entropy_applicable_rows", 0.0)) for row in rows
    )
    statistic_names = set().union(*(row.metric_statistics for row in rows))
    statistics = {
        name: sum(float(row.metric_statistics.get(name, 0.0)) for row in rows)
        for name in statistic_names
    }
    _finalize_statistic_metrics(metrics, statistics)
    return UpdateResult(
        all(row.committed for row in rows),
        max(row.policy_version for row in rows),
        min(row.epochs for row in rows),
        sum(row.minibatches for row in rows),
        metrics,
        next((row.reason for row in rows if not row.committed), None),
        statistics,
    )


def _due(update: int, cadence: int) -> bool:
    return cadence > 0 and update % cadence == 0


def _learning_rate(base: float, update: int, total: int, warmup_fraction: float) -> float:
    warmup = int(total * warmup_fraction)
    if warmup and update <= warmup:
        return base * update / warmup
    remaining = max(1, total - warmup)
    return base * max(0.0, total - update + 1) / remaining


def _source_state():
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
        ).stdout)
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = "unavailable", True
    return revision, dirty


def _lineups_state(lineups):
    return [asdict(lineup) for _, lineup in sorted(lineups.items())]


def _restore_lineups(values):
    from .types import MatchLineup

    result = {}
    for value in values or ():
        value = dict(value)
        value["match_id"] = tuple(value["match_id"])
        value["seat_policy_ids"] = tuple(value["seat_policy_ids"])
        value["draw_trace"] = tuple(dict(row) for row in value.get("draw_trace", ()))
        lineup = MatchLineup(**value)
        result[lineup.match_id] = lineup
    return result


class TrainingOrchestrator:
    def __init__(
        self, config, output, *, resume=None, weights_only=False,
        initial_checkpoint=None, profile_stages=False,
        skip_periodic_evaluation=False,
    ):
        import torch
        import riichi

        from .checkpoint import reproducibility_metadata, resolve_latest, restore
        from .evaluation.ratings import RatingTable
        from .metrics import CanonicalMetrics, TensorBoardProjector
        from .model.actor_critic import ActorCritic
        from .population.registry import CheckpointPool
        from .population.sampler import SelfPlaySampler
        from .ppo.trainer import PPOTrainer
        from .profiling import StageProfiler
        from .seeds import SeedStreams

        self.torch = torch
        self.riichi = riichi
        self.config = config
        self.values = config.values
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.profile = configure(self.values["run"]["profile"])
        self.device = "cuda" if self.profile.device == "cuda" else "cpu"
        self.total_matches = int(self.values["curriculum"]["total_matches"])
        self.schedule_matches = int(self.values["curriculum"].get(
            "schedule_matches", self.total_matches
        ))
        self.matches_per_update = int(self.values["rollout"]["matches_per_update"])
        self.total_updates = (self.total_matches + self.matches_per_update - 1) // self.matches_per_update
        self.profiler = StageProfiler(enabled=profile_stages, device=self.device)
        self.skip_periodic_evaluation = bool(skip_periodic_evaluation)
        self.stop_requested = False
        self.update = 0
        self.environment_decisions = 0
        self.completed_matches = 0
        self.last_checkpoint_id = None
        self.outcomes = []
        self.rating_transactions = []
        self.rating_snapshot_id = None
        self.lineups = {}

        lock_name = (
            "requirements-cuda.lock" if self.device == "cuda" else "requirements-cpu.lock"
        )
        lock_path = Path("training") / lock_name
        lock_digest = sha256(lock_path.read_bytes()).hexdigest()
        revision, dirty = _source_state()
        self.metadata = reproducibility_metadata(
            input_config=config.source,
            resolved_config=config.redacted(),
            source_revision=revision,
            dirty=dirty,
            dependency_lock_digest=lock_digest,
            capabilities={**self.profile.as_dict(), "host": platform.platform()},
            level="weights_only" if weights_only else "exact_same_target",
        )
        self.manifest = {
            "run_id": config.digest,
            "config_digest": config.digest,
            "status": "created",
            "profile": self.profile.as_dict(),
            "source_revision": revision,
            "source_dirty": dirty,
            "dependency_lock": lock_name,
            "dependency_lock_digest": lock_digest,
            "pid": os.getpid(),
            "update": 0,
            "total_updates": self.total_updates,
        }
        if initial_checkpoint is not None and resume is not None:
            raise ValueError("--initial-checkpoint is mutually exclusive with --resume")
        self._preflight(resume)

        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        self.streams = SeedStreams(int(self.values["run"]["seed"]))
        torch.manual_seed(self.streams.root)
        if self.device == "cuda":
            torch.cuda.manual_seed_all(self.streams.root)
        model_config = dict(self.values["model"])
        model_config["context_tokens"] = self.values["encoding"]["context_tokens"]
        self.model_config = model_config
        self.model = ActorCritic(model_config).to(self.device)
        self.initial_policy = None
        if initial_checkpoint is not None:
            source_path = Path(initial_checkpoint)
            if (source_path / "latest").is_file():
                source_path = resolve_latest(source_path)
            source = restore(source_path)
            phase = source.get("state", {}).get("phase")
            if phase not in _INITIAL_POLICY_PHASES:
                raise RuntimeError(
                    "initial checkpoint must be completed behavior cloning"
                )
            self.model.load_state_dict(source["model"])
            baseline_path = (
                self.output / "baselines" / source["manifest"]["checkpoint_id"]
            )
            if source_path.resolve() != baseline_path.resolve():
                if baseline_path.exists():
                    existing = restore(baseline_path)
                    if existing["manifest"]["checkpoint_id"] != source[
                        "manifest"
                    ]["checkpoint_id"]:
                        raise RuntimeError("run BC baseline identity changed")
                else:
                    baseline_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copytree(source_path, baseline_path)
            else:
                baseline_path = source_path
            self.initial_policy = {
                "checkpoint_id": source["manifest"]["checkpoint_id"],
                "checkpoint_path": str(baseline_path.resolve()),
                "selected_epoch": int(source.get("state", {}).get("selected_epoch", 0)),
                "source_phase": phase,
                "mode": _INITIAL_POLICY_PHASES[phase],
            }
            self.manifest["initial_policy"] = self.initial_policy
            self.metadata["initial_policy"] = self.initial_policy
        from .population.league import (
            AdversarialLeague, CheckpointLeague, EMASelfPlayLeague,
            PureSelfPlayLeague,
        )

        self.training_mode = self.values["rollout"].get(
            "training_mode", "pure_self_play"
        )
        checkpoint_sources = {}
        if self.training_mode == "checkpoint_league":
            for configured_path in self.values["rollout"]["league_checkpoints"]:
                source_path = Path(configured_path)
                if (source_path / "latest").is_file():
                    source_path = resolve_latest(source_path)
                source = restore(source_path)
                checkpoint_id = str(source["manifest"]["checkpoint_id"])
                if checkpoint_id in checkpoint_sources:
                    raise RuntimeError(
                        "checkpoint league contains duplicate checkpoint identities"
                    )
                checkpoint_sources[checkpoint_id] = (source_path, source)
            self.league = CheckpointLeague(
                checkpoint_sources,
                minimum_games=int(self.values["rollout"].get(
                    "league_minimum_games", 8
                )),
                uniform_fraction=float(self.values["rollout"].get(
                    "league_uniform_fraction", 0.25
                )),
                rating_parameters=self.values["rating"],
            )
        elif self.training_mode == "pure_self_play":
            self.league = PureSelfPlayLeague()
        elif self.training_mode == "ema_self_play":
            self.league = EMASelfPlayLeague()
        else:
            self.league = AdversarialLeague()
        self.league_models = {self.league.policy_ids[0]: self.model}
        # Experiment runners may make frozen, non-learner policies greedy while
        # retaining stochastic exploration for every PPO-eligible seat.
        self.deterministic_rollout_policy_ids = frozenset()
        initial_state = self.model.state_dict()
        for policy_id in self.league.policy_ids[1:]:
            replica = ActorCritic(model_config).to(self.device)
            if policy_id in checkpoint_sources:
                replica.load_state_dict(checkpoint_sources[policy_id][1]["model"])
                replica.requires_grad_(False)
                replica.eval()
            else:
                replica.load_state_dict(initial_state)
            if policy_id not in self.league.trainable_policy_ids:
                replica.requires_grad_(False)
                replica.eval()
            self.league_models[policy_id] = replica
        self.league_trainers = {
            policy_id: PPOTrainer(
                model,
                self.values["ppo"],
                device_type=self.device,
                use_bf16=self.profile.precision == "bf16",
                profiler=self.profiler,
            )
            for policy_id, model in self.league_models.items()
            if policy_id in self.league.trainable_policy_ids
        }
        self.trainer = self.league_trainers[self.league.policy_ids[0]]
        self.opponent_ema = None
        if self.training_mode == "ema_self_play":
            from .ppo.ema_magnet import EMAMagnet
            self.opponent_ema = EMAMagnet(
                self.model,
                half_life_matches=float(
                    self.values["rollout"]["ema_opponent_half_life_matches"]
                ),
            )
            self.opponent_ema.copy_to(
                self.league_models[self.league.policy_ids[1]]
            )
        self.pool = CheckpointPool()
        self.ratings = RatingTable(parameters=self.values["rating"])
        restored = None
        if resume is not None:
            resume_path = Path(resume)
            if (resume_path / "latest").is_file():
                resume_path = resolve_latest(resume_path)
            restored = restore(resume_path)
            self._restore_training_state(restored, weights_only=weights_only)
            if weights_only and restored.get("state", {}).get("phase") \
                    == "behavior_cloning_complete":
                baseline_path = (
                    self.output / "baselines"
                    / restored["manifest"]["checkpoint_id"]
                )
                if resume_path.resolve() != baseline_path.resolve():
                    if baseline_path.exists():
                        existing = restore(baseline_path)
                        if existing["manifest"]["checkpoint_id"] != restored[
                            "manifest"
                        ]["checkpoint_id"]:
                            raise RuntimeError(
                                "run BC baseline identity changed"
                            )
                    else:
                        baseline_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copytree(resume_path, baseline_path)
                else:
                    baseline_path = resume_path
                self.initial_policy = {
                    "checkpoint_id": restored["manifest"]["checkpoint_id"],
                    "checkpoint_path": str(baseline_path.resolve()),
                    "selected_epoch": int(
                        restored.get("state", {}).get("selected_epoch", 0)
                    ),
                    "mode": "shared_shape_rank_v",
                }
                self.manifest["initial_policy"] = self.initial_policy
                self.metadata["initial_policy"] = self.initial_policy
        from .rewards.curriculum import Curriculum
        curriculum_state = None if weights_only else (
            (restored or {}).get("trainer", {}).get("curriculum")
        )
        self.curriculum = Curriculum(self.values["curriculum"], curriculum_state)

        env_state = (restored or {}).get("trainer", {}).get("env") or {}
        if weights_only:
            env_state = {}
        if (
            restored is not None
            and not weights_only
            and self.completed_matches > 0
            and int(env_state.get("engine_version", 0)) != 1
        ):
            raise RuntimeError(
                "legacy PPO checkpoints cannot resume the native rollout engine"
            )
        if "master_seed" in env_state:
            self.env_master_seed = int(env_state["master_seed"])
        else:
            self.env_master_seed = self.streams.python_rng("env").getrandbits(64)
        env_config = self.values["env"]
        self.native_engine = riichi.RolloutEngine(
            int(env_config["num_envs"]),
            master_seed=self.env_master_seed,
            num_threads=int(env_config["num_threads"]),
            context_tokens=int(self.values["encoding"]["context_tokens"]),
            token_budget=int(self.values["ppo"]["token_budget"]),
            rules_profile=env_config["rules_profile"],
        )
        if env_state.get("snapshots"):
            self.native_engine.restore({
                int(environment_id): bytes(payload)
                for environment_id, payload
                in env_state["snapshots"].items()
            })
        from .rollout.native import NativeInferenceRunner
        self.policy_slots = {
            policy_id: slot
            for slot, policy_id in enumerate(self.league.policy_ids)
        }
        self.native_inference = NativeInferenceRunner(
            {
                self.policy_slots[policy_id]: model
                for policy_id, model in self.league_models.items()
            },
            device=self.device,
            backend=self.profile.attention,
            use_bf16=self.profile.precision == "bf16",
            generator=self.streams.torch_generator("action", self.device),
            deterministic_policy_slots={
                self.policy_slots[policy_id]
                for policy_id in self.deterministic_rollout_policy_ids
            },
            profiler=self.profiler,
        )
        self.sampler = SelfPlaySampler(self.streams, self.league)
        writer_session = self._next_writer_session()
        self.writer_session = writer_session
        metric_cursor = None
        if restored is not None and not weights_only:
            restored_metrics = (
                restored.get("trainer", {}).get("metrics")
                or restored.get("state", {}).get("metric_cursor")
                or {}
            )
            metric_cursor = restored_metrics.get("canonical_sequence")
        self.canonical = CanonicalMetrics(
            self.output / "metrics" / "canonical.jsonl",
            run_id=config.digest,
            writer_session=writer_session,
            resume_sequence=metric_cursor,
        )
        tensorboard = self.values["metrics"]["tensorboard"]
        self.projector = None
        if tensorboard["enabled"]:
            rebuild_projection = restored is not None and not weights_only
            # TensorBoard restart markers do not purge tensor-style scalars.
            # Rebuild this disposable view from the reconciled canonical log.
            self.projector = TensorBoardProjector(
                self.output / "tensorboard",
                run_id=config.digest,
                writer_session=writer_session,
                flush_seconds=tensorboard["flush_seconds"],
                reset=rebuild_projection,
            )
            if rebuild_projection:
                from .metrics import tensorboard_records
                self.projector.enqueue_all(
                    tensorboard_records(self.canonical.records())
                )
        if restored is None:
            from .checkpoint import publish

            initial_id = publish(
                self.output / "checkpoints",
                {
                    "model": self.model.state_dict(),
                    "state": {"architecture": self.trainer.architecture,
                              "update": 0, "completed_matches": 0,
                              "purpose": (
                                  "behavior-cloned-policy-initialization"
                                  if self.initial_policy is not None
                                  else "frozen-random-initialization"
                              ),
                              "initial_policy": self.initial_policy},
                },
                metadata=self.metadata,
            )
            self.last_checkpoint_id = initial_id
            self._admit(initial_id)
        self.manifest.update(status="running", update=self.update)
        self._write_manifest()

    def _preflight(self, resume):
        manifest_path = self.output / "run.json"
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing.get("config_digest") != self.config.digest:
                raise RuntimeError(
                    f"run directory {self.output} belongs to another configuration"
                )
            if resume is None and existing.get("update", 0):
                raise RuntimeError(
                    f"run directory {self.output} already contains training state; use --resume"
                )
        (self.output / "resolved-config.json").write_text(
            json.dumps(self.config.redacted(), sort_keys=True, indent=2), encoding="utf-8"
        )

    def _restore_training_state(self, restored, *, weights_only):
        from .evaluation.ratings import RatingTable
        from .population.registry import CheckpointPool

        if weights_only:
            self.model.load_state_dict(restored["model"])
            for policy_id, model in self.league_models.items():
                if model is not self.model \
                        and policy_id in self.league.trainable_policy_ids:
                    model.load_state_dict(restored["model"])
            for trainer in self.league_trainers.values():
                trainer.reset_magnet()
            if self.opponent_ema is not None:
                self.opponent_ema.reset(self.model)
                self.opponent_ema.copy_to(
                    self.league_models[self.league.policy_ids[1]]
                )
            return
        self.model.load_state_dict(restored["model"])
        if restored.get("state", {}).get("architecture") != self.trainer.architecture:
            raise RuntimeError("resume checkpoint has a different actor/critic architecture")
        self.trainer.load_optimizer_state_dict(restored["optimizer"])
        trainer_state = restored["trainer"]
        self.initial_policy = (trainer_state.get("provenance") or {}).get(
            "initial_policy"
        )
        if self.initial_policy is not None:
            self.manifest["initial_policy"] = self.initial_policy
            self.metadata["initial_policy"] = self.initial_policy
        self.trainer.policy_version = int(trainer_state.get("policy_version", 0))
        self.update = int(restored["state"].get("update", 0))
        counters = trainer_state.get("counters") or {}
        self.environment_decisions = int(counters.get("environment_decisions", 0))
        self.completed_matches = int(counters.get("completed_matches", 0))
        if trainer_state.get("seeds") is not None:
            self.streams.load_state_dict(trainer_state["seeds"])
        population = trainer_state.get("population") or {}
        league_state = population.get("league")
        league_models = population.get("league_models") or {}
        league_optimizers = population.get("league_optimizers") or {}
        if league_state is not None:
            self.league.load_state_dict(league_state)
        for policy_id, model in self.league_models.items():
            if policy_id == self.league.policy_ids[0]:
                continue
            if policy_id in league_models:
                model.load_state_dict(league_models[policy_id])
            elif policy_id in self.league.trainable_policy_ids:
                model.load_state_dict(restored["model"])
            # Frozen checkpoint opponents were loaded from their immutable,
            # config-bound sources during construction and are deliberately
            # absent from the learner checkpoint.
            trainer = self.league_trainers.get(policy_id)
            if trainer is not None and policy_id in league_optimizers:
                trainer.load_optimizer_state_dict(league_optimizers[policy_id])
                trainer.policy_version = int(
                    (population.get("league_policy_versions") or {}).get(policy_id, 0)
                )
        if self.opponent_ema is not None:
            self.opponent_ema.load_state_dict(
                population.get("opponent_ema"), self.model
            )
            self.opponent_ema.copy_to(
                self.league_models[self.league.policy_ids[1]]
            )
        if population.get("pool"):
            self.pool = CheckpointPool.from_state_dict(population["pool"])
        self.lineups = _restore_lineups(population.get("lineups"))
        rating = trainer_state.get("rating") or {}
        if rating.get("table"):
            self.ratings = RatingTable.from_state_dict(rating["table"])
        self.outcomes = list(rating.get("outcomes", ()))
        self.rating_transactions = list(rating.get("transactions", ()))
        self.rating_snapshot_id = rating.get("snapshot_id")
        self.last_checkpoint_id = restored["manifest"]["checkpoint_id"]

    def _next_writer_session(self):
        root = self.output / "tensorboard"
        existing = [path for path in root.iterdir()] if root.exists() else []
        return f"session-{self.update:08d}-{len(existing):04d}"

    def _write_manifest(self):
        temporary = self.output / ".run.json.tmp"
        temporary.write_text(
            json.dumps(self.manifest, sort_keys=True, indent=2), encoding="utf-8"
        )
        os.replace(temporary, self.output / "run.json")

    def _lineup(self, environment_id, generation):
        return self.sampler.sample(
            environment_id=environment_id,
            generation=generation,
            current_id=self.league.policy_ids[0],
            policy_version=max(
                trainer.policy_version for trainer in self.league_trainers.values()
            ),
        )

    def _advance_ema_opponent(self, matches):
        if self.opponent_ema is None:
            return
        self.opponent_ema.update(self.model, matches=matches)
        self.opponent_ema.copy_to(
            self.league_models[self.league.policy_ids[1]]
        )

    def update_once_native(self):
        """Run one native PPO update with retained or streaming logical batches."""
        import numpy as np

        from .metric_registry import REGISTRY
        from .ppo.current_kyoku import RANK_UTILITIES
        from .rollout.game_metrics import (
            add_counts, empty_counts, metric_values, native_owned_counts,
        )
        from .types import MetricPoint, RolloutMatchOutcome

        started = perf_counter()
        update_number = self.update + 1
        target_matches = min(
            self.matches_per_update, self.total_matches - self.completed_matches
        )
        if target_matches <= 0:
            raise RuntimeError("match budget is already complete")
        chunk_capacity = min(
            int(self.values["env"]["num_envs"]), target_matches
        )
        curriculum = self.curriculum.snapshot(
            self.completed_matches,
            max(trainer.policy_version for trainer in self.league_trainers.values()),
        )
        completed_after = self.completed_matches + target_matches
        actor_learning_rate = _learning_rate(
            float(self.values["ppo"]["actor_learning_rate"]),
            completed_after,
            self.schedule_matches,
            float(self.values["ppo"]["warmup_fraction"]),
        )
        critic_learning_rate = _learning_rate(
            float(self.values["ppo"]["critic_learning_rate"]),
            completed_after,
            self.schedule_matches,
            float(self.values["ppo"]["warmup_fraction"]),
        )
        retain_logical_batch = chunk_capacity == target_matches
        logical_actor_batches = {
            policy_id: [] for policy_id in self.league_trainers
        }
        logical_critic_batches = {
            policy_id: [] for policy_id in self.league_trainers
        }
        for trainer in self.league_trainers.values():
            for group in trainer.actor_optimizer.param_groups:
                group["lr"] = actor_learning_rate
            for group in trainer.critic_optimizer.param_groups:
                group["lr"] = critic_learning_rate
            if not retain_logical_batch:
                trainer.begin_streaming_update(ema_matches=target_matches)

        def moment():
            return [0, 0.0, 0.0]

        def add_moment(target, values):
            rows = np.asarray(values, dtype=np.float64).reshape(-1)
            target[0] += int(rows.size)
            target[1] += float(rows.sum())
            target[2] += float(np.square(rows).sum())

        def mean(target):
            return target[1] / target[0] if target[0] else 0.0

        def variance(target):
            return max(0.0, target[2] / target[0] - mean(target) ** 2) \
                if target[0] else 0.0

        def normalized_moment(groups):
            result = moment()
            for group in groups.values():
                count = int(group[0])
                if not count:
                    continue
                deviation = variance(group) ** 0.5
                result[0] += count
                if deviation >= 1e-8:
                    result[2] += count * (
                        deviation / (deviation + 1e-8)
                    ) ** 2
            return result

        moments = {
            "advantage": moment(),
            "policy_advantages": {
                slot: moment() for slot in self.policy_slots.values()
            },
            "rank_target": moment(),
            "rank_residual": moment(),
        }
        totals = {
            "matches": 0,
            "kyoku": 0,
            "decisions": 0,
            "eligible": 0,
            "env_calls": 0,
            "model_queries": 0,
            "minibatches": 0,
            "total_tokens": 0,
            "padded_tokens": 0,
            "native_bot_rows": 0,
            "automatic_rows": 0,
        }
        action_statistics = {
            "call_count": 0.0,
            "pass_count": 0.0,
            "riichi_count": 0.0,
            "dama_count": 0.0,
        }
        game_counts = empty_counts()
        trajectory_digest = sha256()
        rollout_policy_digest = population_digest(self.league_models)
        try:
            while totals["matches"] < target_matches:
                chunk_matches = min(
                    chunk_capacity, target_matches - totals["matches"]
                )
                with self.profiler.measure("rollout.encoding"):
                    match_ids = tuple(
                        tuple(map(int, match_id))
                        for match_id in self.native_engine.reset_chunk(chunk_matches)
                    )
                    lineups = tuple(
                        self._lineup(environment_id, generation)
                        for environment_id, generation in match_ids
                    )
                    self.lineups.update({
                        lineup.match_id: lineup for lineup in lineups
                    })
                    self.native_engine.register_lineups(
                        match_ids,
                        [
                            tuple(
                                self.policy_slots[policy_id]
                                for policy_id in lineup.seat_policy_ids
                            )
                            for lineup in lineups
                        ],
                        [lineup.learner_mask for lineup in lineups],
                    )
                with self.profiler.measure(
                    "rollout.actor_candidate_processing"
                ):
                    chunk = self.native_inference.run_chunk(self.native_engine)
                if population_digest(self.league_models) != rollout_policy_digest:
                    raise RuntimeError(
                        "policy parameters changed during native rollout collection"
                    )
                with self.profiler.measure("rollout.frame_critic_forward"):
                    self.native_inference.prepare_training_chunk(chunk)
                columns = chunk.columns()
                eligibility = np.asarray(columns["eligibility"], dtype=bool)
                advantages = np.asarray(columns["advantages"], dtype=np.float32)
                row_policy_slots = np.asarray(columns["policy_slots"])
                predictions = np.asarray(
                    columns["old_boundary_values"], dtype=np.float32
                )
                placements = np.asarray(
                    columns["terminal_placements"], dtype=np.int64
                )
                rank_targets = RANK_UTILITIES[placements[eligibility]]
                add_moment(moments["advantage"], advantages[eligibility])
                for slot, target in moments["policy_advantages"].items():
                    add_moment(
                        target,
                        advantages[eligibility & (row_policy_slots == slot)],
                    )
                add_moment(moments["rank_target"], rank_targets)
                add_moment(
                    moments["rank_residual"],
                    rank_targets - predictions[eligibility],
                )

                actor_batches_by_policy = {}
                critic_batches_by_policy = {}
                with self.profiler.measure("ppo.packing"):
                    for policy_id, trainer in self.league_trainers.items():
                        slot = self.policy_slots[policy_id]
                        batches = list(chunk.actor_minibatches(
                            slot,
                            int(self.values["ppo"]["token_budget"]),
                            max_padding_fraction=float(
                                self.values["encoding"]["packing_max_waste"]
                            ),
                            backend=self.profile.attention,
                        ))
                        if not batches:
                            continue
                        actor_batches_by_policy[policy_id] = batches
                        critic_batches_by_policy[policy_id] = [
                            chunk.critic_batch(policy_slot=slot)
                        ]
                        if retain_logical_batch:
                            logical_actor_batches[policy_id].extend(batches)
                            logical_critic_batches[policy_id].extend(
                                critic_batches_by_policy[policy_id]
                            )
                        else:
                            trainer.accumulate_streaming_chunk(
                                batches,
                                critic_minibatches=(
                                    critic_batches_by_policy[policy_id]
                                ),
                            )
                        for batch in batches:
                            model_inputs = batch["model_inputs"]
                            lengths = np.asarray(model_inputs["lengths"])
                            totals["total_tokens"] += int(lengths.sum())
                            totals["padded_tokens"] += int(
                                model_inputs["token_factors"].shape[0]
                                * model_inputs["token_factors"].shape[1]
                            )
                            totals["minibatches"] += 1

                terminal_environment_ids = np.asarray(
                    columns["terminal_environment_ids"]
                )
                terminal_generations = np.asarray(
                    columns["terminal_episode_generations"]
                )
                terminal_scores = np.asarray(columns["terminal_scores"])
                terminal_ranks = np.asarray(columns["terminal_ranks"])
                terminal_kyoku = np.asarray(
                    columns["terminal_completed_kyoku"]
                )
                lineup_by_match = {
                    lineup.match_id: lineup for lineup in lineups
                }
                outcomes = tuple(
                    RolloutMatchOutcome(
                        (
                            int(terminal_environment_ids[index]),
                            int(terminal_generations[index]),
                        ),
                        lineup_by_match[(
                            int(terminal_environment_ids[index]),
                            int(terminal_generations[index]),
                        )].seat_policy_ids,
                        tuple(map(int, terminal_ranks[index])),
                        tuple(map(int, terminal_scores[index])),
                        int(terminal_kyoku[index]),
                    )
                    for index in range(len(terminal_environment_ids))
                )
                self.league.record(outcomes)
                for index, outcome in enumerate(outcomes):
                    lineup = lineup_by_match[outcome.match_id]
                    owned = [
                        seat
                        for seat, policy_id in enumerate(lineup.seat_policy_ids)
                        if policy_id in self.league_trainers
                    ]
                    if owned:
                        add_counts(
                            game_counts,
                            native_owned_counts(columns, index, owned),
                        )

                native_metrics = self.native_engine.metrics()
                totals["matches"] += int(chunk.match_completions)
                totals["kyoku"] += int(chunk.kyoku_completions)
                totals["decisions"] += int(chunk.row_count)
                totals["eligible"] += int(eligibility.sum())
                totals["env_calls"] += int(native_metrics["env_calls"])
                totals["model_queries"] += int(native_metrics["inference_rows"])
                totals["native_bot_rows"] += int(
                    native_metrics["native_bot_rows"]
                )
                totals["automatic_rows"] += int(
                    native_metrics["automatic_rows"]
                )
                for name, value in chunk.action_statistics().items():
                    action_statistics[name] += float(value)
                trajectory_digest.update(chunk.trajectory_digest.encode("ascii"))
                for match_id in match_ids:
                    self.lineups.pop(match_id, None)
                del columns, chunk, actor_batches_by_policy
                del critic_batches_by_policy
        except Exception:
            for trainer in self.league_trainers.values():
                trainer.abort_streaming_update()
            raise

        target_variance = variance(moments["rank_target"])
        rank_ev = 0.0 if target_variance < 1e-12 else 1.0 - (
            variance(moments["rank_residual"]) / target_variance
        )
        for trainer in self.league_trainers.values():
            trainer.set_rollout_critic_evidence(rank_explained_variance=rank_ev)
        with self.profiler.measure("system.parameter_digest"):
            before = population_digest(self.league_models)
        with self.profiler.measure("ppo.optimization"):
            if retain_logical_batch:
                update_results = {
                    policy_id: trainer.update_logical_batch(
                        logical_actor_batches[policy_id],
                        critic_minibatches=logical_critic_batches[policy_id],
                        ema_matches=target_matches,
                    )
                    for policy_id, trainer in self.league_trainers.items()
                }
            else:
                update_results = {
                    policy_id: trainer.finish_streaming_update()
                    for policy_id, trainer in self.league_trainers.items()
                }
            update_result = _merge_update_results(update_results)
        with self.profiler.measure("system.parameter_digest"):
            after = population_digest(self.league_models)
        if not update_result.committed:
            raise RuntimeError(f"PPO update rejected: {update_result.reason}")
        if before == after:
            raise RuntimeError("committed PPO update did not change parameters")

        self._advance_ema_opponent(target_matches)
        self.update = update_number
        self.completed_matches = completed_after
        self.environment_decisions += totals["decisions"]
        call_total = action_statistics["call_count"] + action_statistics["pass_count"]
        riichi_total = action_statistics["riichi_count"] + action_statistics["dama_count"]
        elapsed = perf_counter() - started
        normalized_advantages = normalized_moment(
            moments["policy_advantages"]
        )
        values = {
            "critic/match_rank_explained_variance": rank_ev,
            "rollout/current_kyoku_advantage_mean": mean(moments["advantage"]),
            "rollout/current_kyoku_advantage_std": variance(
                moments["advantage"]
            ) ** 0.5,
            "rollout/policy_advantage_mean": mean(normalized_advantages),
            "rollout/policy_advantage_std": variance(
                normalized_advantages
            ) ** 0.5,
            "rollout/kyoku_completions": float(totals["kyoku"]),
            "rollout/match_completions": float(totals["matches"]),
            **metric_values(game_counts),
            "curriculum/progress": curriculum.progress,
            "population/pool_size": float(len(self.pool.snapshot().eligible)),
            **{
                f"league/{key}": value
                for key, value in self.league.metrics().items()
            },
            "performance/model_queries_per_second": totals["model_queries"]
            / max(elapsed, 1e-9),
            "performance/rust_resolved_decisions": float(
                totals["native_bot_rows"] + totals["automatic_rows"]
            ),
            "performance/automatic_resolution_fraction": (
                totals["native_bot_rows"] + totals["automatic_rows"]
            ) / max(
                1,
                totals["native_bot_rows"]
                + totals["automatic_rows"]
                + totals["model_queries"],
            ),
            "curriculum/actor_learning_rate": actor_learning_rate,
            "curriculum/critic_learning_rate": critic_learning_rate,
            "rollout/call_opportunity_selected_call_rate": (
                action_statistics["call_count"] / call_total
                if call_total else 0.0
            ),
            "rollout/riichi_opportunity_selected_riichi_rate": (
                action_statistics["riichi_count"] / riichi_total
                if riichi_total else 0.0
            ),
        }
        with self.profiler.measure("metrics.commit"):
            points = self.trainer.metric_points(
                self.completed_matches,
                update_result,
                source="native-columnar-chunk",
            )
            for name, value in values.items():
                definition = REGISTRY[name]
                points.append(MetricPoint(
                    name,
                    definition.axis,
                    self.completed_matches,
                    value,
                    definition.unit,
                    definition.window,
                    definition.reduction,
                    "native-update",
                ))
            records = self.canonical.commit(points)
            if self.projector is not None:
                from .metrics import tensorboard_records
                self.projector.enqueue(tensorboard_records(records))
        evidence = {
            "status": "passed",
            "profile": self.values["run"]["profile"],
            "device": self.device,
            "environment_count": int(self.values["env"]["num_envs"]),
            "rollout_chunk_matches": chunk_capacity,
            "rollout_chunks": (
                target_matches + chunk_capacity - 1
            ) // chunk_capacity,
            "learner_decisions": totals["decisions"],
            "eligible_decisions": totals["eligible"],
            "env_calls": totals["env_calls"],
            "trajectory_digest": trajectory_digest.hexdigest(),
            "rollout_opponents": self.training_mode.replace("_", "-"),
            "league_updates": {
                policy_id: {
                    key: value for key, value in asdict(result).items()
                    if key != "metric_statistics"
                }
                for policy_id, result in update_results.items()
            },
            "policy_version": update_result.policy_version,
            "update": self.update,
            "parameter_digest_before": before,
            "parameter_digest_after": after,
            "ppo": update_result.metrics,
            "ema_magnet": {
                key: value for key, value in update_result.metrics.items()
                if key.startswith("magnet_")
            },
            "packing": {
                "minibatches": totals["minibatches"],
                "total_tokens": totals["total_tokens"],
                "padded_tokens": totals["padded_tokens"],
            },
            "elapsed_seconds": elapsed,
            "model_queries_per_second": totals["model_queries"]
            / max(elapsed, 1e-9),
        }
        if self.profiler.enabled:
            evidence["profile"] = self.profiler.snapshot()
        return evidence, curriculum

    def _checkpoint(self, curriculum):
        initialized = min(
            int(self.values["env"]["num_envs"]),
            int(self.matches_per_update),
            int(self.total_matches),
        )
        native_snapshots = self.native_engine.snapshot()
        checkpoint_id = self.trainer.publish_checkpoint(
            self.output / "checkpoints",
            metadata=self.metadata,
            counters={
                "update": self.update,
                "environment_decisions": self.environment_decisions,
                "completed_matches": self.completed_matches,
            },
            seeds=self.streams,
            curriculum=self.curriculum.state_dict(),
            population={
                "pool": self.pool.state_dict(),
                "lineups": _lineups_state(self.lineups),
                "league": self.league.state_dict(),
                "opponent_ema": (
                    None if self.opponent_ema is None
                    else self.opponent_ema.state_dict()
                ),
                "league_models": {
                    policy_id: model.state_dict()
                    for policy_id, model in self.league_models.items()
                    if policy_id != self.league.policy_ids[0]
                    and policy_id in self.league.trainable_policy_ids
                },
                "league_optimizers": {
                    policy_id: trainer.optimizer_state_dict()
                    for policy_id, trainer in self.league_trainers.items()
                    if policy_id != self.league.policy_ids[0]
                },
                "league_policy_versions": {
                    policy_id: trainer.policy_version
                    for policy_id, trainer in self.league_trainers.items()
                },
            },
            rating={
                "table": self.ratings.state_dict(),
                "outcomes": self.outcomes,
                "transactions": self.rating_transactions,
                "snapshot_id": self.rating_snapshot_id,
            },
            metrics={
                "canonical_sequence": self.canonical.sequence,
                "writer_session": self.writer_session,
            },
            provenance={"initial_policy": self.initial_policy},
            env={
                "master_seed": self.env_master_seed,
                "engine_version": 1,
                "snapshots": {
                    environment_id: native_snapshots[environment_id]
                    for environment_id in range(initialized)
                },
                "safe_boundary": True,
            },
        )
        self.last_checkpoint_id = checkpoint_id
        return checkpoint_id

    def _admit(self, checkpoint_id):
        from .population.registry import PoolEntry

        path = self.output / "checkpoints" / checkpoint_id
        size = sum(file.stat().st_size for file in path.iterdir() if file.is_file())
        self.pool.admit(PoolEntry(
            checkpoint_id,
            str(path.resolve()),
            self.config.digest,
            self.update,
            bytes=size,
        ))
        self.pool.enforce_retention(
            int(self.values["population"]["retained_checkpoints_max"])
        )

    def _prune_checkpoints(self):
        root = self.output / "checkpoints"
        if not root.exists():
            return
        protected = {self.last_checkpoint_id}
        protected.update(
            entry.checkpoint_id
            for entry in self.pool.snapshot().entries
            if not entry.retired
        )
        protected.update(
            checkpoint_id
            for lineup in self.lineups.values()
            for checkpoint_id in lineup.seat_policy_ids
            if checkpoint_id != "current"
        )
        directories = sorted(
            (path for path in root.iterdir() if path.is_dir() and not path.name.startswith(".")),
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )
        protected.update(
            path.name for path in directories[: int(self.values["checkpoint"]["keep"])]
        )
        for path in directories:
            if path.name not in protected:
                shutil.rmtree(path)

    def _evaluate(self):
        from .checkpoint import publish_evaluation_records, restore
        from .cli.evaluate import (
            _evaluation_device,
            _load_models,
            _play_games,
        )
        from .config import evaluation_seeds
        from .evaluation.runner import paired_bootstrap, run_series_batched
        from .model.actor_critic import ActorCritic

        evaluation_device = _evaluation_device(self.config, self.device)
        evaluation_batch_size = int(
            self.values["evaluation"].get("batch_size", 32)
        )

        held_out_seeds = evaluation_seeds(self.values["evaluation"])

        def evaluate_series(
            checkpoint_ids, models, *, series_id, greedy_checkpoint_ids=(),
        ):
            return run_series_batched(
                checkpoint_ids,
                held_out_seeds,
                lambda requests: _play_games(
                    models,
                    self.values["model"],
                    requests,
                    device=evaluation_device,
                    token_budget=int(
                        self.values["evaluation"].get(
                            "token_budget", 65_536
                        )
                    ),
                    max_padding_fraction=float(
                        self.values["encoding"]["packing_max_waste"]
                    ),
                    max_frames=int(
                        self.values["rollout"]["max_frames_per_match"]
                    ),
                    greedy_checkpoint_ids=greedy_checkpoint_ids,
                ),
                batch_size=evaluation_batch_size,
                series_id=series_id,
            )

        # The immutable BC initialization is the only fixed baseline. It is
        # never sampled into PPO trajectories.
        probe_id = self.league.policy_ids[0]
        probe_model = ActorCritic(self.model_config)
        probe_model.load_state_dict(self.league_models[probe_id].state_dict())
        probe_model.to(evaluation_device).eval()
        evaluation_root = self.output / "evaluations"
        evaluation_root.mkdir(parents=True, exist_ok=True)
        baseline_model = baseline_id = None
        if self.initial_policy is None:
            baseline_report = {
                "status": "skipped",
                "reason": "PPO was not initialized from a BC checkpoint",
            }
        else:
            baseline_path = Path(self.initial_policy["checkpoint_path"])
            baseline_state = restore(baseline_path)
            baseline_id = baseline_state["manifest"]["checkpoint_id"]
            baseline_model = ActorCritic(self.model_config)
            baseline_model.load_state_dict(baseline_state["model"])
            baseline_model.to(evaluation_device).eval()
            probe_models = {
                probe_id: probe_model,
                baseline_id: baseline_model,
            }
            series_id = f"bc-baseline-{self.update:08d}-{probe_id}"
            baseline_outcomes = evaluate_series(
                (probe_id, baseline_id, baseline_id, baseline_id),
                probe_models,
                series_id=series_id,
            )
            valid_baseline = [
                outcome for outcome in baseline_outcomes if outcome["valid"]
            ]
            wins = comparisons = 0.0
            for outcome in valid_baseline:
                learner_ranks = [
                    outcome["ranks"][seat]
                    for seat, value in enumerate(outcome["checkpoint_ids"])
                    if value == probe_id
                ]
                baseline_ranks = [
                    outcome["ranks"][seat]
                    for seat, value in enumerate(outcome["checkpoint_ids"])
                    if value == baseline_id
                ]
                for learner_rank in learner_ranks:
                    for baseline_rank in baseline_ranks:
                        wins += float(learner_rank < baseline_rank)
                        wins += 0.5 * float(learner_rank == baseline_rank)
                        comparisons += 1
            baseline_report = {
                "status": "completed",
                "series_id": series_id,
                "policy_id": probe_id,
                "baseline_checkpoint_id": baseline_id,
                "protocol": "one learner versus three frozen BC seats",
                "learner_seats_per_game": 1,
                "baseline_seats_per_game": 3,
                "valid_games": len(valid_baseline),
                "pairwise_win_rate": wins / max(1.0, comparisons),
                "comparisons": int(comparisons),
                "paired_bootstrap": paired_bootstrap(
                    valid_baseline, probe_id, baseline_id,
                ),
            }
        (evaluation_root / "bc-baseline.json").write_text(
            json.dumps(baseline_report, sort_keys=True, indent=2),
            encoding="utf-8",
        )

        eligible = sorted(
            self.pool.snapshot().eligible,
            key=lambda entry: (entry.source_update, entry.checkpoint_id),
        )

        sampled_id = f"{probe_id}@sampled-u{self.update:08d}"
        greedy_id = f"{probe_id}@greedy-u{self.update:08d}"
        witness_rows = []
        witness_outcomes = []

        def evaluate_witness(
            candidate_id, candidate_model, *, kind, source_update=None,
            greedy=False,
        ):
            series_id = (
                f"exploitability-{self.update:08d}-{kind}-{candidate_id}"
            )
            outcomes = evaluate_series(
                (candidate_id, sampled_id, sampled_id, sampled_id),
                {
                    candidate_id: candidate_model,
                    sampled_id: probe_model,
                },
                series_id=series_id,
                greedy_checkpoint_ids={candidate_id} if greedy else (),
            )
            valid = [outcome for outcome in outcomes if outcome["valid"]]
            comparison = paired_bootstrap(
                valid, candidate_id, sampled_id,
            )
            witness_outcomes.extend(outcomes)
            witness_rows.append({
                "kind": kind,
                "candidate_id": candidate_id,
                "target_id": sampled_id,
                "source_update": source_update,
                "valid_games": len(valid),
                "paired_bootstrap": comparison,
            })

        evaluate_witness(
            greedy_id,
            probe_model,
            kind="current-greedy",
            source_update=self.update,
            greedy=True,
        )
        if baseline_model is not None:
            evaluate_witness(
                baseline_id,
                baseline_model,
                kind="bc",
                source_update=0,
            )

        previous = [
            entry for entry in eligible
            if 0 < int(entry.source_update) < self.update
            and entry.checkpoint_id != baseline_id
        ]
        selected_previous = []
        for lag in (1, 8, 32):
            candidates = [
                entry for entry in previous
                if int(entry.source_update) <= self.update - lag
            ]
            if candidates:
                selected = candidates[-1]
                if selected.checkpoint_id not in {
                    entry.checkpoint_id for entry in selected_previous
                }:
                    selected_previous.append(selected)
        for entry in selected_previous:
            models, records = _load_models(
                self.config, (entry.artifact,), device=evaluation_device,
            )
            candidate_id = records[0]["checkpoint_id"]
            evaluate_witness(
                candidate_id,
                models[candidate_id],
                kind="earlier-checkpoint",
                source_update=int(entry.source_update),
            )

        maximum = max(
            witness_rows,
            key=lambda row: row["paired_bootstrap"][
                "pairwise_win_rate"
            ]["mean"],
        )
        witness_report = {
            "protocol": "restricted unilateral deviation",
            "claim": "lower-bound exploitability witnesses, not best responses",
            "target_update": self.update,
            "target_completed_matches": self.completed_matches,
            "seed_count": len(held_out_seeds),
            "matches_per_witness": 4 * len(held_out_seeds),
            "batch_size": evaluation_batch_size,
            "witnesses": witness_rows,
            "maximum_pairwise_witness": {
                "candidate_id": maximum["candidate_id"],
                "kind": maximum["kind"],
                "pairwise_win_rate": maximum["paired_bootstrap"][
                    "pairwise_win_rate"
                ],
            },
        }
        witness_stem = (
            f"exploitability-{self.completed_matches:09d}"
        )
        (evaluation_root / f"{witness_stem}.json").write_text(
            json.dumps(witness_report, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        with (evaluation_root / f"{witness_stem}.outcomes.jsonl").open(
            "w", encoding="utf-8",
        ) as output:
            for outcome in witness_outcomes:
                output.write(
                    json.dumps(
                        outcome, sort_keys=True, separators=(",", ":"),
                    ) + "\n"
                )

        if len(eligible) < 4:
            return {
                "status": "completed",
                "bc_baseline": baseline_report,
                "exploitability_witnesses": witness_report,
                "checkpoint_series": {
                    "status": "skipped", "reason": "fewer than four checkpoints"
                },
            }
        participants = eligible[-4:]
        paths = [entry.artifact for entry in participants]
        models, records = _load_models(
            self.config, paths, device=evaluation_device,
        )
        checkpoint_ids = tuple(record["checkpoint_id"] for record in records)
        series_id = f"held-out-{self.update:08d}"
        outcomes = evaluate_series(
            checkpoint_ids,
            models,
            series_id=series_id,
        )
        valid = [outcome for outcome in outcomes if outcome["valid"]]
        self.ratings.update(valid)
        self.outcomes.extend(outcomes)
        self.rating_transactions.extend(valid)
        self.rating_snapshot_id = self.pool.publish_rating_snapshot(self.ratings)
        publish_evaluation_records(
            evaluation_root, self.outcomes, self.rating_transactions
        )
        leaderboard = [
            {
                "checkpoint_id": key,
                "mu": rating.mu,
                "sigma": rating.sigma,
                "ordinal": rating.mu
                - float(self.values["rating"]["ordinal_sigma"]) * rating.sigma,
                "games": rating.games,
                "placements": rating.placements,
                "last_series": rating.last_series,
            }
            for key, rating in self.ratings.leaderboard()
        ]
        (evaluation_root / "leaderboard.json").write_text(
            json.dumps(leaderboard, sort_keys=True, indent=2), encoding="utf-8"
        )
        return {
            "status": "completed",
            "series_id": series_id,
            "valid_games": len(valid),
            "participants": checkpoint_ids,
            "bc_baseline": baseline_report,
            "exploitability_witnesses": witness_report,
        }

    def run(self, *, max_updates=None):
        limit = self.total_updates
        if max_updates is not None:
            limit = min(limit, self.update + int(max_updates))
        last_evidence = None
        last_curriculum = None
        while (self.completed_matches < self.total_matches
               and self.update < limit and not self.stop_requested):
            last_evidence, last_curriculum = self.update_once_native()
            checkpoint_due = _due(
                self.completed_matches,
                int(self.values["checkpoint"]["cadence_matches"]),
            )
            scheduled = self.values["evaluation"]["checkpoint_matches"]
            evaluation_due = not self.skip_periodic_evaluation and (
                self.completed_matches in scheduled if scheduled else _due(
                    self.completed_matches,
                    int(self.values["evaluation"]["cadence_matches"]),
                )
            )
            admission_due = checkpoint_due or evaluation_due
            final = self.update == limit or self.stop_requested
            if checkpoint_due or admission_due or evaluation_due or final:
                with self.profiler.measure("checkpoint.publish"):
                    checkpoint_id = self._checkpoint(last_curriculum)
                last_evidence["checkpoint_id"] = checkpoint_id
                last_evidence["checkpoint_path"] = str(
                    self.output / "checkpoints" / checkpoint_id
                )
                if admission_due:
                    self._admit(checkpoint_id)
                if evaluation_due:
                    with _preserve_training_rng_state():
                        last_evidence["evaluation"] = self._evaluate()
                if admission_due or evaluation_due:
                    with self.profiler.measure("checkpoint.publish"):
                        checkpoint_id = self._checkpoint(last_curriculum)
                    last_evidence["checkpoint_id"] = checkpoint_id
                    last_evidence["checkpoint_path"] = str(
                        self.output / "checkpoints" / checkpoint_id
                    )
                self._prune_checkpoints()
            (self.output / "one-update.json").write_text(
                json.dumps(last_evidence, sort_keys=True, indent=2), encoding="utf-8"
            )
            self.manifest.update(
                update=self.update,
                policy_version=max(
                    trainer.policy_version for trainer in self.league_trainers.values()
                ),
                environment_decisions=self.environment_decisions,
                last_checkpoint_id=self.last_checkpoint_id,
            )
            self._write_manifest()
            if _due(self.completed_matches, int(self.values["metrics"]["progress_every_matches"])):
                print(json.dumps({
                    "update": self.update,
                    "total_updates": self.total_updates,
                    "model_queries_per_second": last_evidence["model_queries_per_second"],
                    "pool_size": len(self.pool.snapshot().eligible),
                }, sort_keys=True), flush=True)
        completed = self.completed_matches >= self.total_matches
        status = "completed" if completed else (
            "interrupted" if self.stop_requested else "stopped"
        )
        if last_curriculum is not None and self.last_checkpoint_id is None:
            self._checkpoint(last_curriculum)
        self.manifest.update(status=status, update=self.update)
        self._write_manifest()
        return {
            "status": status,
            "update": self.update,
            "total_updates": self.total_updates,
            "completed_matches": self.completed_matches,
            "total_matches": self.total_matches,
            "policy_version": max(
                trainer.policy_version for trainer in self.league_trainers.values()
            ),
            "environment_decisions": self.environment_decisions,
            "last_checkpoint_id": self.last_checkpoint_id,
            "last_update": last_evidence,
        }

    def close(self):
        if self.projector is not None:
            self.projector.close()
            self.projector = None


def run_training(
    config, output, *, resume=None, weights_only=False, max_updates=None,
    initial_checkpoint=None, profile_stages=False,
    skip_periodic_evaluation=False,
):
    orchestrator = TrainingOrchestrator(
        config,
        output,
        resume=resume,
        weights_only=weights_only,
        initial_checkpoint=initial_checkpoint,
        profile_stages=profile_stages,
        skip_periodic_evaluation=skip_periodic_evaluation,
    )
    old_handlers = {}

    def request_stop(signum, frame):
        del signum, frame
        orchestrator.stop_requested = True

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            try:
                old_handlers[signum] = signal.signal(signum, request_stop)
            except ValueError:
                pass
        result = orchestrator.run(max_updates=max_updates)
        if profile_stages:
            result["profile"] = orchestrator.profiler.snapshot()
            (Path(output) / "profile.json").write_text(
                json.dumps(result["profile"], sort_keys=True, indent=2), encoding="utf-8"
            )
        return result
    except Exception as exc:
        orchestrator.manifest.update(
            status="failed", update=orchestrator.update,
            failure={"type": type(exc).__name__, "message": str(exc)},
        )
        orchestrator._write_manifest()
        raise
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
        orchestrator.close()
