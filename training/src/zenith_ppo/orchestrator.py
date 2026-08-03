"""Durable multi-update PPO training orchestration."""

from __future__ import annotations

from dataclasses import asdict, replace
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
        from .env.adapter import EnvAdapter
        from .env.history import HistoryRegistry
        from .encoding.event_cache import EventPrefixCache
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
        if "master_seed" in env_state:
            self.env_master_seed = int(env_state["master_seed"])
        else:
            self.env_master_seed = self.streams.python_rng("env").getrandbits(64)
        env_config = self.values["env"]
        self.env = riichi.Env(
            int(env_config["num_envs"]),
            master_seed=self.env_master_seed,
            num_threads=int(env_config["num_threads"]),
            rules_profile=env_config["rules_profile"],
            privileged=bool(env_config["privileged"]),
        )
        self.adapter = EnvAdapter(self.env)
        self.event_cache = EventPrefixCache()
        if env_state.get("histories"):
            self.adapter.histories = HistoryRegistry.from_state_dict(env_state["histories"])
        if env_state.get("snapshots"):
            # Historical checkpoints captured every allocated native slot,
            # including slots beyond matches_per_update that had never been
            # initialized.  Privileged projection cannot materialize hidden
            # state for such slots.  Rollouts always launch the dense prefix,
            # so filtering that prefix retains every episode generation that
            # can affect exact continuation.
            initialized_count = min(
                int(env_config["num_envs"]),
                int(self.matches_per_update),
                int(self.total_matches),
            )
            snapshots = {
                int(environment_id): payload
                for environment_id, payload in env_state["snapshots"].items()
                if int(environment_id) < initialized_count
            }
            if snapshots:
                self.adapter.restore(snapshots)
        launch_count = min(
            int(env_config["num_envs"]),
            self.matches_per_update,
            self.total_matches - self.completed_matches,
        )
        self.batch = self.adapter.reset(range(launch_count))
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

    def _streaming_training_batches(self, collection, advantages):
        """Materialize only one bounded rollout chunk for gradient accumulation."""
        import torch

        from .encoding.packing import PackedBatch, pack

        eligible = [
            index for index, sample in enumerate(collection.samples)
            if sample.ppo_eligible
        ]
        advantage_by_sample = {
            int(index): float(value)
            for index, value in zip(
                advantages.indices, advantages.normalized, strict=True
            )
        }
        eligible_by_policy = {
            policy_id: [
                index for index in eligible
                if collection.samples[index].checkpoint_id == policy_id
            ]
            for policy_id in self.league.policy_ids
        }
        eligible_by_policy = {
            policy_id: indices
            for policy_id, indices in eligible_by_policy.items() if indices
        }
        actor_packed_by_policy = {
            policy_id: pack(
                [
                    len(collection.samples[index].encoded.token_factors)
                    for index in indices
                ],
                int(self.values["ppo"]["token_budget"]),
                max_padding_fraction=float(
                    self.values["encoding"]["packing_max_waste"]
                ),
            )
            for policy_id, indices in eligible_by_policy.items()
        }
        packed = PackedBatch(
            tuple(range(len(eligible))),
            tuple(
                batch
                for policy_packed in actor_packed_by_policy.values()
                for batch in policy_packed.batches
            ),
            sum(value.total_tokens for value in actor_packed_by_policy.values()),
            sum(value.padded_tokens for value in actor_packed_by_policy.values()),
        )
        actor_batches = {}
        for policy_id, policy_packed in actor_packed_by_policy.items():
            policy_indices = eligible_by_policy[policy_id]
            batches = []
            for packed_rows in policy_packed.batches:
                sample_indices = [policy_indices[row] for row in packed_rows]
                samples = [collection.samples[index] for index in sample_indices]
                action_counts = torch.tensor(
                    [len(sample.encoded.action_factors) for sample in samples],
                    dtype=torch.long,
                )
                action_starts = torch.cat((
                    torch.zeros(1, dtype=torch.long),
                    action_counts.cumsum(0)[:-1],
                ))
                batches.append({
                    "encoded_rows": tuple(sample.encoded for sample in samples),
                    "backend": self.profile.attention,
                    "action_counts": action_counts,
                    "selected": action_starts + torch.tensor(
                        [sample.selected_group for sample in samples],
                        dtype=torch.long,
                    ),
                    "old_logp": torch.tensor(
                        [sample.old_log_probability for sample in samples],
                        dtype=torch.float32,
                    ),
                    "advantages": torch.tensor(
                        [advantage_by_sample[index] for index in sample_indices],
                        dtype=torch.float32,
                    ),
                })
            actor_batches[policy_id] = batches

        critic_batches = {}

        boundary_by_policy = {
            policy_id: [
                frame for frame in collection.frames
                if frame.ppo_eligible
                and frame.checkpoint_id == policy_id
                and frame.rank_boundary_supervision
                and 0 <= frame.rank_order_target < 24
            ]
            for policy_id in self.league.policy_ids
        }
        for policy_id, rows in boundary_by_policy.items():
            if not rows:
                continue
            boundary_packed = pack(
                [len(row.encoded.token_factors) for row in rows],
                int(self.values["ppo"]["token_budget"]),
                max_padding_fraction=float(
                    self.values["encoding"]["packing_max_waste"]
                ),
            )
            batches = critic_batches.setdefault(policy_id, [])
            for packed_rows in boundary_packed.batches:
                selected_rows = [rows[index] for index in packed_rows]
                batches.append({
                    "encoded_rows": tuple(row.encoded for row in selected_rows),
                    "backend": self.profile.attention,
                    "include_actions": False,
                    "boundary_only": True,
                    "rank_boundary_supervision": torch.ones(
                        len(selected_rows), dtype=torch.bool
                    ),
                    "rank_order_targets": torch.tensor(
                        [row.rank_order_target for row in selected_rows],
                        dtype=torch.long,
                    ),
                })
        return eligible, packed, actor_batches, critic_batches

    def update_once_streaming(self):
        """Accumulate one low-noise update while releasing rollout chunks."""
        from hashlib import sha256

        import numpy as np

        from .metric_registry import REGISTRY
        from .ppo.current_kyoku import (
            RANK_UTILITIES,
            action_family_statistics,
            compute,
        )
        from .rollout.collector import Collector, _game_metric_values
        from .types import MetricPoint

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
            self.total_matches,
            float(self.values["ppo"]["warmup_fraction"]),
        )
        critic_learning_rate = _learning_rate(
            float(self.values["ppo"]["critic_learning_rate"]),
            completed_after,
            self.total_matches,
            float(self.values["ppo"]["warmup_fraction"]),
        )
        for trainer in self.league_trainers.values():
            for group in trainer.actor_optimizer.param_groups:
                group["lr"] = actor_learning_rate
            for group in trainer.critic_optimizer.param_groups:
                group["lr"] = critic_learning_rate
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

        moments = {
            "advantage": moment(),
            "normalized": moment(),
            "rank_target": moment(),
            "rank_residual": moment(),
        }
        action_statistics = {}
        game_counts = {
            "kyoku": 0.0, "player_kyoku": 0.0, "wins": 0.0,
            "deal_ins": 0.0, "riichi_hands": 0.0,
            "calling_hands": 0.0, "tsumo_wins": 0.0,
            "dama_wins": 0.0, "winning_points": 0.0,
            "winning_point_events": 0.0, "deal_in_points": 0.0,
            "deal_in_point_events": 0.0, "winning_turns": 0.0,
            "winning_turn_events": 0.0, "exhaustive_ryukyoku": 0.0,
            "player_matches": 0.0, "bankrupt_matches": 0.0,
        }
        totals = {
            "matches": 0, "kyoku": 0, "decisions": 0, "eligible": 0,
            "env_calls": 0, "model_queries": 0, "minibatches": 0,
            "total_tokens": 0, "padded_tokens": 0,
        }
        trajectory_digest = sha256()
        rollout_policy_digest = population_digest(self.league_models)
        self.env.metrics(reset=True)
        if self.update > 0:
            self.batch = self.adapter.reset(range(chunk_capacity))
        first_chunk = True
        try:
            while totals["matches"] < target_matches:
                chunk_matches = min(
                    chunk_capacity, target_matches - totals["matches"]
                )
                if not first_chunk:
                    self.batch = self.adapter.reset(range(chunk_matches))
                first_chunk = False
                collector = Collector(
                    self.adapter,
                    self.model,
                    self.streams.torch_generator("action", self.device),
                    device=self.device,
                    backend=self.profile.attention,
                    inference_token_budget=int(self.values["ppo"]["token_budget"]),
                    max_padding_fraction=float(
                        self.values["encoding"].get(
                            "inference_packing_max_waste", 0.5
                        )
                    ),
                    use_bf16=self.profile.precision == "bf16",
                    lineups=self.lineups,
                    lineup_provider=self._lineup,
                    policy_models=self.league_models,
                    opponent_agents={},
                    deterministic_policy_ids=self.deterministic_rollout_policy_ids,
                    profiler=self.profiler,
                    event_cache=self.event_cache,
                    diagnostic_dir=self.output / "diagnostics" / "native-env",
                )
                collection = collector.collect(
                    self.batch,
                    target_matches=chunk_matches,
                    curriculum=curriculum,
                    streams=self.streams,
                    current_policy_id=self.league.policy_ids[0],
                    max_env_calls=int(
                        self.values["rollout"]["max_frames_per_match"]
                    ),
                )
                if population_digest(self.league_models) != rollout_policy_digest:
                    raise RuntimeError("policy parameters changed during rollout collection")
                self.batch = collection.continuation
                self.lineups = collector.lineups
                outcomes = tuple(
                    outcome for outcome in collection.match_outcomes
                    if set(outcome.checkpoint_ids) <= set(self.league.policy_ids)
                )
                self.league.record(outcomes)
                with self.profiler.measure("targets.current_kyoku"):
                    advantages = compute(
                        collection.samples,
                        collection.frames,
                    )
                with self.profiler.measure("ppo.packing"):
                    eligible, packed, actor_batches, critic_batches = \
                        self._streaming_training_batches(collection, advantages)
                for policy_id, trainer in self.league_trainers.items():
                    owned = actor_batches.get(policy_id, ())
                    owned_critic = critic_batches.get(policy_id, ())
                    if owned:
                        trainer.accumulate_streaming_chunk(
                            owned, critic_minibatches=owned_critic
                        )

                add_moment(moments["advantage"], advantages.advantages)
                add_moment(moments["normalized"], advantages.normalized)
                predictions = np.asarray([
                    collection.samples[int(index)].old_boundary_rank_value
                    for index in advantages.indices
                ], dtype=np.float64)
                placements = np.asarray([
                    collection.samples[int(index)].terminal_placement
                    for index in advantages.indices
                ], dtype=np.int64)
                rank_targets = RANK_UTILITIES[placements]
                add_moment(moments["rank_target"], rank_targets)
                add_moment(
                    moments["rank_residual"], rank_targets - predictions
                )
                for name, value in action_family_statistics(
                    collection.samples, advantages
                ).items():
                    action_statistics[name] = action_statistics.get(name, 0.0) \
                        + float(value)

                kyoku = float(collection.kyoku_completions)
                player_kyoku = 4.0 * kyoku
                wins = collection.game_metrics["game/player_win_rate"] \
                    * player_kyoku
                deal_ins = collection.game_metrics["game/player_deal_in_rate"] \
                    * player_kyoku
                game_counts["kyoku"] += kyoku
                game_counts["player_kyoku"] += player_kyoku
                game_counts["wins"] += wins
                game_counts["deal_ins"] += deal_ins
                game_counts["riichi_hands"] += \
                    collection.game_metrics["game/player_riichi_rate"] * player_kyoku
                game_counts["calling_hands"] += \
                    collection.game_metrics["game/player_calling_rate"] * player_kyoku
                game_counts["tsumo_wins"] += \
                    collection.game_metrics["game/player_tsumo_rate"] * wins
                game_counts["dama_wins"] += \
                    collection.game_metrics["game/player_dama_rate"] * wins
                game_counts["winning_points"] += \
                    collection.game_metrics["game/player_average_winning_points"] * wins
                game_counts["winning_point_events"] += wins
                game_counts["deal_in_points"] += \
                    collection.game_metrics["game/player_average_deal_in_points"] * deal_ins
                game_counts["deal_in_point_events"] += deal_ins
                game_counts["winning_turns"] += collection.game_metrics[
                    "game/player_average_turns_before_winning"
                ] * wins
                game_counts["winning_turn_events"] += wins
                game_counts["exhaustive_ryukyoku"] += collection.game_metrics[
                    "game/exhaustive_ryukyoku_rate"
                ] * kyoku
                player_matches = 4.0 * collection.match_completions
                game_counts["player_matches"] += player_matches
                game_counts["bankrupt_matches"] += collection.game_metrics[
                    "game/player_bankrupt_rate"
                ] * player_matches

                totals["matches"] += collection.match_completions
                totals["kyoku"] += collection.kyoku_completions
                totals["decisions"] += collection.decisions
                totals["eligible"] += len(eligible)
                totals["env_calls"] += collection.env_calls
                totals["model_queries"] += collection.model_queries
                totals["minibatches"] += len(packed.batches)
                totals["total_tokens"] += packed.total_tokens
                totals["padded_tokens"] += packed.padded_tokens
                trajectory_digest.update(collection.trajectory_digest.encode("ascii"))

                # Every match in the chunk is terminal and its gradients and
                # sufficient statistics are now owned by the trainer.  Release
                # histories and encoded rows before launching the next chunk.
                self.adapter.histories.retain(())
                self.event_cache.retain(())
                self.lineups = {}
                del actor_batches, critic_batches, packed, advantages, collection
        except Exception:
            for trainer in self.league_trainers.values():
                trainer.abort_streaming_update()
            raise

        native_metrics = self.env.metrics(reset=True)
        target_variance = variance(moments["rank_target"])
        rank_ev = 0.0 if target_variance < 1e-12 else 1.0 - (
            variance(moments["rank_residual"]) / target_variance
        )
        for trainer in self.league_trainers.values():
            trainer.set_rollout_critic_evidence(rank_explained_variance=rank_ev)
        with self.profiler.measure("system.parameter_digest"):
            before = population_digest(self.league_models)
        with self.profiler.measure("ppo.optimization"):
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
        game_metrics = _game_metric_values(game_counts)
        call_total = action_statistics.get("call_count", 0.0) + \
            action_statistics.get("pass_count", 0.0)
        riichi_total = action_statistics.get("riichi_count", 0.0) + \
            action_statistics.get("dama_count", 0.0)
        elapsed = perf_counter() - started
        values = {
            "critic/match_rank_explained_variance": rank_ev,
            "rollout/current_kyoku_advantage_mean": mean(
                moments["advantage"]
            ),
            "rollout/current_kyoku_advantage_std": variance(
                moments["advantage"]
            ) ** 0.5,
            "rollout/policy_advantage_mean": mean(moments["normalized"]),
            "rollout/policy_advantage_std": variance(
                moments["normalized"]
            ) ** 0.5,
            "rollout/kyoku_completions": float(totals["kyoku"]),
            "rollout/match_completions": float(totals["matches"]),
            **game_metrics,
            "curriculum/progress": curriculum.progress,
            "population/pool_size": float(len(self.pool.snapshot().eligible)),
            **{f"league/{key}": value for key, value in self.league.metrics().items()},
            "performance/model_queries_per_second": totals["model_queries"] / max(
                elapsed, 1e-9
            ),
            "performance/rust_resolved_decisions": float(
                native_metrics["rust_resolved_decisions"]
            ),
            "performance/automatic_resolution_fraction": float(
                native_metrics["rust_resolved_decisions"]
            ) / max(
                1, int(native_metrics["rust_resolved_decisions"])
                + totals["model_queries"],
            ),
            "curriculum/actor_learning_rate": actor_learning_rate,
            "curriculum/critic_learning_rate": critic_learning_rate,
            "rollout/call_opportunity_selected_call_rate": (
                action_statistics.get("call_count", 0.0) / call_total
                if call_total else 0.0
            ),
            "rollout/riichi_opportunity_selected_riichi_rate": (
                action_statistics.get("riichi_count", 0.0) / riichi_total
                if riichi_total else 0.0
            ),
        }
        with self.profiler.measure("metrics.commit"):
            points = self.trainer.metric_points(
                self.completed_matches,
                update_result,
                source="completed-match-batch",
            )
            for name, value in values.items():
                definition = REGISTRY[name]
                points.append(MetricPoint(
                    name, definition.axis, self.completed_matches, value,
                    definition.unit, definition.window, definition.reduction,
                    "update",
                ))
            records = self.canonical.commit(points)
            if self.projector is not None:
                from .metrics import tensorboard_records
                self.projector.enqueue(tensorboard_records(records))
                tensorboard = self.values["metrics"]["tensorboard"]
                if _due(
                    self.completed_matches,
                    int(tensorboard["histogram_every_matches"]),
                ):
                    self.projector.enqueue_histogram(
                        "model/parameters",
                        next(self.model.parameters()).detach().float().cpu()
                        .numpy().reshape(-1),
                        self.completed_matches,
                        max_elements=tensorboard["histogram_max_elements"],
                        max_bytes=tensorboard["histogram_max_bytes"],
                        rng=self.streams.numpy_rng("histogram"),
                    )
        self._prune_episode_state()
        evidence = {
            "status": "passed",
            "profile": self.values["run"]["profile"],
            "device": self.device,
            "environment_count": int(self.values["env"]["num_envs"]),
            "rollout_chunk_matches": chunk_capacity,
            "rollout_chunks": (target_matches + chunk_capacity - 1) // chunk_capacity,
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
            "model_queries_per_second": totals["model_queries"] / max(
                elapsed, 1e-9
            ),
        }
        if self.profiler.enabled:
            evidence["profile"] = self.profiler.snapshot()
            evidence["event_prefix_cache"] = asdict(self.event_cache.stats)
        return evidence, curriculum

    def update_once(self):
        import torch

        from .encoding.packing import PackedBatch, pack
        from .ppo.current_kyoku import compute, rank_explained_variance
        from .rollout.collector import Collector

        started = perf_counter()
        update_number = self.update + 1
        target_matches = min(
            self.matches_per_update, self.total_matches - self.completed_matches
        )
        if target_matches <= 0:
            raise RuntimeError("match budget is already complete")
        if self.update > 0:
            self.env.metrics(reset=True)
            launch_count = min(
                int(self.values["env"]["num_envs"]), target_matches
            )
            self.batch = self.adapter.reset(range(launch_count))
        curriculum = self.curriculum.snapshot(
            self.completed_matches,
            max(trainer.policy_version for trainer in self.league_trainers.values()),
        )
        collector = Collector(
            self.adapter,
            self.model,
            self.streams.torch_generator("action", self.device),
            device=self.device,
            backend=self.profile.attention,
            inference_token_budget=int(self.values["ppo"]["token_budget"]),
            max_padding_fraction=float(
                self.values["encoding"].get(
                    "inference_packing_max_waste", 0.5
                )
            ),
            use_bf16=self.profile.precision == "bf16",
            lineups=self.lineups,
            lineup_provider=self._lineup,
            policy_models=self.league_models,
            opponent_agents={},
            deterministic_policy_ids=self.deterministic_rollout_policy_ids,
            profiler=self.profiler,
            event_cache=self.event_cache,
            diagnostic_dir=self.output / "diagnostics" / "native-env",
        )
        rollout_policy_digest = population_digest(self.league_models)
        collection = collector.collect(
            self.batch,
            target_matches=target_matches,
            curriculum=curriculum,
            streams=self.streams,
            current_policy_id=self.league.policy_ids[0],
            max_env_calls=int(self.values["rollout"]["max_frames_per_match"]),
        )
        if population_digest(self.league_models) != rollout_policy_digest:
            raise RuntimeError("policy parameters changed during rollout collection")
        native_metrics = self.env.metrics(reset=True)
        collection = replace(
            collection,
            rust_resolved_decisions=int(native_metrics["rust_resolved_decisions"]),
        )
        self.batch = collection.continuation
        self.lineups = collector.lineups
        league_outcomes = tuple(
            outcome for outcome in collection.match_outcomes
            if set(outcome.checkpoint_ids) <= set(self.league.policy_ids)
        )
        self.league.record(league_outcomes)
        eligible = [
            index for index, sample in enumerate(collection.samples)
            if sample.ppo_eligible
        ]
        with self.profiler.measure("targets.current_kyoku"):
            advantages = compute(
                collection.samples,
                collection.frames,
            )
            policy_advantage_by_sample = {
                int(sample): float(advantages.normalized[row])
                for row, sample in enumerate(advantages.indices)
            }
        eligible_by_policy = {
            policy_id: [
                index for index in eligible
                if collection.samples[index].checkpoint_id == policy_id
            ]
            for policy_id in self.league.policy_ids
        }
        eligible_by_policy = {
            policy_id: indices
            for policy_id, indices in eligible_by_policy.items() if indices
        }
        for policy_id, indices in eligible_by_policy.items():
            self.league_trainers[policy_id].set_rollout_critic_evidence(
                rank_explained_variance=rank_explained_variance(
                    [collection.samples[index].old_boundary_rank_value for index in indices],
                    [collection.samples[index].terminal_placement for index in indices],
                ),
            )
        with self.profiler.measure("ppo.packing"):
            actor_packed_by_policy = {
                policy_id: pack(
                    [len(collection.samples[index].encoded.token_factors)
                     for index in indices],
                    int(self.values["ppo"]["token_budget"]),
                    max_padding_fraction=float(
                        self.values["encoding"]["packing_max_waste"]
                    ),
                )
                for policy_id, indices in eligible_by_policy.items()
            }
            packed = PackedBatch(
                tuple(range(len(eligible))),
                tuple(
                    batch
                    for policy_packed in actor_packed_by_policy.values()
                    for batch in policy_packed.batches
                ),
                sum(value.total_tokens for value in actor_packed_by_policy.values()),
                sum(value.padded_tokens for value in actor_packed_by_policy.values()),
            )
        completed_after = self.completed_matches + collection.match_completions
        minibatches_by_policy = {}
        with self.profiler.measure("ppo.batch_transfer"):
            for policy_id, policy_packed in actor_packed_by_policy.items():
                policy_indices = eligible_by_policy[policy_id]
                policy_minibatches = []
                for packed_rows in policy_packed.batches:
                    sample_indices = [policy_indices[row] for row in packed_rows]
                    samples = [collection.samples[index] for index in sample_indices]
                    action_counts = torch.tensor(
                        [len(sample.encoded.action_factors) for sample in samples],
                        dtype=torch.long,
                    )
                    action_starts = torch.cat((
                        torch.zeros(1, dtype=torch.long),
                        action_counts.cumsum(0)[:-1],
                    ))
                    selected = action_starts + torch.tensor(
                        [sample.selected_group for sample in samples],
                        dtype=torch.long,
                    )
                    minibatch = {
                    "encoded_rows": tuple(sample.encoded for sample in samples),
                    "backend": self.profile.attention,
                    "action_counts": action_counts,
                    "selected": selected,
                    "old_logp": torch.tensor(
                        [sample.old_log_probability for sample in samples],
                        dtype=torch.float32,
                    ),
                    "advantages": torch.tensor(
                        [policy_advantage_by_sample[index] for index in sample_indices],
                        dtype=torch.float32,
                    ),
                }
                    policy_minibatches.append(minibatch)
                materialized_rows = sum(
                    int(batch["old_logp"].numel())
                    for batch in policy_minibatches
                )
                if materialized_rows != len(policy_indices):
                    raise RuntimeError(
                        "actor packing dropped learner rows: "
                        f"materialized={materialized_rows}, "
                        f"eligible={len(policy_indices)}"
                    )
                minibatches_by_policy[policy_id] = policy_minibatches
        critic_minibatches_by_policy = {}
        boundary_candidates_by_policy = {
            policy_id: [
                index for index, frame in enumerate(collection.frames)
                if frame.ppo_eligible
                and frame.checkpoint_id == policy_id
                and frame.rank_boundary_supervision
                and 0 <= frame.rank_order_target < 24
            ]
            for policy_id in self.league.policy_ids
        }
        critic_indices_by_policy = boundary_candidates_by_policy
        critic_indices_by_policy = {
            policy_id: indices
            for policy_id, indices in critic_indices_by_policy.items() if indices
        }
        critic_packed_by_policy = {
            policy_id: pack(
                [len(collection.frames[index].encoded.token_factors)
                 for index in indices],
                int(self.values["ppo"]["token_budget"]),
                max_padding_fraction=float(
                    self.values["encoding"]["packing_max_waste"]
                ),
            )
            for policy_id, indices in critic_indices_by_policy.items()
        }
        with self.profiler.measure("ppo.critic_batch_transfer"):
            for policy_id, policy_packed in critic_packed_by_policy.items():
                policy_indices = critic_indices_by_policy[policy_id]
                policy_minibatches = []
                for packed_rows in policy_packed.batches:
                    frame_indices = [policy_indices[row] for row in packed_rows]
                    rows = [collection.frames[index] for index in frame_indices]
                    policy_minibatches.append({
                    "encoded_rows": tuple(row.encoded for row in rows),
                    "backend": self.profile.attention,
                    "include_actions": False,
                    "boundary_only": True,
                    "rank_boundary_supervision": torch.tensor(
                        [row.rank_boundary_supervision for row in rows],
                        dtype=torch.bool,
                    ),
                    "rank_order_targets": torch.tensor(
                        [row.rank_order_target for row in rows],
                        dtype=torch.long,
                    ),
                })
                critic_minibatches_by_policy.setdefault(policy_id, []).extend(
                    policy_minibatches
                )
        actor_learning_rate = _learning_rate(
            float(self.values["ppo"]["actor_learning_rate"]),
            completed_after,
            self.total_matches,
            float(self.values["ppo"]["warmup_fraction"]),
        )
        critic_learning_rate = _learning_rate(
            float(self.values["ppo"]["critic_learning_rate"]),
            completed_after,
            self.total_matches,
            float(self.values["ppo"]["warmup_fraction"]),
        )
        for trainer in self.league_trainers.values():
            for group in trainer.actor_optimizer.param_groups:
                group["lr"] = actor_learning_rate
            for group in trainer.critic_optimizer.param_groups:
                group["lr"] = critic_learning_rate
        with self.profiler.measure("system.parameter_digest"):
            before = population_digest(self.league_models)
        with self.profiler.measure("ppo.optimization"):
            update_results = {}
            for policy_id, trainer in self.league_trainers.items():
                owned = minibatches_by_policy.get(policy_id, ())
                owned_critic = critic_minibatches_by_policy.get(policy_id, ())
                if owned:
                    update_results[policy_id] = trainer.update(
                        owned,
                        critic_minibatches=owned_critic,
                        actor_rng=self.streams.python_rng("actor_minibatch"),
                        critic_rng=self.streams.python_rng("critic_minibatch"),
                        ema_matches=collection.match_completions,
                    )
            update_result = _merge_update_results(update_results)
        with self.profiler.measure("system.parameter_digest"):
            after = population_digest(self.league_models)
        if not update_result.committed:
            raise RuntimeError(f"PPO update rejected: {update_result.reason}")
        if before == after:
            raise RuntimeError("committed PPO update did not change parameters")
        self._advance_ema_opponent(collection.match_completions)
        self.update = update_number
        self.completed_matches = completed_after
        self.environment_decisions += collection.decisions
        with self.profiler.measure("metrics.commit"):
            self._emit_update_metrics(
                collection, advantages, eligible, packed, curriculum, update_result,
                actor_learning_rate, critic_learning_rate,
                started,
            )
        self._prune_episode_state()
        evidence = {
            "status": "passed",
            "profile": self.values["run"]["profile"],
            "device": self.device,
            "environment_count": int(self.values["env"]["num_envs"]),
            "learner_decisions": collection.decisions,
            "eligible_decisions": len(eligible),
            "env_calls": collection.env_calls,
            "trajectory_digest": collection.trajectory_digest,
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
                "minibatches": len(packed.batches),
                "total_tokens": packed.total_tokens,
                "padded_tokens": packed.padded_tokens,
            },
            "elapsed_seconds": perf_counter() - started,
        }
        evidence["model_queries_per_second"] = collection.model_queries / max(
            evidence["elapsed_seconds"], 1e-9
        )
        if self.profiler.enabled:
            evidence["profile"] = self.profiler.snapshot()
            evidence["event_prefix_cache"] = asdict(self.event_cache.stats)
        return evidence, curriculum

    def _prune_episode_state(self):
        active = {
            (int(state.environment_id), int(state.episode_generation))
            for state in self.batch.transition.states
        }
        self.adapter.histories.retain(active)
        self.event_cache.retain(active)
        self.lineups = {
            key: lineup for key, lineup in self.lineups.items() if key in active
        }

    def _emit_update_metrics(
        self, collection, advantages, eligible, packed, curriculum, update_result,
        actor_learning_rate, critic_learning_rate, started,
    ):
        from .metric_registry import REGISTRY
        from .ppo.current_kyoku import (
            action_family_diagnostics,
            rank_explained_variance,
        )
        from .types import MetricPoint

        values = {
            "critic/match_rank_explained_variance": rank_explained_variance(
                [
                    collection.samples[int(index)].old_boundary_rank_value
                    for index in advantages.indices
                ],
                [
                    collection.samples[int(index)].terminal_placement
                    for index in advantages.indices
                ],
            ),
            "rollout/current_kyoku_advantage_mean": (
                float(advantages.advantages.mean())
                if len(advantages.advantages) else 0.0
            ),
            "rollout/current_kyoku_advantage_std": (
                float(advantages.advantages.std())
                if len(advantages.advantages) else 0.0
            ),
            "rollout/policy_advantage_mean": (
                float(advantages.normalized.mean())
                if len(advantages.normalized) else 0.0
            ),
            "rollout/policy_advantage_std": (
                float(advantages.normalized.std())
                if len(advantages.normalized) else 0.0
            ),
            "rollout/kyoku_completions": float(collection.kyoku_completions),
            "rollout/match_completions": float(collection.match_completions),
            **collection.game_metrics,
            "curriculum/progress": curriculum.progress,
            "population/pool_size": float(len(self.pool.snapshot().eligible)),
            **{
                f"league/{key}": value
                for key, value in self.league.metrics().items()
            },
            "performance/model_queries_per_second": collection.model_queries / max(
                perf_counter() - started, 1e-9
            ),
            "performance/rust_resolved_decisions": float(collection.rust_resolved_decisions),
            "performance/automatic_resolution_fraction": collection.rust_resolved_decisions / max(
                1, collection.rust_resolved_decisions + collection.model_queries
            ),
            "curriculum/actor_learning_rate": actor_learning_rate,
            "curriculum/critic_learning_rate": critic_learning_rate,
            **action_family_diagnostics(collection.samples, advantages),
        }
        points = self.trainer.metric_points(
            self.completed_matches, update_result, source="completed-match-batch"
        )
        for name, value in values.items():
            definition = REGISTRY[name]
            points.append(MetricPoint(
                name, definition.axis, self.completed_matches, value, definition.unit,
                definition.window, definition.reduction, "update",
            ))
        records = self.canonical.commit(points)
        if self.projector is not None:
            from .metrics import tensorboard_records
            self.projector.enqueue(tensorboard_records(records))
            tensorboard = self.values["metrics"]["tensorboard"]
            if _due(self.completed_matches, int(tensorboard["histogram_every_matches"])):
                self.projector.enqueue_histogram(
                    "model/parameters",
                    next(self.model.parameters()).detach().float().cpu().numpy().reshape(-1),
                    self.completed_matches,
                    max_elements=tensorboard["histogram_max_elements"],
                    max_bytes=tensorboard["histogram_max_bytes"],
                    rng=self.streams.numpy_rng("histogram"),
                )

    def _checkpoint(self, curriculum):
        # Only the dense launch prefix has initialized hanchan state.  Saving
        # unused capacity creates snapshots that cannot be projected under a
        # privileged exact resume.
        env_ids = list(range(min(
            int(self.values["env"]["num_envs"]),
            int(self.matches_per_update),
            int(self.total_matches),
        )))
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
                "snapshots": dict(self.env.snapshot(env_ids)),
                "histories": self.adapter.histories.state_dict(),
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
        from .evaluation.runner import paired_bootstrap, run_series_batched
        from .model.actor_critic import ActorCritic

        evaluation_device = _evaluation_device(self.config, self.device)
        evaluation_batch_size = int(
            self.values["evaluation"].get("batch_size", 32)
        )

        def evaluate_series(checkpoint_ids, models, *, series_id):
            return run_series_batched(
                checkpoint_ids,
                self.values["evaluation"]["held_out_seeds"],
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
        if len(eligible) < 4:
            return {
                "status": "completed",
                "bc_baseline": baseline_report,
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
        }

    def run(self, *, max_updates=None):
        limit = self.total_updates
        if max_updates is not None:
            limit = min(limit, self.update + int(max_updates))
        last_evidence = None
        last_curriculum = None
        while (self.completed_matches < self.total_matches
               and self.update < limit and not self.stop_requested):
            last_evidence, last_curriculum = self.update_once_streaming()
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
        self.env.close()


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
