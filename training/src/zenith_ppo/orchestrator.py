"""Durable multi-update PPO training orchestration."""

from __future__ import annotations

from dataclasses import asdict, replace
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


def parameter_digest(model) -> str:
    return sha256(
        b"".join(
            value.detach().cpu().contiguous().numpy().tobytes()
            for value in model.state_dict().values()
        )
    ).hexdigest()


def _due(update: int, cadence: int) -> bool:
    return cadence > 0 and update % cadence == 0


def _learning_rate(base: float, update: int, total: int, warmup_fraction: float) -> float:
    warmup = int(total * warmup_fraction)
    if warmup and update <= warmup:
        return base * update / warmup
    remaining = max(1, total - warmup)
    return base * max(0.0, total - update + 1) / remaining


def _entropy_coefficient(start: float, end: float, update: int, total: int) -> float:
    progress = min(max(update / max(1, total), 0.0), 1.0)
    return start + (end - start) * progress


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
        self, config, output, *, resume=None, weights_only=False, profile_stages=False
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
        self.stop_requested = False
        self.update = 0
        self.environment_decisions = 0
        self.completed_matches = 0
        self.last_checkpoint_id = None
        self.outcomes = []
        self.rating_transactions = []
        self.rating_snapshot_id = None
        self.lineups = {}
        from .rollout.rating import ConservativeBotRating
        self.rollout_rating = ConservativeBotRating()

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
        self.trainer = PPOTrainer(
            self.model,
            self.values["ppo"],
            device_type=self.device,
            use_bf16=self.profile.precision == "bf16",
            profiler=self.profiler,
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
            self.adapter.restore(env_state["snapshots"])
        launch_count = min(self.matches_per_update, self.total_matches - self.completed_matches)
        self.batch = self.adapter.reset(range(launch_count))
        self.sampler = SelfPlaySampler(self.streams)
        writer_session = self._next_writer_session()
        self.writer_session = writer_session
        self.canonical = CanonicalMetrics(
            self.output / "metrics" / "canonical.jsonl",
            run_id=config.digest,
            writer_session=writer_session,
        )
        tensorboard = self.values["metrics"]["tensorboard"]
        self.projector = None
        if tensorboard["enabled"]:
            self.projector = TensorBoardProjector(
                self.output / "tensorboard",
                run_id=config.digest,
                writer_session=writer_session,
                flush_seconds=tensorboard["flush_seconds"],
                purge_step=self.update + 1 if resume is not None else None,
            )
        if restored is None:
            from .checkpoint import publish

            initial_id = publish(
                self.output / "checkpoints",
                {
                    "model": self.model.state_dict(),
                    "state": {"architecture": "contextual-actor-shared-oracle-v1",
                              "update": 0, "completed_matches": 0,
                              "purpose": "frozen-random-initialization"},
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
        from .rollout.rating import ConservativeBotRating

        self.model.load_state_dict(restored["model"])
        if weights_only:
            return
        if restored.get("state", {}).get("architecture") != "contextual-actor-shared-oracle-v1":
            raise RuntimeError("resume checkpoint predates the current actor/oracle architecture")
        self.trainer.load_optimizer_state_dict(restored["optimizer"])
        trainer_state = restored["trainer"]
        self.trainer.policy_version = int(trainer_state.get("policy_version", 0))
        self.update = int(restored["state"].get("update", 0))
        counters = trainer_state.get("counters") or {}
        self.environment_decisions = int(counters.get("environment_decisions", 0))
        self.completed_matches = int(counters.get("completed_matches", 0))
        if trainer_state.get("seeds") is not None:
            self.streams.load_state_dict(trainer_state["seeds"])
        population = trainer_state.get("population") or {}
        if population.get("pool"):
            self.pool = CheckpointPool.from_state_dict(population["pool"])
        self.lineups = _restore_lineups(population.get("lineups"))
        rating = trainer_state.get("rating") or {}
        if rating.get("table"):
            self.ratings = RatingTable.from_state_dict(rating["table"])
        if rating.get("rollout_bot"):
            self.rollout_rating = ConservativeBotRating.from_state_dict(
                rating["rollout_bot"]
            )
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
            current_id="current",
            policy_version=self.trainer.policy_version,
        )

    def update_once(self):
        import numpy as np
        import torch

        from .encoding.packing import model_batch, pack
        from .ppo.gae import compute
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
            self.batch = self.adapter.reset(range(target_matches))
        curriculum = self.curriculum.snapshot(
            self.completed_matches, self.trainer.policy_version
        )
        self.sampler.conservative_bot_match_fraction = curriculum.bot_fraction
        collector = Collector(
            self.adapter,
            self.model,
            self.streams.torch_generator("action", self.device),
            device=self.device,
            backend=self.profile.attention,
            inference_token_budget=int(self.values["ppo"]["token_budget"]),
            max_padding_fraction=float(
                self.values["encoding"]["packing_max_waste"]
            ),
            use_bf16=self.profile.precision == "bf16",
            lineups=self.lineups,
            lineup_provider=self._lineup,
            profiler=self.profiler,
            event_cache=self.event_cache,
            teacher_config=self.values["teacher"],
            diagnostic_dir=self.output / "diagnostics" / "native-env",
        )
        collection = collector.collect(
            self.batch,
            target_matches=target_matches,
            curriculum=curriculum,
            streams=self.streams,
            current_policy_id="current",
            critic_mode=self.values["observation"]["critic_mode"],
            max_env_calls=int(self.values["rollout"]["max_frames_per_match"]),
        )
        native_metrics = self.env.metrics(reset=True)
        collection = replace(
            collection,
            rust_resolved_decisions=int(native_metrics["rust_resolved_decisions"]),
        )
        self.batch = collection.continuation
        self.lineups = collector.lineups
        with self.profiler.measure("ratings.rollout"):
            self.rollout_rating.update(collection.match_outcomes)
        with self.profiler.measure("targets.gae"):
            advantages = compute(
                collection.samples,
                gamma=float(self.values["ppo"]["gamma"]),
                score_gae_lambda=float(self.values["ppo"]["score_gae_lambda"]),
                rank_gae_lambda=float(self.values["ppo"]["rank_gae_lambda"]),
            )
            advantage_by_sample = {
                int(sample): (
                    float(advantages.normalized[row]),
                    float(advantages.score_returns[row]),
                    float(advantages.rank_returns[row]),
                )
                for row, sample in enumerate(advantages.indices)
            }
            eligible = [
                index for index, sample in enumerate(collection.samples)
                if sample.ppo_eligible
            ]
        with self.profiler.measure("ppo.packing"):
            packed = pack(
                [len(collection.samples[index].encoded.token_factors) for index in eligible],
                int(self.values["ppo"]["token_budget"]),
                max_padding_fraction=float(
                    self.values["encoding"]["packing_max_waste"]
                ),
            )
        completed_after = self.completed_matches + collection.match_completions
        entropy = _entropy_coefficient(
            float(self.values["ppo"]["entropy_start"]),
            float(self.values["ppo"]["entropy_end"]),
            completed_after,
            self.total_matches,
        )
        from .teachers import (
            coefficients as teacher_coefficients,
            pack_targets,
            rollout_metrics as teacher_rollout_metrics,
        )
        auxiliary_coefficients = teacher_coefficients(
            self.values["teacher"], guidance_scale=curriculum.guidance_scale
        )
        minibatches = []
        with self.profiler.measure("ppo.batch_transfer"):
            for packed_rows in packed.batches:
                sample_indices = [eligible[row] for row in packed_rows]
                samples = [collection.samples[index] for index in sample_indices]
                inputs = model_batch(
                    [sample.encoded for sample in samples],
                    device=self.device,
                    backend=self.profile.attention,
                )
                selected = inputs["action_offsets"][:-1] + torch.tensor(
                    [sample.selected_group for sample in samples],
                    dtype=torch.long, device=self.device,
                )
                minibatch = {
                    "model_inputs": inputs,
                    "selected": selected,
                    "old_logp": torch.tensor(
                        [sample.old_log_probability for sample in samples],
                        dtype=torch.float32,
                        device=self.device,
                    ),
                    "old_score_values": torch.tensor(
                        [sample.old_score_value for sample in samples],
                        dtype=torch.float32,
                        device=self.device,
                    ),
                    "old_rank_values": torch.tensor(
                        [sample.old_rank_value for sample in samples],
                        dtype=torch.float32,
                        device=self.device,
                    ),
                    "advantages": torch.tensor(
                        [advantage_by_sample[index][0] for index in sample_indices],
                        dtype=torch.float32,
                        device=self.device,
                    ),
                    "score_returns": torch.tensor(
                        [advantage_by_sample[index][1] for index in sample_indices],
                        dtype=torch.float32,
                        device=self.device,
                    ),
                    "rank_returns": torch.tensor(
                        [advantage_by_sample[index][2] for index in sample_indices],
                        dtype=torch.float32,
                        device=self.device,
                    ),
                    "rank_targets": torch.tensor(
                        [sample.terminal_placement for sample in samples],
                        dtype=torch.long, device=self.device,
                    ),
                    "opponent_count_targets": torch.as_tensor(
                        np.stack([sample.encoded.opponent_count_targets for sample in samples]),
                        dtype=torch.long, device=self.device,
                    ),
                    "opponent_tenpai_targets": torch.as_tensor(
                        np.stack([sample.encoded.opponent_tenpai_targets for sample in samples]),
                        dtype=torch.float32, device=self.device,
                    ),
                    "entropy_coefficient": entropy,
                    "teacher_coefficients": auxiliary_coefficients,
                }
                if any(auxiliary_coefficients.values()):
                    minibatch["teacher_targets"] = pack_targets(
                        tuple(sample.encoded.teachers for sample in samples),
                        tuple(len(sample.encoded.action_factors) for sample in samples),
                        device=self.device,
                    )
                minibatches.append(minibatch)
        learning_rate = _learning_rate(
            float(self.values["ppo"]["learning_rate"]),
            completed_after,
            self.total_matches,
            float(self.values["ppo"]["warmup_fraction"]),
        )
        for optimizer in (
            self.trainer.actor_optimizer, self.trainer.critic_optimizer
        ):
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
        with self.profiler.measure("system.parameter_digest"):
            before = parameter_digest(self.model)
        with self.profiler.measure("ppo.optimization"):
            update_result = self.trainer.update(
                minibatches, rng=self.streams.python_rng("minibatch")
            )
        with self.profiler.measure("system.parameter_digest"):
            after = parameter_digest(self.model)
        if not update_result.committed:
            raise RuntimeError(f"PPO update rejected: {update_result.reason}")
        if before == after:
            raise RuntimeError("committed PPO update did not change parameters")
        self.update = update_number
        self.completed_matches = completed_after
        self.environment_decisions += collection.decisions
        teacher_observation = teacher_rollout_metrics(
            collection.samples, collection.kyoku_completions
        )
        self.curriculum.observe(
            applicable_rows=int(teacher_observation["teacher/discard_applicable_rows"]),
            worse_shanten_rate=float(teacher_observation["teacher/discard_worse_shanten_rate"]),
            completed_matches=collection.match_completions,
        )
        with self.profiler.measure("metrics.commit"):
            self._emit_update_metrics(
                collection, advantages, eligible, packed, curriculum, update_result,
                learning_rate, started
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
            "rollout_opponents": "self-play-with-conservative-bot-probes",
            "policy_version": update_result.policy_version,
            "update": self.update,
            "parameter_digest_before": before,
            "parameter_digest_after": after,
            "ppo": update_result.metrics,
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
        learning_rate, started
    ):
        from .metric_registry import REGISTRY
        from .metrics import completed_match_metric_values
        from .ppo.gae import explained_variance
        from .teachers import rollout_metrics as teacher_rollout_metrics
        from .types import MetricPoint

        reward_values = [
            sample.reward.total for sample in collection.samples if sample.ppo_eligible
        ]
        learner_samples = [sample for sample in collection.samples if sample.ppo_eligible]
        match_ids = {
            (sample.binding.environment_id, sample.binding.episode_generation)
            for sample in collection.samples
        }
        bot_matches = {
            (sample.binding.environment_id, sample.binding.episode_generation)
            for sample in collection.samples if sample.checkpoint_id == "conservative_bot"
        }
        rollout_progress = self.rollout_rating.metrics()
        values = {
            "ppo/score_explained_variance": explained_variance(
                [
                    collection.samples[int(index)].old_score_value
                    for index in advantages.indices
                ],
                advantages.score_returns,
            ),
            "ppo/rank_explained_variance": explained_variance(
                [
                    collection.samples[int(index)].old_rank_value
                    for index in advantages.indices
                ],
                advantages.rank_returns,
            ),
            "rollout/reward_mean": float(sum(reward_values) / max(1, len(reward_values))),
            "rollout/kyoku_reward_mean": float(sum(
                sample.reward.weights[0] * sample.reward.kyoku_delta
                for sample in learner_samples
            ) / max(1, len(learner_samples))),
            "rollout/rank_reward_mean": float(sum(
                sample.reward.weights[1] * sample.reward.rank_reward
                for sample in learner_samples
            ) / max(1, len(learner_samples))),
            "rollout/score_return_mean": (
                float(advantages.score_returns.mean())
                if len(advantages.score_returns) else 0.0
            ),
            "rollout/rank_return_mean": (
                float(advantages.rank_returns.mean())
                if len(advantages.rank_returns) else 0.0
            ),
            "rollout/score_advantage_mean": (
                float(advantages.score_advantages.mean())
                if len(advantages.score_advantages) else 0.0
            ),
            "rollout/rank_advantage_mean": (
                float(advantages.rank_advantages.mean())
                if len(advantages.rank_advantages) else 0.0
            ),
            "rollout/policy_advantage_mean": (
                float(advantages.advantages.mean()) if len(advantages.advantages) else 0.0
            ),
            "rollout/kyoku_completions": float(collection.kyoku_completions),
            "rollout/match_completions": float(collection.match_completions),
            **completed_match_metric_values(collection.match_outcomes),
            **collection.game_metrics,
            **teacher_rollout_metrics(collection.samples, collection.kyoku_completions),
            **{f"rollout_rating/{key}": float(value)
               for key, value in rollout_progress.items()},
            "curriculum/progress": curriculum.progress,
            "curriculum/kyoku_weight": curriculum.weights[0],
            "curriculum/rank_weight": curriculum.weights[1],
            "curriculum/guidance_scale": curriculum.guidance_scale,
            "curriculum/competence_streak": float(curriculum.competence_streak),
            "curriculum/regression_streak": float(curriculum.regression_streak),
            "curriculum/taper_progress": curriculum.taper_progress,
            "curriculum/last_valid_worse_shanten_rate": float(
                curriculum.last_valid_worse_shanten_rate or 0.0
            ),
            "population/pool_size": float(len(self.pool.snapshot().eligible)),
            "population/conservative_bot_match_fraction": len(bot_matches) / max(1, len(match_ids)),
            "population/target_bot_fraction": curriculum.bot_fraction,
            "performance/model_queries_per_second": collection.model_queries / max(
                perf_counter() - started, 1e-9
            ),
            "performance/rust_resolved_decisions": float(collection.rust_resolved_decisions),
            "performance/automatic_resolution_fraction": collection.rust_resolved_decisions / max(
                1, collection.rust_resolved_decisions + collection.model_queries
            ),
            "system/learning_rate": learning_rate,
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
            self.projector.enqueue(records)
            tensorboard = self.values["metrics"]["tensorboard"]
            if _due(self.completed_matches, int(tensorboard["histogram_every_matches"])):
                self.projector.enqueue_histogram(
                    "model/parameters",
                    next(self.model.parameters()).detach().float().cpu().numpy().reshape(-1),
                    self.update,
                    max_elements=tensorboard["histogram_max_elements"],
                    max_bytes=tensorboard["histogram_max_bytes"],
                    rng=self.streams.numpy_rng("histogram"),
                )

    def _checkpoint(self, curriculum):
        env_ids = list(range(int(self.values["env"]["num_envs"])))
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
            },
            rating={
                "table": self.ratings.state_dict(),
                "rollout_bot": self.rollout_rating.state_dict(),
                "outcomes": self.outcomes,
                "transactions": self.rating_transactions,
                "snapshot_id": self.rating_snapshot_id,
            },
            metrics={
                "canonical_sequence": self.canonical.sequence,
                "writer_session": self.writer_session,
            },
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
        from .checkpoint import publish_evaluation_records
        from .cli.evaluate import _load_models, _play_game
        from .evaluation.runner import run_series

        eligible = sorted(
            self.pool.snapshot().eligible,
            key=lambda entry: (entry.source_update, entry.checkpoint_id),
        )
        if len(eligible) < 4:
            return {"status": "skipped", "reason": "fewer than four checkpoints"}
        participants = eligible[-4:]
        paths = [entry.artifact for entry in participants]
        models, records = _load_models(self.config, paths)
        checkpoint_ids = tuple(record["checkpoint_id"] for record in records)
        series_id = f"held-out-{self.update:08d}"
        outcomes = run_series(
            checkpoint_ids,
            self.values["evaluation"]["held_out_seeds"],
            lambda lineup, seed, **contract: _play_game(
                models, self.values["model"], lineup, seed, **contract
            ),
            series_id=series_id,
        )
        valid = [outcome for outcome in outcomes if outcome["valid"]]
        self.ratings.update(valid)
        self.outcomes.extend(outcomes)
        self.rating_transactions.extend(valid)
        self.rating_snapshot_id = self.pool.publish_rating_snapshot(self.ratings)
        evaluation_root = self.output / "evaluations"
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
        }

    def run(self, *, max_updates=None):
        limit = self.total_updates
        if max_updates is not None:
            limit = min(limit, self.update + int(max_updates))
        last_evidence = None
        last_curriculum = None
        while (self.completed_matches < self.total_matches
               and self.update < limit and not self.stop_requested):
            last_evidence, last_curriculum = self.update_once()
            checkpoint_due = _due(
                self.completed_matches,
                int(self.values["checkpoint"]["cadence_matches"]),
            )
            scheduled = self.values["evaluation"]["checkpoint_matches"]
            evaluation_due = (
                self.completed_matches in scheduled if scheduled else _due(
                    self.completed_matches, int(self.values["evaluation"]["cadence_matches"])
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
                policy_version=self.trainer.policy_version,
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
            "policy_version": self.trainer.policy_version,
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
    profile_stages=False
):
    orchestrator = TrainingOrchestrator(
        config,
        output,
        resume=resume,
        weights_only=weights_only,
        profile_stages=profile_stages,
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
