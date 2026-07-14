"""Seeded native-env PPO training orchestration."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import platform
import subprocess
from time import perf_counter

from ..capabilities import configure
from ..config import load


def parameter_digest(model) -> str:
    return sha256(
        b"".join(
            value.detach().cpu().contiguous().numpy().tobytes()
            for value in model.state_dict().values()
        )
    ).hexdigest()


def run_one_update(config, output: str | Path, *, resume=None, weights_only=False):
    import numpy as np
    import torch
    import riichi

    from ..encoding.packing import mean_token_length, model_batch, pack, padding_fraction
    from ..env.adapter import EnvAdapter
    from ..model.actor_critic import ActorCritic
    from ..ppo.gae import compute
    from ..ppo.trainer import PPOTrainer
    from ..rewards.curriculum import Curriculum
    from ..rollout.collector import Collector
    from ..seeds import SeedStreams

    started = perf_counter()
    values = config.values
    profile = configure(values["run"]["profile"])
    device = "cuda" if profile.device == "cuda" else "cpu"
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    run_manifest = output / "run.json"
    if run_manifest.exists():
        existing = json.loads(run_manifest.read_text(encoding="utf-8"))
        if existing.get("config_digest") != config.digest:
            raise RuntimeError(f"run directory {output} belongs to another configuration")
    lock_name = "requirements-cuda.lock" if device == "cuda" else "requirements-cpu.lock"
    lock_path = Path("training") / lock_name
    lock_digest = sha256(lock_path.read_bytes()).hexdigest()
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
        ).stdout)
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = "unavailable", True
    from ..checkpoint import reproducibility_metadata

    metadata = reproducibility_metadata(
        input_config=config.source,
        resolved_config=config.redacted(),
        source_revision=revision,
        dirty=dirty,
        dependency_lock_digest=lock_digest,
        capabilities={**profile.as_dict(), "host": platform.platform()},
        level="weights_only" if weights_only else "exact_same_target",
    )
    run_manifest.write_text(json.dumps({
        "run_id": config.digest,
        "config_digest": config.digest,
        "profile": profile.as_dict(),
        "source_revision": revision,
        "source_dirty": dirty,
        "dependency_lock": lock_name,
        "dependency_lock_digest": lock_digest,
    }, sort_keys=True, indent=2), encoding="utf-8")
    (output / "resolved-config.json").write_text(
        json.dumps(config.redacted(), sort_keys=True, indent=2), encoding="utf-8"
    )
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    streams = SeedStreams(int(values["run"]["seed"]))
    torch.manual_seed(streams.root)
    if device == "cuda":
        torch.cuda.manual_seed_all(streams.root)
    model_config = dict(values["model"])
    model_config["context_tokens"] = values["encoding"]["context_tokens"]
    model = ActorCritic(model_config).to(device)
    trainer = PPOTrainer(
        model,
        values["ppo"],
        device_type=device,
        use_bf16=profile.precision == "bf16",
    )
    update_index = 0
    if resume is not None:
        from ..checkpoint import resolve_latest, restore

        resume_path = Path(resume)
        if (resume_path / "latest").is_file():
            resume_path = resolve_latest(resume_path)
        restored = restore(resume_path, expected=config.compatibility)
        model.load_state_dict(restored["model"])
        if not weights_only:
            trainer.optimizer.load_state_dict(restored["optimizer"])
            trainer.policy_version = int(
                restored["trainer"].get("policy_version", 0)
            )
            update_index = int(restored["state"].get("update", 0))
            seed_state = restored["trainer"].get("seeds")
            if seed_state is not None:
                streams.load_state_dict(seed_state)
    before = parameter_digest(model)
    env_config = values["env"]
    env = riichi.Env(
        int(env_config["num_envs"]),
        master_seed=streams.python_rng("env").getrandbits(64),
        num_threads=int(env_config["num_threads"]),
        rules_profile=env_config["rules_profile"],
        privileged=bool(env_config["privileged"]),
    )
    adapter = EnvAdapter(env)
    initial = adapter.reset(range(int(env_config["num_envs"])))
    curriculum = Curriculum(values["curriculum"]).snapshot(
        update_index, trainer.policy_version
    )
    collector = Collector(
        adapter,
        model,
        streams.torch_generator("action", device),
        device=device,
        backend=profile.attention,
        inference_token_budget=int(values["ppo"]["token_budget"]),
        max_padding_fraction=float(values["encoding"]["packing_max_waste"]),
        use_bf16=profile.precision == "bf16",
    )
    collection = collector.collect(
        initial,
        target_decisions=int(values["rollout"]["learner_decisions_per_update"]),
        curriculum=curriculum,
        streams=streams,
        critic_mode=values["observation"]["critic_mode"],
        complete_kyoku_per_env=bool(
            values["rollout"]["complete_kyoku_per_env"]
        ),
        max_env_calls=int(values["rollout"]["max_frames_per_call"]),
    )
    advantages = compute(
        collection.samples,
        gamma=float(values["ppo"]["gamma"]),
        gae_lambda=float(values["ppo"]["gae_lambda"]),
    )
    advantage_by_sample = {
        int(sample): (float(advantages.normalized[row]), float(advantages.returns[row]))
        for row, sample in enumerate(advantages.indices)
    }
    eligible = [
        index for index, sample in enumerate(collection.samples) if sample.ppo_eligible
    ]
    packed = pack(
        [len(collection.samples[index].encoded.token_factors) for index in eligible],
        int(values["ppo"]["token_budget"]),
        max_padding_fraction=float(values["encoding"]["packing_max_waste"]),
    )
    minibatches = []
    for packed_rows in packed.batches:
        sample_indices = [eligible[row] for row in packed_rows]
        samples = [collection.samples[index] for index in sample_indices]
        inputs = model_batch(
            [sample.encoded for sample in samples],
            device=device,
            backend=profile.attention,
        )
        selected = inputs["action_offsets"][:-1] + torch.tensor(
            [sample.selected_group for sample in samples],
            dtype=torch.long, device=device,
        )
        minibatches.append(
            {
                "model_inputs": inputs,
                "selected": selected,
                "old_logp": torch.tensor(
                    [sample.old_log_probability for sample in samples],
                    dtype=torch.float32,
                    device=device,
                ),
                "advantages": torch.tensor(
                    [advantage_by_sample[index][0] for index in sample_indices],
                    dtype=torch.float32,
                    device=device,
                ),
                "returns": torch.tensor(
                    [advantage_by_sample[index][1] for index in sample_indices],
                    dtype=torch.float32,
                    device=device,
                ),
                "opponent_count_targets": torch.as_tensor(
                    np.stack([sample.encoded.opponent_count_targets for sample in samples]),
                    dtype=torch.long, device=device,
                ),
                "opponent_tenpai_targets": torch.as_tensor(
                    np.stack([sample.encoded.opponent_tenpai_targets for sample in samples]),
                    dtype=torch.float32, device=device,
                ),
            }
        )
    update = trainer.update(minibatches)
    after = parameter_digest(model)
    if not update.committed:
        raise RuntimeError(f"PPO update rejected: {update.reason}")
    if before == after:
        raise RuntimeError("committed PPO update did not change parameters")
    metrics = env.metrics()
    env.close()
    evidence = {
        "status": "passed",
        "profile": values["run"]["profile"],
        "device": device,
        "environment_count": int(env_config["num_envs"]),
        "learner_decisions": collection.decisions,
        "eligible_decisions": len(eligible),
        "env_calls": collection.env_calls,
        "trajectory_digest": collection.trajectory_digest,
        "policy_version": update.policy_version,
        "update": update_index + 1,
        "parameter_digest_before": before,
        "parameter_digest_after": after,
        "ppo": update.metrics,
        "env": metrics,
        "packing": {
            "minibatches": len(packed.batches),
            "total_tokens": packed.total_tokens,
            "padded_tokens": packed.padded_tokens,
        },
    }
    (output / "one-update.json").write_text(
        json.dumps(evidence, sort_keys=True, indent=2), encoding="utf-8"
    )
    from ..metric_registry import REGISTRY
    from ..metrics import CanonicalMetrics, TensorBoardProjector
    from ..types import MetricPoint

    writer_session = f"session-{update_index:08d}"
    canonical = CanonicalMetrics(
        output / "metrics" / "canonical.jsonl",
        run_id=config.digest,
        writer_session=writer_session,
    )
    elapsed = max(perf_counter() - started, 1e-9)
    reward_values = [sample.reward.total for sample in collection.samples if sample.ppo_eligible]
    metric_values = {
        "rollout/decisions": float(collection.decisions),
        "rollout/ppo_eligible": float(len(eligible)),
        "rollout/reward_mean": float(sum(reward_values) / max(1, len(reward_values))),
        "rollout/return_mean": float(advantages.returns.mean()) if len(advantages.returns) else 0.0,
        "rollout/advantage_mean": float(advantages.advantages.mean()) if len(advantages.advantages) else 0.0,
        "rollout/kyoku_completions": float(collection.kyoku_completions),
        "rollout/match_completions": float(collection.match_completions),
        "rollout/kyoku_environment_coverage": collection.kyoku_environment_coverage,
        "rollout/boundary_aligned": float(collection.boundary_aligned),
        "curriculum/progress": curriculum.progress,
        "curriculum/discard_weight": curriculum.weights[0],
        "curriculum/kyoku_weight": curriculum.weights[1],
        "curriculum/rank_weight": curriculum.weights[2],
        "population/pool_size": 0.0,
        "population/historical_fraction": 1.0 - len(eligible) / max(1, collection.decisions),
        "population/checkpoint_cohort_size": 0.0,
        "population/inference_model_count": 1.0,
        "encoding/mean_token_length": mean_token_length(packed, len(eligible)),
        "encoding/padding_fraction": padding_fraction(packed),
        "performance/decisions_per_second": collection.decisions / elapsed,
        "system/learning_rate": float(trainer.optimizer.param_groups[0]["lr"]),
    }
    points = trainer.metric_points(update_index + 1, update)
    for name, value in metric_values.items():
        definition = REGISTRY[name]
        points.append(MetricPoint(
            name, definition.axis, update_index + 1, value, definition.unit,
            definition.window, definition.reduction, "update",
        ))
    records = canonical.commit(points)
    tensorboard_config = values["metrics"]["tensorboard"]
    writer_failure = None
    if tensorboard_config["enabled"]:
        try:
            projector = TensorBoardProjector(
                output / "tensorboard" / writer_session,
                run_id=config.digest,
                writer_session=writer_session,
                flush_seconds=tensorboard_config["flush_seconds"],
            )
            projector.enqueue(records)
            if (update_index + 1) % tensorboard_config["histogram_every_updates"] == 0:
                projector.enqueue_histogram(
                    "model/parameters",
                    next(model.parameters()).detach().float().cpu().numpy().reshape(-1),
                    update_index + 1,
                    max_elements=tensorboard_config["histogram_max_elements"],
                    max_bytes=tensorboard_config["histogram_max_bytes"],
                    rng=streams.numpy_rng("histogram"),
                )
            projector.close()
        except Exception as exc:
            writer_failure = f"{type(exc).__name__}: {exc}"
            if tensorboard_config["runtime_failure"] != "canonical_only_at_safe_boundary":
                raise
    evidence["metrics"] = {
        "canonical_records": len(records),
        "last_sequence": canonical.sequence - 1,
        "writer_session": writer_session,
        "tensorboard_failure": writer_failure,
    }
    checkpoint_id = trainer.publish_checkpoint(
        output / "checkpoints",
        compatibility=config.compatibility,
        metadata=metadata,
        counters={"update": update_index + 1},
        seeds=streams,
        curriculum={
            "update": curriculum.update,
            "progress": curriculum.progress,
            "weights": curriculum.weights,
            "boundary_mode": curriculum.boundary_mode,
        },
        metrics={"canonical_sequence": canonical.sequence, "writer_session": writer_session},
        env={"master_seed_stream": "env", "safe_boundary": True},
    )
    checkpoint_path = output / "checkpoints" / checkpoint_id
    evidence["checkpoint_id"] = checkpoint_id
    evidence["checkpoint_path"] = str(checkpoint_path)
    evidence["elapsed_seconds"] = perf_counter() - started
    evidence["decisions_per_second"] = (
        collection.decisions / max(evidence["elapsed_seconds"], 1e-9)
    )
    if device == "cuda":
        evidence["cuda_memory"] = {
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        }
    (output / "one-update.json").write_text(
        json.dumps(evidence, sort_keys=True, indent=2), encoding="utf-8"
    )
    return evidence


def main(argv=None):
    parser = argparse.ArgumentParser(prog="zenith-ppo-train")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--weights-only", action="store_true")
    parser.add_argument(
        "--max-updates",
        type=int,
        help="stop after this many updates in this process (for bounded runs/tests)",
    )
    args = parser.parse_args(argv)
    from ..orchestrator import run_training

    run_training(
        load(args.config),
        args.output,
        resume=args.resume,
        weights_only=args.weights_only,
        max_updates=args.max_updates,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
