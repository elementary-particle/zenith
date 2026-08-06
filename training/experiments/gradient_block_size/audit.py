"""Full-actor gradient block-size and PPO replay-epoch audit.

Run from the repository root. The block-size phase collects independent
complete-match chunks at every requested size and standardizes raw advantages
over the complete block, matching logical-batch production normalization.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import json
import math
from pathlib import Path
import random
from time import perf_counter

import numpy as np


@dataclass
class GradientMoments:
    count: int = 0
    total: np.ndarray | None = None
    alternating_total: np.ndarray | None = None
    sum_norm_squared: float = 0.0

    def add(self, gradient: np.ndarray) -> None:
        gradient = np.asarray(gradient, dtype=np.float32)
        if self.total is None:
            self.total = np.zeros_like(gradient)
            self.alternating_total = np.zeros_like(gradient)
        self.total += gradient
        if self.count % 2 == 0:
            self.alternating_total += gradient
        self.sum_norm_squared += _dot(gradient, gradient)
        self.count += 1

    def mean(self) -> np.ndarray | None:
        return None if self.total is None else self.total / self.count


def _dot(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.sum(
        left.astype(np.float64) * right.astype(np.float64), dtype=np.float64,
    ))


def cosine(left: np.ndarray | None, right: np.ndarray | None) -> float | None:
    if left is None or right is None:
        return None
    denominator = math.sqrt(max(_dot(left, left) * _dot(right, right), 0.0))
    return _dot(left, right) / denominator if denominator else None


def gradient_report(moment: GradientMoments, matches_per_block: int) -> dict:
    blocks = moment.count
    result = {
        "blocks": blocks,
        "matches_per_block": int(matches_per_block),
        "total_matches": blocks * int(matches_per_block),
    }
    if blocks < 2 or moment.total is None or moment.alternating_total is None:
        return result
    mean = moment.total / blocks
    mean_norm_squared = _dot(mean, mean)
    sample_variance_trace = max(
        0.0,
        (moment.sum_norm_squared - _dot(moment.total, moment.total) / blocks)
        / (blocks - 1),
    )
    noise_trace_per_match = matches_per_block * sample_variance_trace
    signal_squared = mean_norm_squared - sample_variance_trace / blocks
    first_count = (blocks + 1) // 2
    second_count = blocks // 2
    first = moment.alternating_total / first_count
    second = (moment.total - moment.alternating_total) / second_count
    critical = (
        noise_trace_per_match / signal_squared if signal_squared > 0 else None
    )
    result.update({
        "mean_gradient_norm_squared": mean_norm_squared,
        "sample_gradient_variance_trace": sample_variance_trace,
        "gradient_noise_trace_per_match": noise_trace_per_match,
        "debiased_gradient_signal_squared": signal_squared,
        "critical_batch_matches": critical,
        "snr_at_block_size": (
            math.sqrt(matches_per_block / critical) if critical else None
        ),
        "snr_at_total_observation": (
            math.sqrt(blocks * matches_per_block / critical) if critical else None
        ),
        "split_half_gradient_cosine": cosine(first, second),
    })
    return result


def _checkpoint_path(path: Path) -> Path:
    if (path / "latest").is_file():
        return path / (path / "latest").read_text(encoding="ascii").strip()
    return path


def _model(config, state, device):
    from zenith_ppo.model.actor_critic import ActorCritic

    model_config = dict(config.values["model"])
    model_config["context_tokens"] = config.values["encoding"]["context_tokens"]
    model = ActorCritic(model_config).to(device)
    model.load_state_dict(state)
    return model


def _to_device(value, device):
    import torch

    if isinstance(value, dict):
        return {name: _to_device(item, device) for name, item in value.items()}
    if isinstance(value, str):
        return value
    if isinstance(value, np.ndarray):
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="The given NumPy array is not writable"
            )
            value = torch.as_tensor(value)
    if torch.is_tensor(value):
        return value.to(device=device)
    return value


def _policy_gradient(model, batches, *, matches, device, use_bf16):
    import torch

    model.requires_grad_(True).eval()
    model.zero_grad(set_to_none=True)
    rows = 0
    advantage_sum = advantage_square_sum = 0.0
    raw = np.concatenate([
        np.asarray(batch["raw_advantages"], dtype=np.float64).reshape(-1)
        for batch in batches
    ])
    mean = float(raw.mean())
    deviation = float(raw.std())
    for host_batch in batches:
        batch = _to_device(host_batch, device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_bf16,
        ):
            output = model.forward_actor(**batch["model_inputs"])
            selected = batch["selected"].long()
            raw_advantages = batch["raw_advantages"].float()
            advantages = (
                torch.zeros_like(raw_advantages)
                if deviation < 1e-8
                else (raw_advantages - mean) / (deviation + 1e-8)
            )
            logp = output.log_probabilities.index_select(0, selected).float()
            loss = -(logp * advantages).sum() / matches
        loss.backward()
        rows += int(advantages.numel())
        advantage_sum += float(advantages.sum().detach().cpu())
        advantage_square_sum += float(advantages.square().sum().detach().cpu())
    pieces = []
    for _, parameter in model.actor_named_parameters():
        gradient = parameter.grad
        pieces.append(
            torch.zeros_like(parameter).flatten().cpu()
            if gradient is None
            else gradient.detach().float().flatten().cpu()
        )
    gradient = torch.cat(pieces).numpy()
    model.zero_grad(set_to_none=True)
    return gradient, {
        "actor_rows": rows,
        "rows_per_match": rows / matches,
        "advantage_mean": advantage_sum / max(rows, 1),
        "advantage_rms": math.sqrt(advantage_square_sum / max(rows, 1)),
        "gradient_norm": math.sqrt(_dot(gradient, gradient)),
    }


def _collect(model, config, profile, *, matches, seed, action_seed):
    import riichi
    import torch

    from zenith_ppo.rollout.native import NativeInferenceRunner

    values = config.values
    device = torch.device(profile.device)
    engine = riichi.RolloutEngine(
        matches,
        master_seed=int(seed),
        num_threads=min(int(values["env"]["num_threads"]), matches),
        rules_profile=values["env"]["rules_profile"],
        context_tokens=int(values["encoding"]["context_tokens"]),
        token_budget=int(values["ppo"]["token_budget"]),
    )
    match_ids = engine.reset_chunk(matches)
    engine.register_lineups(match_ids, [(0, 0, 0, 0)] * matches, [15] * matches)
    generator = torch.Generator(device=device).manual_seed(int(action_seed))
    runner = NativeInferenceRunner(
        {0: model},
        device=device,
        backend=profile.attention,
        use_bf16=profile.precision == "bf16",
        generator=generator,
        compile_cuda=False,
    )
    chunk = runner.run_chunk(engine)
    runner.prepare_training_chunk(chunk)
    actor_batches = tuple(chunk.actor_minibatches(
        0,
        int(values["ppo"]["token_budget"]),
        max_padding_fraction=float(values["encoding"]["packing_max_waste"]),
        backend=profile.attention,
    ))
    return engine, runner, chunk, actor_batches


def _surrogate(model, batches, *, device, use_bf16, ratio_clip):
    import torch

    totals = {"rows": 0, "objective": 0.0, "clipped_objective": 0.0,
              "approximate_kl": 0.0, "clip_rows": 0}
    model.eval()
    with torch.inference_mode():
        for host_batch in batches:
            batch = _to_device(host_batch, device)
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16,
            ):
                output = model.forward_actor(**batch["model_inputs"])
            selected = batch["selected"].long()
            new_logp = output.log_probabilities.index_select(0, selected).float()
            old_logp = batch["old_logp"].float()
            advantages = batch["advantages"].float()
            log_ratio = new_logp - old_logp
            ratio = log_ratio.exp()
            clipped = ratio.clamp(1.0 - ratio_clip, 1.0 + ratio_clip)
            objective = ratio * advantages
            clipped_objective = torch.minimum(objective, clipped * advantages)
            rows = int(advantages.numel())
            totals["rows"] += rows
            totals["objective"] += float(objective.sum().cpu())
            totals["clipped_objective"] += float(clipped_objective.sum().cpu())
            totals["approximate_kl"] += float(
                ((ratio - 1.0) - log_ratio).sum().cpu()
            )
            totals["clip_rows"] += int(
                ((ratio - 1.0).abs() > ratio_clip).sum().cpu()
            )
    rows = max(1, totals["rows"])
    return {
        "rows": totals["rows"],
        "importance_objective": totals["objective"] / rows,
        "clipped_importance_objective": totals["clipped_objective"] / rows,
        "approximate_kl": totals["approximate_kl"] / rows,
        "clip_fraction": totals["clip_rows"] / rows,
    }


def _trainer_compatible_batches(batches):
    """Tensorize fields inspected before PPOTrainer materializes a batch.

    The native rollout intentionally exposes NumPy views. The streaming path
    materializes them inside ``_process_actor``, while the retained-rollout
    multi-epoch path computes entropy row counts first. Keep large arrays on
    the host and tensorize only the two fields used by that pre-pass.
    """
    import torch

    result = []
    for batch in batches:
        batch = dict(batch)
        inputs = dict(batch["model_inputs"])
        for name in ("action_factors", "action_lengths"):
            inputs[name] = torch.as_tensor(np.asarray(inputs[name]))
        batch["model_inputs"] = inputs
        result.append(batch)
    return tuple(result)


def _actor_rms_distance(model, reference_state):
    squared = elements = 0
    for name, parameter in model.actor_named_parameters():
        difference = parameter.detach().float().cpu() - reference_state[name].float()
        squared += float(difference.square().sum())
        elements += difference.numel()
    return math.sqrt(squared / max(elements, 1))


def _epoch_audit(
    args, config, profile, restored, baseline_model, *, seed_offset,
):
    from zenith_ppo.ppo.trainer import PPOTrainer
    from zenith_ppo.seeds import derive_seed

    matches = int(args.epoch_audit_matches)
    import torch
    device = torch.device(profile.device)
    _, _, train_chunk, train_batches = _collect(
        baseline_model, config, profile,
        matches=matches,
        seed=derive_seed(args.seed + seed_offset, "epoch_train_environment"),
        action_seed=derive_seed(args.seed + seed_offset, "epoch_train_action"),
    )
    train_batches = _trainer_compatible_batches(train_batches)
    critic_batch = train_chunk.critic_batch(policy_slot=0)
    _, _, validation_chunk, validation_batches = _collect(
        baseline_model, config, profile,
        matches=matches,
        seed=derive_seed(args.seed + seed_offset, "epoch_validation_environment"),
        action_seed=derive_seed(args.seed + seed_offset, "epoch_validation_action"),
    )
    del validation_chunk
    reference_actor = {
        name: parameter.detach().float().cpu().clone()
        for name, parameter in baseline_model.actor_named_parameters()
    }
    use_bf16 = profile.precision == "bf16"
    results = []
    for epochs in args.actor_epochs:
        model = _model(config, restored["model"], device)
        ppo = dict(config.values["ppo"])
        ppo["epochs"] = int(epochs)
        ppo["minibatches"] = int(args.epoch_minibatches)
        trainer = PPOTrainer(
            model, ppo, device_type=device.type,
            use_bf16=use_bf16,
        )
        optimizer_state = "fresh"
        try:
            trainer.load_optimizer_state_dict(restored["optimizer"])
            optimizer_state = "checkpoint"
        except (KeyError, ValueError):
            pass
        actor_learning_rate = float(
            args.epoch_actor_learning_rate
            if args.epoch_actor_learning_rate is not None
            else ppo["actor_learning_rate"]
        )
        for group in trainer.actor_optimizer.param_groups:
            group["lr"] = actor_learning_rate
        before = _surrogate(
            model, validation_batches, device=device,
            use_bf16=use_bf16, ratio_clip=float(ppo["ratio_clip"]),
        )
        update = trainer.update(
            train_batches,
            critic_minibatches=(critic_batch,),
            rng=random.Random(args.seed + epochs),
            ema_matches=matches,
        )
        after = _surrogate(
            model, validation_batches, device=device,
            use_bf16=use_bf16, ratio_clip=float(ppo["ratio_clip"]),
        )
        results.append({
            "actor_epochs": int(epochs),
            "minibatches": int(args.epoch_minibatches),
            "optimizer_state": optimizer_state,
            "actor_learning_rate": actor_learning_rate,
            "committed": update.committed,
            "reason": update.reason,
            "update_metrics": update.metrics,
            "held_out_before": before,
            "held_out_after": after,
            "held_out_clipped_objective_change": (
                after["clipped_importance_objective"]
                - before["clipped_importance_objective"]
            ),
            "actor_parameter_rms_distance": _actor_rms_distance(
                model, reference_actor
            ),
        })
        del model, trainer
        gc.collect()
    del train_chunk
    return {"matches": matches, "arms": results}


def run(args) -> dict:
    import torch

    from zenith_ppo.capabilities import configure
    from zenith_ppo.checkpoint import restore
    from zenith_ppo.config import load
    from zenith_ppo.seeds import derive_seed

    started = perf_counter()
    config = load(args.config)
    profile = configure(config.values["run"]["profile"])
    device = torch.device(profile.device)
    checkpoint = _checkpoint_path(Path(args.checkpoint))
    restored = restore(checkpoint)
    model = _model(config, restored["model"], device)
    model.eval()
    moments = {size: GradientMoments() for size in args.block_sizes}
    history = []
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    use_bf16 = profile.precision == "bf16"

    sequence = 0
    for size in args.block_sizes:
        for block in range(args.blocks_per_size):
            sequence += 1
            block_started = perf_counter()
            environment_seed = derive_seed(
                args.seed + sequence, f"gradient_environment_{size}_{block}"
            )
            action_seed = derive_seed(
                args.seed + sequence, f"gradient_action_{size}_{block}"
            )
            print(
                f"gradient block size={size} block={block + 1}/"
                f"{args.blocks_per_size}", flush=True,
            )
            engine, runner, chunk, batches = _collect(
                model, config, profile, matches=size,
                seed=environment_seed, action_seed=action_seed,
            )
            gradient, diagnostics = _policy_gradient(
                model, batches, matches=size, device=device,
                use_bf16=use_bf16,
            )
            moments[size].add(gradient)
            history.append({
                "block_size": size,
                "block": block + 1,
                "environment_seed": environment_seed,
                "action_seed": action_seed,
                "seconds": perf_counter() - block_started,
                **diagnostics,
            })
            del gradient, batches, chunk, runner, engine
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            partial = {
                "status": "running",
                "protocol": vars(args),
                "history": history,
                "block_sizes": {
                    str(value): gradient_report(moments[value], value)
                    for value in args.block_sizes
                },
            }
            output.write_text(
                json.dumps(partial, indent=2, sort_keys=True), encoding="utf-8"
            )

    reference_size = max(args.block_sizes)
    reference = moments[reference_size].mean()
    reports = {}
    for size in args.block_sizes:
        reports[str(size)] = gradient_report(moments[size], size) | {
            "mean_gradient_cosine_to_largest_block_mean": cosine(
                moments[size].mean(), reference
            ),
        }
    epoch_audit = None
    if args.actor_epochs:
        epoch_audit = _epoch_audit(
            args, config, profile, restored, model, seed_offset=sequence + 1,
        )
    result = {
        "status": "complete",
        "schema": "zenith-gradient-block-size-audit-v1",
        "protocol": {
            **vars(args),
            "checkpoint": str(checkpoint),
            "checkpoint_id": restored["manifest"]["checkpoint_id"],
            "gradient_scope": "all actor parameters, policy term only",
            "advantages": "production logical-block standardized current-kyoku",
            "independent_collection_per_size": True,
            "device": str(device),
            "precision": profile.precision,
            "actor_parameter_count": int(next(
                moment.total.size for moment in moments.values()
                if moment.total is not None
            )),
        },
        "block_sizes": reports,
        "epoch_replay": epoch_audit,
        "history": history,
        "seconds": perf_counter() - started,
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def _integer_list(value: str) -> tuple[int, ...]:
    result = tuple(dict.fromkeys(int(item) for item in value.split(",") if item))
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return result


def _optional_integer_list(value: str) -> tuple[int, ...]:
    if not value:
        return ()
    return _integer_list(value)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="training/configs/default.toml")
    parser.add_argument(
        "--checkpoint", default="runs/behavior-cloning-rank-v/checkpoints",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument(
        "--block-sizes", type=_integer_list, default=(64, 128, 256, 512),
    )
    parser.add_argument("--blocks-per-size", type=int, default=8)
    parser.add_argument(
        "--actor-epochs", type=_optional_integer_list, default=(1, 2, 4),
        help="Empty string disables the fixed-rollout replay audit.",
    )
    parser.add_argument("--epoch-audit-matches", type=int, default=512)
    parser.add_argument("--epoch-minibatches", type=int, default=1)
    parser.add_argument(
        "--epoch-actor-learning-rate", type=float,
        help=(
            "Actor LR for fixed-rollout replay; defaults to the configured "
            "base LR instead of a checkpoint's schedule-decayed optimizer LR."
        ),
    )
    args = parser.parse_args()
    if args.blocks_per_size < 2:
        parser.error("--blocks-per-size must be at least two")
    return args


if __name__ == "__main__":
    run(parse_args())
