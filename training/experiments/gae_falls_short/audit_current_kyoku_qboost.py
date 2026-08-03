"""Cross-fitted Mahjong audit of current-kyoku Q-boosting at lambda one."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np

from current_kyoku_qboost import kyoku_segments, lambda_one_advantages


@dataclass
class GradientMoments:
    count: int = 0
    sum: np.ndarray | None = None
    first_half_sum: np.ndarray | None = None
    sum_norm_squared: float = 0.0

    def add(self, value: np.ndarray) -> None:
        value = np.asarray(value, dtype=np.float32)
        if self.sum is None:
            self.sum = np.zeros_like(value)
            self.first_half_sum = np.zeros_like(value)
        self.sum += value
        if self.count % 2 == 0:
            self.first_half_sum += value
        self.sum_norm_squared += float(
            np.sum(value.astype(np.float64) ** 2, dtype=np.float64)
        )
        self.count += 1


def _dot(left, right) -> float:
    return float(np.sum(
        left.astype(np.float64) * right.astype(np.float64), dtype=np.float64
    ))


def _gradient_report(moment: GradientMoments, matches_per_block: int) -> dict:
    blocks = moment.count
    if blocks < 2 or moment.sum is None or moment.first_half_sum is None:
        return {"blocks": blocks, "matches": blocks * matches_per_block}
    mean = moment.sum / blocks
    mean_norm_squared = _dot(mean, mean)
    sample_variance_trace = max(
        0.0,
        (
            moment.sum_norm_squared
            - _dot(moment.sum, moment.sum) / blocks
        ) / (blocks - 1),
    )
    noise_trace = matches_per_block * sample_variance_trace
    signal_squared = mean_norm_squared - sample_variance_trace / blocks
    first_count = (blocks + 1) // 2
    second_count = blocks // 2
    first = moment.first_half_sum / first_count
    second = (moment.sum - moment.first_half_sum) / second_count
    denominator = math.sqrt(max(_dot(first, first) * _dot(second, second), 0.0))
    return {
        "blocks": blocks,
        "matches": blocks * matches_per_block,
        "mean_gradient_norm_squared": mean_norm_squared,
        "gradient_noise_trace_per_match": noise_trace,
        "debiased_gradient_signal_squared": signal_squared,
        "critical_batch_matches": (
            noise_trace / signal_squared if signal_squared > 0 else None
        ),
        "snr_at_observed_batch": (
            math.sqrt(blocks * matches_per_block * signal_squared / noise_trace)
            if signal_squared > 0 and noise_trace > 0 else None
        ),
        "split_half_gradient_cosine": (
            _dot(first, second) / denominator if denominator else None
        ),
    }


def _standardize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    deviation = float(values.std(dtype=np.float64))
    if deviation < 1e-8:
        return np.zeros_like(values)
    return np.asarray(
        (values - values.mean(dtype=np.float64)) / (deviation + 1e-8),
        dtype=np.float32,
    )


def _checkpoint_path(path: Path) -> Path:
    if (path / "latest").is_file():
        checkpoint_id = (path / "latest").read_text(encoding="ascii").strip()
        return path / checkpoint_id
    return path


def _collect(
    model,
    *,
    matches: int,
    seed: int,
    values: dict,
    profile,
    streams,
    diagnostic_dir: Path,
):
    import riichi

    from zenith_ppo.env.adapter import EnvAdapter
    from zenith_ppo.rollout.collector import Collector
    from zenith_ppo.types import CurriculumSnapshot

    env_count = min(int(values["env"]["num_envs"]), matches)
    env = riichi.Env(
        env_count,
        master_seed=int(seed),
        num_threads=min(int(values["env"]["num_threads"]), env_count),
        rules_profile=values["env"]["rules_profile"],
        privileged=True,
    )
    adapter = EnvAdapter(env)
    initial = adapter.reset(range(env_count))
    collector = Collector(
        adapter,
        model,
        streams.torch_generator("action", profile.device),
        device=profile.device,
        backend=profile.attention,
        inference_token_budget=int(values["ppo"]["token_budget"]),
        max_padding_fraction=float(
            values["encoding"].get("inference_packing_max_waste", 0.5)
        ),
        use_bf16=profile.precision == "bf16",
        diagnostic_dir=diagnostic_dir,
    )
    result = collector.collect(
        initial,
        target_matches=matches,
        curriculum=CurriculumSnapshot(0, 0.0, 0),
        streams=streams,
        current_policy_id="current",
        max_env_calls=int(values["rollout"]["max_frames_per_match"]),
    )
    return result, env


def _packed_rows(collection, token_budget, packing_waste):
    from zenith_ppo.encoding.packing import pack

    indices = tuple(
        index for index, sample in enumerate(collection.samples)
        if sample.ppo_eligible
    )
    plan = pack(
        [len(collection.samples[index].encoded.token_factors) for index in indices],
        token_budget,
        max_padding_fraction=packing_waste,
    )
    return indices, tuple(
        tuple(indices[local] for local in shard) for shard in plan.batches
    )


def _actor_inputs(collection, shard, device, backend):
    from zenith_ppo.encoding.packing import model_batch

    return model_batch(
        [collection.samples[index].encoded for index in shard],
        device=device,
        backend=backend,
    )


def _selected_flat(output, inputs, samples, shard):
    import torch

    offsets = inputs["action_offsets"].long()
    groups = torch.as_tensor(
        [samples[index].selected_group for index in shard],
        dtype=torch.long,
        device=offsets.device,
    )
    return offsets[:-1] + groups


def _train_q_head(
    model,
    q_head,
    collection,
    *,
    epochs,
    learning_rate,
    device,
    backend,
    use_bf16,
    token_budget,
    packing_waste,
):
    import torch

    from zenith_ppo.ppo.current_kyoku import compute

    advantages = compute(collection.samples, collection.frames)
    targets = {
        int(index): float(value)
        for index, value in zip(
            advantages.indices, advantages.end_values, strict=True
        )
    }
    indices, shards = _packed_rows(collection, token_budget, packing_waste)
    if set(indices) != set(targets):
        raise ValueError("Q training rows and current-kyoku targets differ")
    optimizer = torch.optim.AdamW(q_head.parameters(), lr=learning_rate)
    model.requires_grad_(False).eval()
    q_head.train()
    history = []
    for epoch in range(epochs):
        squared_error = 0.0
        rows = 0
        for shard in shards:
            inputs = _actor_inputs(collection, shard, device, backend)
            with torch.no_grad(), torch.autocast(
                device_type=device,
                dtype=torch.bfloat16,
                enabled=use_bf16,
            ):
                output = model.forward_actor(**inputs)
            with torch.autocast(
                device_type=device,
                dtype=torch.bfloat16,
                enabled=use_bf16,
            ):
                padded_q = q_head(output.action_states).squeeze(-1).float()
                local = torch.arange(len(shard), device=device)
                groups = torch.as_tensor(
                    [collection.samples[index].selected_group for index in shard],
                    dtype=torch.long,
                    device=device,
                )
                selected_q = padded_q[local, groups]
                target = torch.as_tensor(
                    [targets[index] for index in shard],
                    dtype=torch.float32,
                    device=device,
                )
                loss = torch.nn.functional.mse_loss(selected_q, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(q_head.parameters(), 2.0)
            optimizer.step()
            squared_error += float(
                torch.sum((selected_q.detach() - target) ** 2).cpu()
            )
            rows += len(shard)
        history.append({"epoch": epoch + 1, "mse": squared_error / rows})
        print(f"Q epoch {epoch + 1}/{epochs}: mse={history[-1]['mse']:.6f}", flush=True)
    return history


def _q_predictions(
    model,
    q_head,
    collection,
    shards,
    dense_by_sample,
    *,
    device,
    backend,
    use_bf16,
):
    import torch

    selected = np.empty(len(dense_by_sample), dtype=np.float32)
    expected = np.empty(len(dense_by_sample), dtype=np.float32)
    model.requires_grad_(False).eval()
    q_head.eval()
    with torch.no_grad():
        for shard in shards:
            inputs = _actor_inputs(collection, shard, device, backend)
            with torch.autocast(
                device_type=device,
                dtype=torch.bfloat16,
                enabled=use_bf16,
            ):
                output = model.forward_actor(**inputs)
                padded_q = q_head(output.action_states).squeeze(-1).float()
            offsets = inputs["action_offsets"].long()
            for local, sample_index in enumerate(shard):
                dense = dense_by_sample[sample_index]
                length = int(inputs["action_lengths"][local])
                start, end = int(offsets[local]), int(offsets[local + 1])
                probabilities = output.log_probabilities[start:end].float().exp()
                values = padded_q[local, :length]
                group = int(collection.samples[sample_index].selected_group)
                selected[dense] = float(values[group].cpu())
                expected[dense] = float((probabilities * values).sum().cpu())
    return selected, expected


def _policy_gradient(
    model,
    collection,
    shards,
    advantage_by_sample,
    *,
    matches,
    device,
    backend,
    use_bf16,
):
    import torch

    model.requires_grad_(True).eval()
    model.zero_grad(set_to_none=True)
    for shard in shards:
        inputs = _actor_inputs(collection, shard, device, backend)
        with torch.autocast(
            device_type=device,
            dtype=torch.bfloat16,
            enabled=use_bf16,
        ):
            output = model.forward_actor(**inputs)
            chosen = _selected_flat(output, inputs, collection.samples, shard)
            logp = output.log_probabilities.index_select(0, chosen).float()
            weights = torch.as_tensor(
                [advantage_by_sample[index] for index in shard],
                dtype=torch.float32,
                device=device,
            )
            loss = -(logp * weights).sum() / matches
        loss.backward()
    pieces = []
    for _, parameter in model.actor_named_parameters():
        gradient = parameter.grad
        pieces.append(
            torch.zeros_like(parameter, memory_format=torch.contiguous_format)
            .flatten().cpu()
            if gradient is None
            else gradient.detach().float().flatten().cpu()
        )
    return torch.cat(pieces).numpy()


def _correlation(left, right):
    if len(left) < 2 or float(np.std(left)) < 1e-12 or float(np.std(right)) < 1e-12:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def run(args) -> dict:
    import torch
    from torch import nn

    from zenith_ppo.capabilities import configure
    from zenith_ppo.checkpoint import restore
    from zenith_ppo.config import load
    from zenith_ppo.model.actor_critic import ActorCritic
    from zenith_ppo.ppo.current_kyoku import compute, explained_variance
    from zenith_ppo.seeds import SeedStreams, derive_seed

    started = perf_counter()
    config = load(args.config)
    values = config.values
    profile = configure(values["run"]["profile"])
    device = profile.device
    use_bf16 = profile.precision == "bf16"
    checkpoint_path = _checkpoint_path(Path(args.checkpoint))
    restored = restore(checkpoint_path)
    model_config = dict(values["model"])
    model_config["context_tokens"] = values["encoding"]["context_tokens"]
    model = ActorCritic(model_config).to(device)
    model.load_state_dict(restored["model"])
    d_model = int(values["model"]["d_model"])
    torch.manual_seed(args.seed)
    q_head = nn.Sequential(
        nn.RMSNorm(d_model),
        nn.Linear(d_model, args.q_width),
        nn.SiLU(),
        nn.Linear(args.q_width, 1),
    ).to(device)
    streams = SeedStreams(args.seed)
    token_budget = int(values["ppo"]["token_budget"])
    packing_waste = float(values["encoding"]["packing_max_waste"])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    diagnostic_dir = output.parent / "native-env-diagnostics"

    print(f"collecting {args.critic_matches} cross-fit critic matches", flush=True)
    critic_collection, critic_env = _collect(
        model,
        matches=args.critic_matches,
        seed=derive_seed(args.seed, "visibility"),
        values=values,
        profile=profile,
        streams=streams,
        diagnostic_dir=diagnostic_dir,
    )
    q_training = _train_q_head(
        model,
        q_head,
        critic_collection,
        epochs=args.critic_epochs,
        learning_rate=args.critic_learning_rate,
        device=device,
        backend=profile.attention,
        use_bf16=use_bf16,
        token_budget=token_budget,
        packing_waste=packing_waste,
    )
    critic_rows = sum(sample.ppo_eligible for sample in critic_collection.samples)
    del critic_collection, critic_env
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    names = (
        "current_raw", "qboost_raw",
        "current_standardized", "qboost_standardized",
    )
    moments = {name: GradientMoments() for name in names}
    error_norm_sums = {"raw": 0.0, "standardized": 0.0}
    advantage_rows = []
    q_rows = []
    history = []
    for block in range(args.audit_blocks):
        block_started = perf_counter()
        block_seed = derive_seed(args.seed + block + 1, "visibility")
        print(
            f"audit block {block + 1}/{args.audit_blocks}: "
            f"collecting {args.matches_per_block} matches",
            flush=True,
        )
        collection, audit_env = _collect(
            model,
            matches=args.matches_per_block,
            seed=block_seed,
            values=values,
            profile=profile,
            streams=streams,
            diagnostic_dir=diagnostic_dir,
        )
        current = compute(collection.samples, collection.frames)
        eligible = tuple(map(int, current.indices))
        dense_by_sample = {
            sample_index: dense for dense, sample_index in enumerate(eligible)
        }
        packed_indices, shards = _packed_rows(
            collection, token_budget, packing_waste
        )
        if tuple(sorted(packed_indices)) != tuple(sorted(eligible)):
            raise ValueError("packed policy rows differ from advantage rows")
        selected_q, expected_q = _q_predictions(
            model,
            q_head,
            collection,
            shards,
            dense_by_sample,
            device=device,
            backend=profile.attention,
            use_bf16=use_bf16,
        )
        segments = kyoku_segments(collection.samples, eligible)
        qboost = lambda_one_advantages(
            current.end_values,
            selected_q,
            expected_q,
            segments,
        )
        estimator_values = {
            "current_raw": current.advantages,
            "qboost_raw": qboost,
            "current_standardized": _standardize(current.advantages),
            "qboost_standardized": _standardize(qboost),
        }
        gradients = {}
        for name in names:
            by_sample = {
                sample_index: float(estimator_values[name][dense])
                for dense, sample_index in enumerate(eligible)
            }
            gradients[name] = _policy_gradient(
                model,
                collection,
                shards,
                by_sample,
                matches=args.matches_per_block,
                device=device,
                backend=profile.attention,
                use_bf16=use_bf16,
            )
            moments[name].add(gradients[name])
        for mode in ("raw", "standardized"):
            difference = gradients[f"qboost_{mode}"] - gradients[f"current_{mode}"]
            error_norm_sums[mode] += _dot(difference, difference)

        residual = selected_q.astype(np.float64) - current.end_values
        q_rows.append({
            "rows": len(eligible),
            "selected_q_mse": float(np.mean(residual ** 2)),
            "selected_q_explained_variance": explained_variance(
                selected_q, current.end_values
            ),
            "selected_q_target_correlation": _correlation(
                selected_q, current.end_values
            ),
            "mean_absolute_centered_q": float(
                np.mean(np.abs(selected_q - expected_q))
            ),
        })
        advantage_rows.append({
            "rows": len(eligible),
            "current_variance": float(np.var(current.advantages, dtype=np.float64)),
            "qboost_variance": float(np.var(qboost, dtype=np.float64)),
            "variance_ratio": float(
                np.var(qboost, dtype=np.float64)
                / np.var(current.advantages, dtype=np.float64)
            ),
            "correlation": _correlation(qboost, current.advantages),
        })
        del gradients, collection, audit_env
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        history.append({
            "block": block + 1,
            "matches": (block + 1) * args.matches_per_block,
            "seconds": perf_counter() - block_started,
            "gradients": {
                name: _gradient_report(value, args.matches_per_block)
                for name, value in moments.items()
            },
        })
        partial = {
            "status": "running",
            "protocol": vars(args),
            "history": history,
        }
        output.write_text(json.dumps(partial, indent=2, sort_keys=True), encoding="utf-8")
        raw = history[-1]["gradients"]
        print(
            "block complete: critical batches current/qboost="
            f"{raw['current_raw'].get('critical_batch_matches')}/"
            f"{raw['qboost_raw'].get('critical_batch_matches')}",
            flush=True,
        )

    reports = {
        name: _gradient_report(value, args.matches_per_block)
        for name, value in moments.items()
    }
    paired = {}
    for mode in ("raw", "standardized"):
        difference_sum = (
            moments[f"qboost_{mode}"].sum - moments[f"current_{mode}"].sum
        )
        mean_difference_norm_squared = _dot(
            difference_sum / args.audit_blocks,
            difference_sum / args.audit_blocks,
        )
        sample_variance = max(
            0.0,
            (
                error_norm_sums[mode]
                - _dot(difference_sum, difference_sum) / args.audit_blocks
            ) / (args.audit_blocks - 1),
        )
        paired[mode] = {
            "mean_gradient_difference_norm_squared": mean_difference_norm_squared,
            "debiased_systematic_difference_squared": (
                mean_difference_norm_squared
                - sample_variance / args.audit_blocks
            ),
            "difference_noise_trace_per_match": (
                args.matches_per_block * sample_variance
            ),
        }
    result = {
        "status": "complete",
        "protocol": {
            **vars(args),
            "checkpoint": str(checkpoint_path),
            "checkpoint_id": restored["manifest"]["checkpoint_id"],
            "critic_cross_fit": True,
            "critic_input": "frozen public actor action states",
            "critic_target": "absolute next-kyoku-boundary rank potential",
            "trace_lambda": 1.0,
            "trace_reset": "seat-local end_kyoku",
            "gradient_scope": "all actor parameters",
            "gradient_parameter_count": int(
                moments["current_raw"].sum.size
            ),
            "device": device,
            "precision": profile.precision,
        },
        "critic_training": {
            "matches": args.critic_matches,
            "rows": critic_rows,
            "history": q_training,
        },
        "critic_held_out": {
            key: float(np.average(
                [row[key] for row in q_rows if row[key] is not None],
                weights=[row["rows"] for row in q_rows if row[key] is not None],
            ))
            for key in q_rows[0] if key != "rows"
        } | {"rows": sum(row["rows"] for row in q_rows)},
        "advantages": {
            key: float(np.average(
                [row[key] for row in advantage_rows if row[key] is not None],
                weights=[row["rows"] for row in advantage_rows if row[key] is not None],
            ))
            for key in advantage_rows[0] if key != "rows"
        } | {"rows": sum(row["rows"] for row in advantage_rows)},
        "gradients": reports,
        "paired_gradient_error": paired,
        "history": history,
        "seconds": perf_counter() - started,
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="training/configs/default.toml"
    )
    parser.add_argument(
        "--checkpoint",
        default="runs/behavior-cloning-rank-v/checkpoints",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--critic-matches", type=int, default=128)
    parser.add_argument("--critic-epochs", type=int, default=2)
    parser.add_argument("--critic-learning-rate", type=float, default=3e-4)
    parser.add_argument("--q-width", type=int, default=128)
    parser.add_argument("--audit-blocks", type=int, default=8)
    parser.add_argument("--matches-per-block", type=int, default=64)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
