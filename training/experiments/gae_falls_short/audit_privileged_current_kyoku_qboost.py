"""Lambda-one current-kyoku audit with an independent privileged Q critic."""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np

from audit_current_kyoku_qboost import (
    GradientMoments,
    _actor_inputs,
    _checkpoint_path,
    _collect,
    _correlation,
    _dot,
    _gradient_report,
    _packed_rows,
    _policy_gradient,
    _standardize,
)
from current_kyoku_qboost import kyoku_segments, lambda_one_advantages
from privileged_critic import (
    PrivilegedActionQ,
    encode_privileged_snapshot,
    privileged_batch,
)


def _collect_privileged(*args, **kwargs):
    """Capture hidden state beside the unmodified production collector."""
    import zenith_ppo.rollout.collector as collector_module

    original = collector_module.encode_native_batch
    snapshots = {}

    def capturing(batch, histories, **options):
        rows = original(batch, histories, **options)
        states = {
            (int(state.environment_id), int(state.episode_generation)): state
            for state in batch.transition.states
        }
        encoded_by_state = {}
        for row in rows:
            key = (
                int(row.binding.environment_id),
                int(row.binding.episode_generation),
            )
            if key not in encoded_by_state:
                encoded_by_state[key] = encode_privileged_snapshot(states[key])
            snapshots[row.binding] = encoded_by_state[key]
        return rows

    collector_module.encode_native_batch = capturing
    try:
        collection, env = _collect(*args, **kwargs)
    finally:
        collector_module.encode_native_batch = original
    required = {
        sample.binding for sample in collection.samples if sample.ppo_eligible
    }
    missing = required - snapshots.keys()
    if missing:
        raise ValueError(f"privileged capture missed {len(missing)} policy rows")
    return collection, env, {
        binding: snapshots[binding] for binding in required
    }


def _train_chunk(
    critic,
    optimizer,
    collection,
    snapshots,
    *,
    actor,
    actor_hidden_states,
    epochs,
    batch_rows,
    device,
    backend,
    token_budget,
    packing_waste,
    use_bf16,
    rng,
):
    import torch

    from zenith_ppo.ppo.current_kyoku import compute

    current = compute(collection.samples, collection.frames)
    targets = {
        int(index): float(value)
        for index, value in zip(current.indices, current.end_values, strict=True)
    }
    indices = np.asarray(tuple(targets), dtype=np.int64)
    packed_shards = None
    if actor_hidden_states:
        packed_indices, packed = _packed_rows(
            collection, token_budget, packing_waste
        )
        if set(packed_indices) != set(targets):
            raise ValueError("actor-hidden Q training rows differ from targets")
        packed_shards = [
            tuple(shard[start:start + batch_rows])
            for shard in packed
            for start in range(0, len(shard), batch_rows)
        ]
        actor.requires_grad_(False).eval()
    statistics = []
    critic.train()
    for epoch in range(epochs):
        if packed_shards is None:
            rng.shuffle(indices)
            shards = tuple(
                tuple(map(int, indices[start:start + batch_rows]))
                for start in range(0, len(indices), batch_rows)
            )
        else:
            rng.shuffle(packed_shards)
            shards = tuple(packed_shards)
        squared_error = 0.0
        rows = 0
        for shard in shards:
            samples = [collection.samples[index] for index in shard]
            inputs = privileged_batch(samples, snapshots, device=device)
            if actor_hidden_states:
                actor_inputs = _actor_inputs(
                    collection, shard, device, backend
                )
                with torch.no_grad(), torch.autocast(
                    device_type=device,
                    dtype=torch.bfloat16,
                    enabled=use_bf16,
                ):
                    actor_output = actor.forward_actor(**actor_inputs)
                inputs["actor_states"] = actor_output.actor_states.detach()
                inputs["actor_action_states"] = (
                    actor_output.action_states.detach()
                )
            with torch.autocast(
                device_type=device,
                dtype=torch.bfloat16,
                enabled=use_bf16,
            ):
                q_values = critic(**inputs)
                local = torch.arange(len(shard), device=device)
                groups = torch.as_tensor(
                    [sample.selected_group for sample in samples],
                    dtype=torch.long,
                    device=device,
                )
                selected_q = q_values[local, groups]
                target = torch.as_tensor(
                    [targets[index] for index in shard],
                    dtype=torch.float32,
                    device=device,
                )
                loss = torch.nn.functional.mse_loss(selected_q, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 2.0)
            optimizer.step()
            squared_error += float(
                torch.sum((selected_q.detach() - target) ** 2).cpu()
            )
            rows += len(shard)
        statistics.append({
            "epoch": epoch + 1,
            "rows": rows,
            "mse": squared_error / rows,
        })
    return statistics


def _q_predictions(
    actor,
    critic,
    collection,
    snapshots,
    shards,
    dense_by_sample,
    *,
    device,
    backend,
    use_bf16,
    actor_hidden_states,
):
    import torch

    selected = np.empty(len(dense_by_sample), dtype=np.float32)
    expected = np.empty(len(dense_by_sample), dtype=np.float32)
    actor.requires_grad_(False).eval()
    critic.eval()
    with torch.no_grad():
        for shard in shards:
            actor_inputs = _actor_inputs(collection, shard, device, backend)
            q_inputs = privileged_batch(
                [collection.samples[index] for index in shard],
                snapshots,
                device=device,
            )
            with torch.autocast(
                device_type=device,
                dtype=torch.bfloat16,
                enabled=use_bf16,
            ):
                policy = actor.forward_actor(**actor_inputs)
                if actor_hidden_states:
                    q_inputs["actor_states"] = policy.actor_states.detach()
                    q_inputs["actor_action_states"] = (
                        policy.action_states.detach()
                    )
                q_values = critic(**q_inputs)
            offsets = actor_inputs["action_offsets"].long()
            for local, sample_index in enumerate(shard):
                dense = dense_by_sample[sample_index]
                length = int(actor_inputs["action_lengths"][local])
                start, end = int(offsets[local]), int(offsets[local + 1])
                probabilities = policy.log_probabilities[start:end].float().exp()
                legal_q = q_values[local, :length].float()
                group = int(collection.samples[sample_index].selected_group)
                selected[dense] = float(legal_q[group].cpu())
                expected[dense] = float((probabilities * legal_q).sum().cpu())
    return selected, expected


def _weighted(rows, key):
    usable = [row for row in rows if row[key] is not None]
    return float(np.average(
        [row[key] for row in usable],
        weights=[row["rows"] for row in usable],
    ))


def run(args) -> dict:
    import torch

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
    actor = ActorCritic(model_config).to(device)
    actor.load_state_dict(restored["model"])
    actor.requires_grad_(False).eval()
    torch.manual_seed(args.seed)
    critic = PrivilegedActionQ(
        width=args.q_width,
        heads=args.q_heads,
        layers=args.q_layers,
        actor_hidden_width=(
            int(model_config["d_model"]) if args.actor_hidden_states else None
        ),
    ).to(device)
    optimizer = torch.optim.AdamW(
        critic.parameters(),
        lr=args.critic_learning_rate,
        weight_decay=args.critic_weight_decay,
    )
    streams = SeedStreams(args.seed)
    rng = np.random.default_rng(args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    diagnostic_dir = output.parent / "native-env-diagnostics"
    packing_waste = float(values["encoding"]["packing_max_waste"])
    token_budget = int(values["ppo"]["token_budget"])

    if args.critic_matches % args.critic_chunk_matches:
        raise ValueError("critic matches must divide evenly into critic chunks")
    critic_history = []
    trained_rows = 0
    chunks = args.critic_matches // args.critic_chunk_matches
    for chunk in range(chunks):
        print(
            f"privileged critic chunk {chunk + 1}/{chunks}: "
            f"collecting {args.critic_chunk_matches} matches",
            flush=True,
        )
        collection, env, snapshots = _collect_privileged(
            actor,
            matches=args.critic_chunk_matches,
            seed=derive_seed(args.seed + chunk, "visibility"),
            values=values,
            profile=profile,
            streams=streams,
            diagnostic_dir=diagnostic_dir,
        )
        rows = sum(sample.ppo_eligible for sample in collection.samples)
        chunk_statistics = _train_chunk(
            critic,
            optimizer,
            collection,
            snapshots,
            actor=actor,
            actor_hidden_states=args.actor_hidden_states,
            epochs=args.critic_epochs,
            batch_rows=args.critic_batch_rows,
            device=device,
            backend=profile.attention,
            token_budget=token_budget,
            packing_waste=packing_waste,
            use_bf16=use_bf16,
            rng=rng,
        )
        trained_rows += rows * args.critic_epochs
        critic_history.append({
            "chunk": chunk + 1,
            "matches": (chunk + 1) * args.critic_chunk_matches,
            "rows": rows,
            "epochs": chunk_statistics,
        })
        print(
            f"critic chunk complete: mse={chunk_statistics[-1]['mse']:.6f}",
            flush=True,
        )
        del collection, env, snapshots
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    names = (
        "current_raw", "qboost_raw",
        "current_standardized", "qboost_standardized",
    )
    moments = {name: GradientMoments() for name in names}
    error_norm_sums = {"raw": 0.0, "standardized": 0.0}
    q_rows = []
    advantage_rows = []
    history = []
    for block in range(args.audit_blocks):
        block_started = perf_counter()
        print(
            f"privileged audit block {block + 1}/{args.audit_blocks}: "
            f"collecting {args.matches_per_block} matches",
            flush=True,
        )
        collection, env, snapshots = _collect_privileged(
            actor,
            matches=args.matches_per_block,
            seed=derive_seed(args.seed + 10000 + block, "visibility"),
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
            actor,
            critic,
            collection,
            snapshots,
            shards,
            dense_by_sample,
            device=device,
            backend=profile.attention,
            use_bf16=use_bf16,
            actor_hidden_states=args.actor_hidden_states,
        )
        qboost = lambda_one_advantages(
            current.end_values,
            selected_q,
            expected_q,
            kyoku_segments(collection.samples, eligible),
        )
        estimator_values = {
            "current_raw": current.advantages,
            "qboost_raw": qboost,
            "current_standardized": _standardize(current.advantages),
            "qboost_standardized": _standardize(qboost),
        }
        gradients = {}
        for name in names:
            gradients[name] = _policy_gradient(
                actor,
                collection,
                shards,
                {
                    sample_index: float(estimator_values[name][dense])
                    for dense, sample_index in enumerate(eligible)
                },
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
        current_variance = float(np.var(current.advantages, dtype=np.float64))
        qboost_variance = float(np.var(qboost, dtype=np.float64))
        advantage_rows.append({
            "rows": len(eligible),
            "current_variance": current_variance,
            "qboost_variance": qboost_variance,
            "variance_ratio": qboost_variance / current_variance,
            "correlation": _correlation(qboost, current.advantages),
        })
        del gradients, collection, env, snapshots
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
        output.write_text(json.dumps({
            "status": "running",
            "protocol": vars(args),
            "critic_training": critic_history,
            "history": history,
        }, indent=2, sort_keys=True), encoding="utf-8")
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
        difference_mean = difference_sum / args.audit_blocks
        mean_norm_squared = _dot(difference_mean, difference_mean)
        sample_variance = max(0.0, (
            error_norm_sums[mode]
            - _dot(difference_sum, difference_sum) / args.audit_blocks
        ) / (args.audit_blocks - 1))
        noise = args.matches_per_block * sample_variance
        signal = mean_norm_squared - sample_variance / args.audit_blocks
        paired[mode] = {
            "mean_gradient_difference_norm_squared": mean_norm_squared,
            "debiased_systematic_difference_squared": signal,
            "difference_noise_trace_per_match": noise,
            "critical_batch_matches": noise / signal if signal > 0 else None,
            "snr_at_observed_batch": (
                math.sqrt(
                    args.audit_blocks * args.matches_per_block * signal / noise
                ) if signal > 0 and noise > 0 else None
            ),
        }
    result = {
        "status": "complete",
        "protocol": {
            **vars(args),
            "checkpoint": str(checkpoint_path),
            "checkpoint_id": restored["manifest"]["checkpoint_id"],
            "critic_cross_fit": True,
            "critic_input": (
                "independent dealer-relative all-hands, exact-wall, public-state, "
                "and legal-action encoder"
                + (
                    ", plus detached actor state and per-action hidden states"
                    if args.actor_hidden_states else ""
                )
            ),
            "actor_hidden_states": args.actor_hidden_states,
            "actor_hidden_gradient_path": False,
            "critic_target": "absolute next-kyoku-boundary rank potential",
            "trace_lambda": 1.0,
            "trace_reset": "seat-local end_kyoku",
            "gradient_scope": "all actor parameters",
            "gradient_parameter_count": int(moments["current_raw"].sum.size),
            "critic_parameter_count": sum(
                parameter.numel() for parameter in critic.parameters()
            ),
            "device": device,
            "precision": profile.precision,
        },
        "critic_training": {
            "matches": args.critic_matches,
            "optimizer_rows": trained_rows,
            "history": critic_history,
        },
        "critic_held_out": {
            key: _weighted(q_rows, key) for key in q_rows[0] if key != "rows"
        } | {"rows": sum(row["rows"] for row in q_rows)},
        "advantages": {
            key: _weighted(advantage_rows, key)
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
    parser.add_argument("--config", default="training/configs/default.toml")
    parser.add_argument(
        "--checkpoint", default="runs/behavior-cloning-rank-v/checkpoints"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--critic-matches", type=int, default=512)
    parser.add_argument("--critic-chunk-matches", type=int, default=64)
    parser.add_argument("--critic-epochs", type=int, default=2)
    parser.add_argument("--critic-batch-rows", type=int, default=256)
    parser.add_argument("--critic-learning-rate", type=float, default=3e-4)
    parser.add_argument("--critic-weight-decay", type=float, default=1e-4)
    parser.add_argument("--q-width", type=int, default=96)
    parser.add_argument("--q-heads", type=int, default=4)
    parser.add_argument("--q-layers", type=int, default=2)
    parser.add_argument("--actor-hidden-states", action="store_true")
    parser.add_argument("--audit-blocks", type=int, default=8)
    parser.add_argument("--matches-per-block", type=int, default=64)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
