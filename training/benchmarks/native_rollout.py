"""Repeatable 256-match BF16/SDPA native-rollout benchmark.

Run from the repository root after installing the local ``riichi`` extension:

    PYTHONPATH=training/src python training/benchmarks/native_rollout.py \
        --device cuda --output native-rollout-benchmark.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
from time import perf_counter

import torch

import riichi
from zenith_ppo.model.actor_critic import ActorCritic
from zenith_ppo.rollout.native import NativeInferenceRunner


BASELINE = {
    "revision": "ee700c0",
    "decisions_per_second": 3007.0,
    "rollout_seconds_per_256_matches": 53.73,
}


def _cuda_utilization(device):
    if device.type != "cuda" or not hasattr(torch.cuda, "utilization"):
        return None
    try:
        return int(torch.cuda.utilization(device))
    except (ImportError, ModuleNotFoundError, RuntimeError):
        return None


def _model(context_tokens):
    return ActorCritic({
        "layers": 3,
        "d_model": 192,
        "query_heads": 4,
        "kv_heads": 1,
        "head_dim": 48,
        "ffn_dim": 384,
        "context_tokens": context_tokens,
        "action_memory_layers": 4,
        "action_memory_ffn_dim": 384,
        "share_all_action_tiles": True,
        "concealed_shape_channels": 24,
        "concealed_shape_blocks": 2,
        "rank_critic_width": 64,
    })


def run(args):
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA benchmark requested but CUDA is unavailable")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()
    model = _model(args.context_tokens).to(device).eval()
    engine = riichi.RolloutEngine(
        args.matches,
        master_seed=args.seed,
        num_threads=args.threads,
        context_tokens=args.context_tokens,
        token_budget=args.token_budget,
    )
    matches = engine.reset_chunk(args.matches)
    engine.register_lineups(
        matches,
        [(0, 0, 0, 0)] * args.matches,
        [15] * args.matches,
    )
    runner = NativeInferenceRunner(
        {0: model},
        device=device,
        backend="sdpa",
        use_bf16=True,
        generator=torch.Generator(device=device).manual_seed(args.seed + 1),
        compile_cuda=not args.eager,
    )
    started = perf_counter()
    chunk = runner.run_chunk(engine)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = perf_counter() - started
    ppo = None
    if args.ppo_update:
        from zenith_ppo.config import load
        from zenith_ppo.ppo.trainer import PPOTrainer

        ppo_started = perf_counter()
        runner.prepare_training_chunk(chunk)
        actor_batches = list(chunk.actor_minibatches(
            0,
            args.token_budget,
            max_padding_fraction=0.10,
            backend="sdpa",
        ))
        critic_batches = [chunk.critic_batch(policy_slot=0)]
        trainer = PPOTrainer(
            model,
            load("training/configs/default.toml").values["ppo"],
            device_type=device.type,
            use_bf16=device.type == "cuda",
        )
        if args.ppo_mode == "logical":
            update = trainer.update_logical_batch(
                actor_batches,
                critic_minibatches=critic_batches,
                ema_matches=args.matches,
            )
        else:
            trainer.begin_streaming_update(ema_matches=args.matches)
            trainer.accumulate_streaming_chunk(
                actor_batches, critic_minibatches=critic_batches,
            )
            update = trainer.finish_streaming_update()
        if not update.committed:
            raise RuntimeError(f"benchmark PPO update failed: {update.reason}")
        if device.type == "cuda":
            torch.cuda.synchronize()
        ppo_seconds = perf_counter() - ppo_started
        complete_seconds = elapsed + ppo_seconds
        ppo = {
            "optimization_seconds": ppo_seconds,
            "complete_update_seconds": complete_seconds,
            "updates_per_second": 1.0 / complete_seconds,
            "matches_per_hour": args.matches * 3600.0 / complete_seconds,
            "actor_minibatches": len(actor_batches),
            "critic_minibatches": len(critic_batches),
            "batching": args.ppo_mode,
        }
    stats = runner.stats(engine)
    native = engine.metrics()
    result = {
        "schema": "zenith-native-rollout-benchmark-v1",
        "baseline": BASELINE,
        "configuration": {
            "matches": args.matches,
            "environments": args.matches,
            "threads": args.threads,
            "device": str(device),
            "precision": "bf16" if device.type == "cuda" else "fp32",
            "attention": "sdpa",
            "token_budget": args.token_budget,
            "context_tokens": args.context_tokens,
            "compiled": bool(device.type == "cuda" and not args.eager),
            "seed": args.seed,
            "model": "ee700c0-default-3x192",
        },
        "rollout": {
            "elapsed_seconds": elapsed,
            "decisions": chunk.row_count,
            "decisions_per_second": chunk.row_count / elapsed,
            "matches": chunk.match_completions,
            "matches_per_hour": chunk.match_completions * 3600.0 / elapsed,
            "inference_launches": stats.requests,
            "inference_rows": stats.rows,
            "inference_rows_per_launch": stats.rows / max(1, stats.requests),
            "useful_tokens": stats.useful_tokens,
            "padded_tokens": stats.padded_tokens,
            "padding_fraction": (
                1.0 - stats.useful_tokens / stats.padded_tokens
                if stats.padded_tokens else 0.0
            ),
            "useful_actions": stats.useful_actions,
            "padded_actions": stats.padded_actions,
            "action_padding_fraction": (
                1.0 - stats.useful_actions / stats.padded_actions
                if stats.padded_actions else 0.0
            ),
            "env_calls": int(native["env_calls"]),
            "compile_fallbacks": stats.compile_fallbacks,
            "inference_seconds": stats.inference_seconds,
            "native_seconds": stats.native_seconds,
            "inference_wall_fraction": stats.inference_seconds / elapsed,
            "automatic_rows": stats.automatic_rows,
        },
        "memory": {
            "peak_host_bytes": int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            ),
            "peak_cuda_allocated_bytes": (
                int(torch.cuda.max_memory_allocated())
                if device.type == "cuda" else 0
            ),
            "peak_cuda_reserved_bytes": (
                int(torch.cuda.max_memory_reserved())
                if device.type == "cuda" else 0
            ),
        },
        "utilization": {
            "gpu_percent_at_end": (
                _cuda_utilization(device)
            ),
            "cpu_time_seconds": resource.getrusage(
                resource.RUSAGE_SELF
            ).ru_utime,
        },
        "ppo": ppo,
    }
    result["comparison"] = {
        "rollout_decisions_per_second_speedup": (
            result["rollout"]["decisions_per_second"]
            / BASELINE["decisions_per_second"]
        ),
        "meets_2x_rollout_gate": (
            result["rollout"]["decisions_per_second"]
            >= 2 * BASELINE["decisions_per_second"]
        ),
        "rollout_wall_time_speedup": (
            BASELINE["rollout_seconds_per_256_matches"] / elapsed
            if args.matches == 256 else None
        ),
        "meets_2x_wall_time_gate": (
            elapsed <= BASELINE["rollout_seconds_per_256_matches"] / 2
            if args.matches == 256 else None
        ),
    }
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--matches", type=int, default=256)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--seed", type=int, default=700)
    parser.add_argument("--context-tokens", type=int, default=2048)
    parser.add_argument("--token-budget", type=int, default=65536)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--ppo-update", action="store_true")
    parser.add_argument(
        "--ppo-mode", choices=("logical", "streaming"), default="logical",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.matches <= 0 or args.threads <= 0:
        parser.error("matches and threads must be positive")
    result = run(args)
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
