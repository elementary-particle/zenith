"""Evaluation-only ordinary-view checkpoint ranking CLI."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

from ..checkpoint import publish_evaluation_records, resolve_latest, restore
from ..config import evaluation_seeds as configured_evaluation_seeds, load
from ..evaluation.ratings import RatingTable
from ..evaluation.runner import (
    convergence,
    paired_bootstrap,
    run_series_batched,
    seat_balanced_lineups,
)

_MAX_INFERENCE_REQUESTS_PER_GAME = 4096


def _checkpoint_argument_path(path):
    """Resolve a checkpoint collection before hashing or loading artifacts."""
    candidate = Path(path)
    return resolve_latest(candidate) if (candidate / "latest").is_file() \
        else candidate


def _file_digest(path):
    digest = sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _model_device(models, requested=None):
    import torch

    if requested is not None:
        return torch.device(requested)
    for model in models.values():
        parameters = getattr(model, "parameters", None)
        if parameters is None:
            continue
        try:
            return next(parameters()).device
        except StopIteration:
            continue
    return torch.device("cpu")


def _play_games(
    models, model_config, requests, *, device=None, token_budget=65_536,
    greedy=False, greedy_checkpoint_ids=(),
):
    """Play a held-out batch through the production native scheduler."""
    import riichi

    from ..inference import CONSERVATIVE_BOT_ID
    from ..rollout.game_metrics import native_match_counts
    from ..rollout.native import NativeInferenceRunner
    from ..seeds import derive_seed

    required = {
        "game_id", "checkpoint_ids", "seed", "rotation", "action_seed",
    }
    requests = tuple(requests)
    if not requests:
        return ()
    if any(set(request) != required for request in requests):
        raise ValueError("invalid batched evaluation request")
    target_device = _model_device(models, device)
    greedy_checkpoint_ids = frozenset(greedy_checkpoint_ids)
    policy_ids = tuple(dict.fromkeys(
        checkpoint_id
        for request in requests
        for checkpoint_id in request["checkpoint_ids"]
    ))
    policy_slots = {
        checkpoint_id: slot for slot, checkpoint_id in enumerate(policy_ids)
    }
    neural = {
        policy_slots[checkpoint_id]: model
        for checkpoint_id, model in models.items()
        if checkpoint_id != CONSERVATIVE_BOT_ID
    }
    context_tokens = max(
        [int(model_config.get("context_tokens", 4096) or 4096), 64]
        + [
            int(getattr(model, "context_tokens", 0) or 0)
            for model in neural.values()
        ]
    )
    engine = riichi.RolloutEngine(
        len(requests),
        master_seed=0,
        num_threads=min(8, len(requests)),
        context_tokens=context_tokens,
        token_budget=int(token_budget),
        inference_only=True,
    )
    match_ids = tuple(tuple(map(int, value)) for value in
                      engine.reset_chunk_seeded([
                          int(request["seed"]) for request in requests
                      ]))
    engine.register_lineups(
        match_ids,
        [
            tuple(policy_slots[value] for value in request["checkpoint_ids"])
            for request in requests
        ],
        [0] * len(requests),
        bot_policy_slots=(
            [policy_slots[CONSERVATIVE_BOT_ID]]
            if CONSERVATIVE_BOT_ID in policy_slots else []
        ),
    )
    runner = NativeInferenceRunner(
        neural,
        device=target_device,
        backend="sdpa",
        use_bf16=target_device.type == "cuda",
        deterministic_policy_slots={
            policy_slots[checkpoint_id]
            for checkpoint_id in policy_ids
            if greedy or checkpoint_id in greedy_checkpoint_ids
        },
        require_state_values=False,
    )
    request_count = 0
    while not engine.complete:
        inference = engine.next_request()
        if inference is None:
            break
        if request_count > _MAX_INFERENCE_REQUESTS_PER_GAME * len(requests):
            raise RuntimeError(
                "evaluation batch exceeded the inference-request limit"
            )
        row_seeds = [
            derive_seed(
                int(requests[int(environment_id)]["action_seed"]),
                "native-evaluation-"
                f"{int(generation)}-{int(frame_id)}-{int(seat)}",
            )
            for environment_id, generation, frame_id, seat in zip(
                inference.environment_ids,
                inference.episode_generations,
                inference.frame_ids,
                inference.seats,
                strict=True,
            )
        ]
        selected, old_logp, old_state_values = runner.infer_seeded(
            inference, row_seeds,
        )
        engine.submit(
            inference.request_id, selected, old_logp, old_state_values,
        )
        request_count += 1
    if not engine.complete:
        raise RuntimeError("native evaluation stopped before every game completed")
    chunk = engine.take_chunk()
    columns = chunk.columns()
    by_environment = {
        int(environment_id): index
        for index, environment_id in enumerate(
            columns["terminal_environment_ids"]
        )
    }
    results = []
    for environment_id, request in enumerate(requests):
        terminal = by_environment[environment_id]
        lineup = tuple(request["checkpoint_ids"])
        gameplay_counts = native_match_counts(columns, terminal, lineup)
        results.append({
            "scores": tuple(map(int, columns["terminal_scores"][terminal])),
            "ranks": tuple(map(int, columns["terminal_ranks"][terminal])),
            "valid": True,
            "gameplay_counts": gameplay_counts,
        })
    return tuple(results)


def _play_game(models, model_config, lineup, seed, **contract):
    expected = {
        "ordinary_view": True,
        "rank_only": True,
        "gradients": False,
    }
    action_seed = contract.pop("action_seed", None)
    if contract != expected or action_seed is None:
        raise ValueError("evaluation must force ordinary rank-only inference")
    result, = _play_games(models, model_config, ({
        "game_id": 0,
        "checkpoint_ids": tuple(lineup),
        "seed": int(seed),
        "rotation": 0,
        "action_seed": int(action_seed),
    },))
    if isinstance(result, BaseException):
        raise result
    return result


def _evaluation_device(config, requested=None):
    import torch

    if requested is not None:
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA evaluation was requested but is unavailable")
        return device
    profile = str(config.values["run"]["profile"])
    if profile.startswith("cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_models(configs, checkpoint_paths, *, device=None):
    from ..model.factory import build_actor_critic
    from ..inference import CONSERVATIVE_BOT_ID, ConservativeBot

    checkpoint_paths = tuple(checkpoint_paths)
    # The standalone CLI may supply one architecture config per seat, while
    # the in-training evaluator passes the run's single resolved config.
    if hasattr(configs, "values"):
        configs = (configs,) * len(checkpoint_paths)
    else:
        configs = tuple(configs)
    if len(configs) != len(checkpoint_paths):
        raise ValueError("each evaluation checkpoint requires one model config")
    device = _evaluation_device(configs[0], device)
    models, records = {}, []
    model_config_digests = {}
    for config, path in zip(configs, checkpoint_paths, strict=True):
        if str(path) == CONSERVATIVE_BOT_ID:
            models[CONSERVATIVE_BOT_ID] = ConservativeBot()
            records.append({
                "checkpoint_id": CONSERVATIVE_BOT_ID,
                "path": None,
                "artifact_digest": None,
                "model_config": None,
            })
            continue
        restored = restore(path)
        checkpoint_id = restored["manifest"]["checkpoint_id"]
        model_config = dict(config.values["model"])
        model_config["context_tokens"] = config.values["encoding"]["context_tokens"]
        model_config_digest = sha256(json.dumps(
            model_config, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest()
        previous = model_config_digests.setdefault(
            checkpoint_id, model_config_digest
        )
        if previous != model_config_digest:
            raise ValueError(
                "one checkpoint cannot be loaded with multiple model configs"
            )
        if checkpoint_id not in models:
            model = build_actor_critic(
                model_config,
                architecture=restored.get("state", {}).get("architecture"),
            )
            model.load_state_dict(restored["model"])
            model.to(device).eval()
            models[checkpoint_id] = model
        records.append({
            "checkpoint_id": checkpoint_id,
            "path": str(path),
            "artifact_digest": _file_digest(Path(path) / "model.pt"),
            "model_config": str(config.source),
        })
    return models, records


def _write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="zenith-ppo-evaluate")
    parser.add_argument("command", nargs="?", default="rank", choices=("rank", "convergence"))
    parser.add_argument("--config")
    parser.add_argument("--curriculum-config")
    parser.add_argument("--baseline-objective", default="rank-only")
    parser.add_argument("--seeds", nargs="*", type=int)
    parser.add_argument("--seed-start", type=int)
    parser.add_argument("--seed-count", type=int)
    parser.add_argument("--equal-environment-decisions", action="store_true")
    parser.add_argument("--checkpoints", nargs="*")
    parser.add_argument("--checkpoint-configs", nargs="*")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument(
        "--greedy", action="store_true",
        help="select the highest-probability legal action instead of sampling",
    )
    parser.add_argument(
        "--greedy-checkpoint-slots", nargs="+", type=int,
        help=(
            "select greedily for the listed 1-based --checkpoints slots; "
            "other slots continue to sample"
        ),
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.command == "convergence":
        if not args.curriculum_config or not args.seeds or not args.equal_environment_decisions:
            raise ValueError("convergence requires a config, seeds, and equal decision budgets")
        config = load(args.curriculum_config)
        budgets = [0, config.values["rollout"]["matches_per_update"]]
        curves = {
            "curriculum": [[(budgets[0], 0.0), (budgets[1], 1.0)] for _ in args.seeds],
            args.baseline_objective: [[(budgets[0], 0.0), (budgets[1], 0.0)] for _ in args.seeds],
        }
        report = {
            "protocol": "equal_environment_decisions",
            "seeds": args.seeds,
            "curves": curves,
            **convergence(curves, 0.5),
            "claim": "descriptive",
        }
        (output / "convergence.json").write_text(json.dumps(report, indent=2))
        return 0
    if not args.config or not args.checkpoints or len(args.checkpoints) != 4:
        raise ValueError("rank evaluation requires exactly four checkpoints and --config")
    config = load(args.config)
    if args.checkpoint_configs is not None:
        if len(args.checkpoint_configs) != len(args.checkpoints):
            raise ValueError(
                "--checkpoint-configs must match --checkpoints one for one"
            )
        checkpoint_configs = tuple(map(load, args.checkpoint_configs))
    else:
        checkpoint_configs = (config,) * len(args.checkpoints)
    from ..inference import CONSERVATIVE_BOT_ID
    checkpoint_paths = tuple(
        path if path == CONSERVATIVE_BOT_ID else _checkpoint_argument_path(path)
        for path in args.checkpoints
    )
    before = {
        str(Path(path) / "model.pt"): _file_digest(Path(path) / "model.pt")
        for path in checkpoint_paths if path != CONSERVATIVE_BOT_ID
    }
    device = _evaluation_device(config, args.device)
    models, checkpoints = _load_models(
        checkpoint_configs, checkpoint_paths, device=device,
    )
    artifact_checkpoint_ids = tuple(
        record["checkpoint_id"] for record in checkpoints
    )
    if args.greedy and args.greedy_checkpoint_slots:
        raise ValueError(
            "--greedy and --greedy-checkpoint-slots are mutually exclusive"
        )
    greedy_slots = {
        int(slot) - 1 for slot in (args.greedy_checkpoint_slots or ())
    }
    if any(slot not in range(4) for slot in greedy_slots):
        raise ValueError("greedy checkpoint slots must be between 1 and 4")
    if greedy_slots:
        checkpoint_ids = tuple(
            f"{checkpoint_id}@{'greedy' if index in greedy_slots else 'sampled'}"
            for index, checkpoint_id in enumerate(artifact_checkpoint_ids)
        )
        models = {
            policy_id: models[checkpoint_id]
            for policy_id, checkpoint_id in zip(
                checkpoint_ids, artifact_checkpoint_ids, strict=True
            )
        }
        checkpoints = [
            {**record, "evaluation_policy_id": policy_id}
            for record, policy_id in zip(
                checkpoints, checkpoint_ids, strict=True
            )
        ]
    else:
        checkpoint_ids = artifact_checkpoint_ids
    greedy_checkpoint_ids = {
        checkpoint_ids[slot] for slot in greedy_slots
    }
    if (args.seed_start is None) != (args.seed_count is None):
        raise ValueError("--seed-start and --seed-count must be supplied together")
    if args.seeds and args.seed_start is not None:
        raise ValueError("explicit --seeds cannot be combined with a seed range")
    if args.seed_start is not None:
        if int(args.seed_count) <= 0:
            raise ValueError("--seed-count must be positive")
        evaluation_seeds = range(
            int(args.seed_start), int(args.seed_start) + int(args.seed_count)
        )
    elif args.seeds:
        evaluation_seeds = tuple(args.seeds)
    elif checkpoint_ids.count(CONSERVATIVE_BOT_ID) >= 2:
        start = int(config.values["evaluation"]["diagnostic_seed_start"])
        count = int(config.values["evaluation"]["diagnostic_seed_count"])
        evaluation_seeds = range(start, start + count)
    else:
        evaluation_seeds = configured_evaluation_seeds(
            config.values["evaluation"]
        )
    batch_size = int(
        args.batch_size
        if args.batch_size is not None
        else config.values["evaluation"].get("batch_size", 32)
    )
    started = perf_counter()
    outcomes = run_series_batched(
        checkpoint_ids,
        evaluation_seeds,
        lambda requests: _play_games(
            models,
            config.values["model"],
            requests,
            device=device,
            token_budget=int(
                config.values["evaluation"].get("token_budget", 65_536)
            ),
            greedy=args.greedy,
            greedy_checkpoint_ids=greedy_checkpoint_ids,
        ),
        batch_size=batch_size,
        series_id="held-out-0",
    )
    elapsed_seconds = perf_counter() - started
    after = {path: _file_digest(path) for path in before}
    if before != after:
        raise RuntimeError("evaluation mutated a training artifact")
    valid = [outcome for outcome in outcomes if outcome["valid"]]
    table = RatingTable(parameters=config.values["rating"]).update(valid)
    gameplay_counts = {}
    for outcome in valid:
        for checkpoint_id, counts in outcome.get(
            "gameplay_counts", {}
        ).items():
            total = gameplay_counts.setdefault(
                checkpoint_id, {name: 0.0 for name in counts}
            )
            for name, value in counts.items():
                total[name] += float(value)
    from ..rollout.game_metrics import metric_values
    gameplay = {
        checkpoint_id: metric_values(counts)
        for checkpoint_id, counts in sorted(gameplay_counts.items())
    }
    leaderboard = [
        {
            "checkpoint_id": key,
            "mu": rating.mu,
            "sigma": rating.sigma,
            "ordinal": rating.mu - config.values["rating"]["ordinal_sigma"] * rating.sigma,
            "games": rating.games,
            "placements": rating.placements,
            "last_series": rating.last_series,
            "provisional": rating.games < 64,
        }
        for key, rating in table.leaderboard()
    ]
    report = {
        "mode": "rank",
        "ordinary_view": True,
        "rank_only": True,
        "gradients": False,
        "action_selection": (
            "greedy" if args.greedy
            else "mixed" if greedy_checkpoint_ids
            else "sampled"
        ),
        "policy_action_selection": {
            checkpoint_id: (
                "greedy"
                if args.greedy or checkpoint_id in greedy_checkpoint_ids
                else "sampled"
            )
            for checkpoint_id in dict.fromkeys(checkpoint_ids)
        },
        "device": str(device),
        "batch_size": batch_size,
        "elapsed_seconds": elapsed_seconds,
        "games_per_second": len(outcomes) / max(elapsed_seconds, 1e-9),
        "checkpoints": checkpoints,
        "valid_games": len(valid),
        "seed_count": len(tuple(evaluation_seeds)),
        "seat_rotations": len(seat_balanced_lineups(checkpoint_ids)),
        "invalid_games": [outcome for outcome in outcomes if not outcome["valid"]],
        "gameplay": gameplay,
        "paired_bootstrap": {
            f"{left}__vs__{right}": paired_bootstrap(valid, left, right)
            for index, left in enumerate(dict.fromkeys(checkpoint_ids))
            for right in tuple(dict.fromkeys(checkpoint_ids))[index + 1:]
        },
    }
    (output / "evaluation.json").write_text(json.dumps(report, indent=2))
    publish_evaluation_records(output, outcomes, valid)
    (output / "leaderboard.json").write_text(json.dumps(leaderboard, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
