"""Evaluation-only ordinary-view checkpoint ranking CLI."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

from ..checkpoint import publish_evaluation_records, restore
from ..config import load
from ..evaluation.ratings import RatingTable
from ..evaluation.runner import (
    convergence,
    paired_bootstrap,
    run_series_batched,
    seat_balanced_lineups,
)


def _file_digest(path):
    digest = sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(slots=True)
class _EvaluationGame:
    index: int
    lineup: tuple[str, str, str, str]
    env: object
    adapter: object
    batch: object
    event_cache: object
    action_generator: object
    game_metrics: dict[str, object]
    frames: int = 0
    final: dict | None = None


def _terminal_result(batch):
    final = None
    for event in batch.transition.events:
        if int(event.kind) != 16:
            continue
        payload = bytes(event.payload or b"")
        if len(payload) != 20:
            raise ValueError("invalid end_game payload")
        scores = tuple(
            int.from_bytes(payload[index:index + 4], "little", signed=True)
            for index in range(0, 16, 4)
        )
        ranks = tuple(int(rank) - 1 for rank in payload[16:20])
        final = {"scores": scores, "ranks": ranks, "valid": True}
    return final


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
    max_padding_fraction=0.10, max_frames=4096, greedy=False,
    greedy_checkpoint_ids=(),
):
    """Play independent seeded hanchan with shared batched neural inference."""
    import riichi
    import torch

    from ..encoding.actions import segmented_sample
    from ..encoding.packing import encode_native_batch, model_batch, pack
    from ..encoding.event_cache import EventPrefixCache
    from ..env.adapter import EnvAdapter
    from ..inference import CONSERVATIVE_BOT_ID
    from ..rollout.collector import _GameMetricAccumulator

    required = {
        "game_id", "checkpoint_ids", "seed", "rotation", "action_seed",
    }
    requests = tuple(requests)
    if any(set(request) != required for request in requests):
        raise ValueError("invalid batched evaluation request")
    greedy_checkpoint_ids = frozenset(greedy_checkpoint_ids)
    target_device = _model_device(models, device)
    contexts = {}
    results = [None] * len(requests)
    try:
        for index, request in enumerate(requests):
            env = riichi.Env(
                1, master_seed=int(request["seed"]), num_threads=1,
                privileged=True,
            )
            adapter = EnvAdapter(env)
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(request["action_seed"]))
            batch = adapter.reset([0])
            lineup = tuple(request["checkpoint_ids"])
            game_metrics = {
                checkpoint_id: _GameMetricAccumulator(
                    lambda _key, mask=sum(
                        int(identity == checkpoint_id) << seat
                        for seat, identity in enumerate(lineup)
                    ): mask
                )
                for checkpoint_id in dict.fromkeys(lineup)
            }
            for accumulator in game_metrics.values():
                accumulator.observe(batch.transition.events)
            contexts[index] = _EvaluationGame(
                index=index,
                lineup=lineup,
                env=env,
                adapter=adapter,
                batch=batch,
                event_cache=EventPrefixCache(max_entries=16),
                final=_terminal_result(batch),
                action_generator=generator,
                game_metrics=game_metrics,
            )

        while contexts:
            ready = []
            for index, context in tuple(contexts.items()):
                try:
                    while True:
                        state = context.batch.transition.states[0]
                        if int(state.lifecycle) == 3:
                            if context.final is None:
                                raise RuntimeError(
                                    "terminal evaluation game lacks end_game event"
                                )
                            results[index] = {
                                **context.final,
                                "gameplay_counts": {
                                    checkpoint_id: dict(accumulator.counts)
                                    for checkpoint_id, accumulator
                                    in context.game_metrics.items()
                                },
                            }
                            context.env.close()
                            del contexts[index]
                            break
                        if state.action_spaces:
                            ready.append((context, state))
                            break
                        if context.frames >= int(max_frames):
                            raise RuntimeError(
                                "evaluation game exceeded the frame limit"
                            )
                        context.batch = context.adapter.advance(
                            [int(state.environment_id)]
                        )
                        for accumulator in context.game_metrics.values():
                            accumulator.observe(context.batch.transition.events)
                        context.frames += 1
                        context.final = (
                            _terminal_result(context.batch) or context.final
                        )
                except Exception as exc:
                    results[index] = exc
                    context.env.close()
                    del contexts[index]
            if not ready:
                continue

            entries = []
            actions = {}
            states = {}
            for context, state in ready:
                try:
                    encoded = encode_native_batch(
                        context.batch,
                        context.adapter.histories,
                        event_cache=context.event_cache,
                    )
                    if not encoded:
                        raise RuntimeError(
                            "decision frame produced no encoded policy rows"
                        )
                    actions[context.index] = [None] * len(encoded)
                    states[context.index] = state
                    for local, row in enumerate(encoded):
                        entries.append((context, local, row))
                except Exception as exc:
                    results[context.index] = exc
                    context.env.close()
                    del contexts[context.index]

            neural_groups = {}
            row_log_probabilities = {}
            for entry_index, (context, local, row) in enumerate(entries):
                if context.index not in contexts:
                    continue
                checkpoint_id = context.lineup[row.binding.seat]
                if checkpoint_id == CONSERVATIVE_BOT_ID:
                    policy = models[checkpoint_id]
                    group = policy.select_group(
                        row, state=states[context.index]
                    )
                    representative = row.action_representatives[group]
                    actions[context.index][local] = row.native_candidates[
                        representative
                    ]
                else:
                    neural_groups.setdefault(checkpoint_id, []).append(
                        entry_index
                    )

            with torch.inference_mode():
                for checkpoint_id, indices in neural_groups.items():
                    model = models[checkpoint_id]
                    lengths = [
                        len(entries[index][2].token_factors)
                        for index in indices
                    ]
                    shards = pack(
                        lengths, int(token_budget),
                        max_padding_fraction=float(max_padding_fraction),
                    ).batches
                    with torch.autocast(
                        device_type=target_device.type,
                        dtype=torch.bfloat16,
                        enabled=target_device.type == "cuda",
                    ):
                      for shard in shards:
                        shard_indices = [indices[local] for local in shard]
                        inputs = model_batch(
                            [entries[index][2] for index in shard_indices],
                            device=target_device,
                        )
                        output = model.forward_actor(
                            **inputs, policy_only=True
                        )
                        offsets = inputs["action_offsets"].cpu().tolist()
                        host_logp = output.log_probabilities.detach().float().cpu()
                        for local, entry_index in enumerate(shard_indices):
                            start = offsets[local]
                            end = offsets[local + 1]
                            row_log_probabilities[entry_index] = (
                                host_logp[start:end]
                            )

            by_game_policy = {}
            for entry_index, (context, _, row) in enumerate(entries):
                if entry_index not in row_log_probabilities:
                    continue
                checkpoint_id = context.lineup[row.binding.seat]
                by_game_policy.setdefault(
                    (context.index, checkpoint_id), []
                ).append(entry_index)
            for (game_index, checkpoint_id), indices in by_game_policy.items():
                context = contexts[game_index]
                lengths = [
                    int(row_log_probabilities[index].numel())
                    for index in indices
                ]
                offsets = torch.tensor(
                    [0, *torch.tensor(lengths).cumsum(0).tolist()],
                    dtype=torch.long,
                )
                sampled = segmented_sample(
                    torch.cat([
                        row_log_probabilities[index] for index in indices
                    ]),
                    offsets,
                    generator=context.action_generator,
                    deterministic=(
                        bool(greedy)
                        or checkpoint_id in greedy_checkpoint_ids
                    ),
                )
                for local, entry_index in enumerate(indices):
                    context, candidate_index, row = entries[entry_index]
                    group = int(sampled[local]) - int(offsets[local])
                    if not 0 <= group < lengths[local]:
                        raise RuntimeError(
                            "sampled evaluation action is out of range"
                        )
                    representative = row.action_representatives[group]
                    actions[game_index][candidate_index] = row.native_candidates[
                        representative
                    ]

            for context, _ in ready:
                if context.index not in contexts:
                    continue
                try:
                    native_candidates = actions[context.index]
                    if any(action is None for action in native_candidates):
                        raise RuntimeError(
                            "evaluation did not select every required action"
                        )
                    context.batch = context.adapter.step(
                        [candidate.select() for candidate in native_candidates]
                    )
                    for accumulator in context.game_metrics.values():
                        accumulator.observe(context.batch.transition.events)
                    context.frames += 1
                    context.final = (
                        _terminal_result(context.batch) or context.final
                    )
                except Exception as exc:
                    results[context.index] = exc
                    context.env.close()
                    del contexts[context.index]
        return tuple(results)
    finally:
        for context in contexts.values():
            context.env.close()


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
    from ..model.actor_critic import ActorCritic
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
            model = ActorCritic(model_config)
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
    before = {
        str(Path(path) / "model.pt"): _file_digest(Path(path) / "model.pt")
        for path in args.checkpoints if path != CONSERVATIVE_BOT_ID
    }
    device = _evaluation_device(config, args.device)
    models, checkpoints = _load_models(
        checkpoint_configs, args.checkpoints, device=device,
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
        evaluation_seeds = config.values["evaluation"]["held_out_seeds"]
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
            max_padding_fraction=float(
                config.values["encoding"].get(
                    "inference_packing_max_waste", 0.5
                )
            ),
            max_frames=int(config.values["rollout"]["max_frames_per_match"]),
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
    from ..rollout.collector import _game_metric_values
    gameplay = {
        checkpoint_id: _game_metric_values(counts)
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
