"""Evaluation-only ordinary-view checkpoint ranking CLI."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from ..checkpoint import publish_evaluation_records, restore
from ..config import load
from ..evaluation.ratings import RatingTable
from ..evaluation.runner import convergence, run_series


def _file_digest(path):
    digest = sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _play_game(models, model_config, lineup, seed, **contract):
    import riichi
    import torch

    from ..encoding.packing import encode_native_batch, model_batch
    from ..encoding.event_cache import EventPrefixCache
    from ..env.adapter import EnvAdapter

    if contract != {"ordinary_view": True, "rank_only": True, "gradients": False}:
        raise ValueError("evaluation must force ordinary rank-only inference")
    env = riichi.Env(1, master_seed=int(seed), num_threads=1, privileged=True)
    adapter = EnvAdapter(env)
    batch = adapter.reset([0])
    event_cache = EventPrefixCache(max_entries=16)
    final = None
    try:
        for _ in range(4096):
            state = batch.transition.states[0]
            if int(state.lifecycle) == 3:
                break
            encoded = encode_native_batch(
                batch,
                adapter.histories,
                critic_mode="ordinary",
                event_cache=event_cache,
            )
            actions = [None] * len(encoded)
            groups = {}
            for index, row in enumerate(encoded):
                groups.setdefault(lineup[row.binding.seat], []).append(index)
            with torch.no_grad():
                for checkpoint_id, indices in groups.items():
                    model = models[checkpoint_id]
                    inputs = model_batch([encoded[index] for index in indices])
                    output = model(**inputs)
                    for local, index in enumerate(indices):
                        start = int(inputs["action_offsets"][local])
                        end = int(inputs["action_offsets"][local + 1])
                        group = int(output.log_probabilities[start:end].argmax())
                        representative = encoded[index].action_representatives[group]
                        actions[index] = encoded[index].native_actions[representative]
            batch = adapter.step(actions)
            for event in batch.transition.events:
                if int(event.kind) == 16:
                    payload = bytes(event.payload or b"")
                    if len(payload) != 20:
                        raise ValueError("invalid end_game payload")
                    scores = tuple(
                        int.from_bytes(payload[index:index + 4], "little", signed=True)
                        for index in range(0, 16, 4)
                    )
                    ranks = tuple(int(rank) - 1 for rank in payload[16:20])
                    final = {"scores": scores, "ranks": ranks, "valid": True}
        if final is None:
            raise RuntimeError("evaluation game exceeded the frame limit")
        return final
    finally:
        env.close()


def _load_models(config, checkpoint_paths):
    import torch

    from ..model.actor_critic import ActorCritic

    model_config = dict(config.values["model"])
    model_config["context_tokens"] = config.values["encoding"]["context_tokens"]
    models, records = {}, []
    for path in checkpoint_paths:
        restored = restore(path, expected=config.compatibility)
        checkpoint_id = restored["manifest"]["checkpoint_id"]
        model = ActorCritic(model_config)
        model.load_state_dict(restored["model"])
        model.eval()
        models[checkpoint_id] = model
        records.append({
            "checkpoint_id": checkpoint_id,
            "path": str(path),
            "model_schema": config.values["model"]["schema"],
            "artifact_digest": _file_digest(Path(path) / "model.pt"),
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
    parser.add_argument("--equal-environment-decisions", action="store_true")
    parser.add_argument("--checkpoints", nargs="*")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.command == "convergence":
        if not args.curriculum_config or not args.seeds or not args.equal_environment_decisions:
            raise ValueError("convergence requires a config, seeds, and equal decision budgets")
        config = load(args.curriculum_config)
        budgets = [0, config.values["rollout"]["learner_decisions_per_update"]]
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
    before = {
        str(Path(path) / "model.pt"): _file_digest(Path(path) / "model.pt")
        for path in args.checkpoints
    }
    models, checkpoints = _load_models(config, args.checkpoints)
    checkpoint_ids = tuple(record["checkpoint_id"] for record in checkpoints)
    outcomes = run_series(
        checkpoint_ids,
        config.values["evaluation"]["held_out_seeds"],
        lambda lineup, seed, **contract: _play_game(
            models, config.values["model"], lineup, seed, **contract
        ),
        series_id="held-out-0",
    )
    after = {path: _file_digest(path) for path in before}
    if before != after:
        raise RuntimeError("evaluation mutated a training artifact")
    valid = [outcome for outcome in outcomes if outcome["valid"]]
    table = RatingTable(parameters=config.values["rating"]).update(valid)
    leaderboard = [
        {
            "checkpoint_id": key,
            "mu": rating.mu,
            "sigma": rating.sigma,
            "ordinal": rating.mu - config.values["rating"]["ordinal_sigma"] * rating.sigma,
            "games": rating.games,
            "placements": rating.placements,
            "last_series": rating.last_series,
            "model_schema": config.values["model"]["schema"],
            "rating_namespace": table.namespace,
            "provisional": rating.games < 64,
        }
        for key, rating in table.leaderboard()
    ]
    report = {
        "mode": "rank",
        "ordinary_view": True,
        "rank_only": True,
        "gradients": False,
        "checkpoints": checkpoints,
        "valid_games": len(valid),
        "invalid_games": [outcome for outcome in outcomes if not outcome["valid"]],
        "rating_namespace": table.namespace,
    }
    (output / "evaluation.json").write_text(json.dumps(report, indent=2))
    publish_evaluation_records(output, outcomes, valid)
    (output / "leaderboard.json").write_text(json.dumps(leaderboard, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
