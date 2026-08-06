"""Greedy native match evaluation for cache-friendly BC checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch

from audit import EventPrefixActorCritic, StableBoundaryActorCritic
from zenith_ppo.checkpoint import resolve_latest, restore
from zenith_ppo.cli.evaluate import _play_games
from zenith_ppo.config import load
from zenith_ppo.evaluation.runner import paired_bootstrap, run_series_batched
from zenith_ppo.model.actor_critic import ActorCritic
from zenith_ppo.rollout.game_metrics import metric_values


def _checkpoint(path):
    path = Path(path)
    return resolve_latest(path) if (path / "latest").is_file() else path


def run(args):
    config = load(args.config)
    model_config = dict(config.values["model"])
    model_config["context_tokens"] = config.values["encoding"]["context_tokens"]
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation requested but unavailable")

    reference_path = _checkpoint(args.reference_checkpoint)
    reference = ActorCritic(model_config)
    reference.load_state_dict(restore(reference_path)["model"])
    reference.to(device).eval()

    candidate_class = {
        "event_prefix": EventPrefixActorCritic,
        "stable_boundary": StableBoundaryActorCritic,
    }[args.candidate_layout]
    candidate = candidate_class(model_config)
    candidate.load_state_dict(torch.load(
        args.candidate_model, map_location="cpu", weights_only=True
    ))
    candidate.to(device).eval()
    candidate_id = args.candidate_layout
    models = {candidate_id: candidate, "reference": reference}
    identities = (candidate_id, "reference", "reference", "reference")
    seeds = range(int(args.seed_start), int(args.seed_start) + int(args.seed_count))
    started = perf_counter()
    outcomes = run_series_batched(
        identities,
        seeds,
        lambda requests: _play_games(
            models, model_config, requests, device=device,
            token_budget=int(config.values["evaluation"].get(
                "token_budget", 65_536
            )),
            max_frames=int(config.values["rollout"]["max_frames_per_match"]),
            greedy=True,
        ),
        batch_size=int(args.batch_size),
        series_id="cache-friendly-bc-greedy",
    )
    valid = [row for row in outcomes if row["valid"]]
    totals = {}
    for outcome in valid:
        for identity, counts in outcome.get("gameplay_counts", {}).items():
            target = totals.setdefault(identity, {name: 0.0 for name in counts})
            for name, value in counts.items():
                target[name] += float(value)
    placements = {
        identity: [0, 0, 0, 0] for identity in (candidate_id, "reference")
    }
    for outcome in valid:
        for seat, identity in enumerate(outcome["checkpoint_ids"]):
            placements[identity][int(outcome["ranks"][seat])] += 1
    elapsed = perf_counter() - started
    report = {
        "schema": "zenith-cache-friendly-bc-evaluation-v1",
        "candidate_model": str(Path(args.candidate_model).resolve()),
        "candidate_layout": args.candidate_layout,
        "reference_checkpoint": str(reference_path.resolve()),
        "action_selection": "greedy",
        "seed_start": int(args.seed_start),
        "seed_count": int(args.seed_count),
        "games": len(valid),
        "invalid": len(outcomes) - len(valid),
        "elapsed_seconds": elapsed,
        "games_per_second": len(valid) / max(elapsed, 1e-9),
        "paired_bootstrap": paired_bootstrap(
            valid, candidate_id, "reference", seed=int(args.seed_start)
        ),
        "placements": placements,
        "gameplay": {
            identity: metric_values(counts)
            for identity, counts in totals.items()
        },
        "outcomes": outcomes,
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / ".evaluation.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(output / "evaluation.json")
    print(json.dumps({
        "evaluation": str(output / "evaluation.json"),
        "paired_bootstrap": report["paired_bootstrap"],
    }, sort_keys=True))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="training/configs/default.toml")
    parser.add_argument("--candidate-model", required=True)
    parser.add_argument(
        "--candidate-layout", choices=("event_prefix", "stable_boundary"),
        default="event_prefix",
    )
    parser.add_argument("--reference-checkpoint", required=True)
    parser.add_argument("--seed-start", type=int, default=10001)
    parser.add_argument("--seed-count", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
