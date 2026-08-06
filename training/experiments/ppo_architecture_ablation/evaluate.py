"""Greedy native evaluation for a PPO architecture-ablation candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch

from audit import _model
from zenith_ppo.cli.evaluate import _play_games
from zenith_ppo.config import load
from zenith_ppo.evaluation.runner import paired_bootstrap, run_series_batched
from zenith_ppo.rollout.game_metrics import metric_values


def run(args):
    config = load(args.config)
    values = config.values
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation requested but unavailable")

    models = {}
    for identity, variant, path in (
        ("candidate", args.candidate_variant, args.candidate_model),
        ("baseline", "baseline", args.baseline_model),
    ):
        model = _model(values, variant, device, int(values["run"]["seed"]))
        model.load_state_dict(torch.load(
            path, map_location=device, weights_only=True
        ))
        model.eval()
        models[identity] = model

    identities = ("candidate", "baseline", "baseline", "baseline")
    if int(args.matches) % 4:
        raise ValueError("matches must be divisible by four seat rotations")
    paired_seeds = int(args.matches) // 4
    seeds = range(int(args.seed_start), int(args.seed_start) + paired_seeds)
    model_config = dict(values["model"])
    model_config["context_tokens"] = values["encoding"]["context_tokens"]
    started = perf_counter()
    outcomes = run_series_batched(
        identities,
        seeds,
        lambda requests: _play_games(
            models, model_config, requests, device=device,
            token_budget=int(values["evaluation"].get("token_budget", 65_536)),
            max_frames=int(values["rollout"]["max_frames_per_match"]),
            greedy=True,
        ),
        batch_size=int(args.batch_size),
        series_id="ppo-architecture-ablation-greedy",
    )
    valid = [row for row in outcomes if row["valid"]]
    totals = {}
    placements = {identity: [0, 0, 0, 0] for identity in models}
    for outcome in valid:
        for identity, counts in outcome.get("gameplay_counts", {}).items():
            target = totals.setdefault(identity, {
                name: 0.0 for name in counts
            })
            for name, value in counts.items():
                target[name] += float(value)
        for seat, identity in enumerate(outcome["checkpoint_ids"]):
            placements[identity][int(outcome["ranks"][seat])] += 1
    elapsed = perf_counter() - started
    report = {
        "schema": "zenith-ppo-architecture-ablation-evaluation-v1",
        "candidate_variant": args.candidate_variant,
        "candidate_model": str(Path(args.candidate_model).resolve()),
        "baseline_model": str(Path(args.baseline_model).resolve()),
        "action_selection": "greedy",
        "requested_matches": int(args.matches),
        "paired_seeds": paired_seeds,
        "matches": len(valid),
        "invalid": len(outcomes) - len(valid),
        "batch_size": int(args.batch_size),
        "elapsed_seconds": elapsed,
        "games_per_second": len(valid) / max(elapsed, 1e-9),
        "paired_bootstrap": paired_bootstrap(
            valid, "candidate", "baseline", seed=int(args.seed_start)
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="training/configs/default.toml")
    parser.add_argument("--candidate-model", required=True)
    parser.add_argument("--baseline-model", required=True)
    parser.add_argument(
        "--candidate-variant", choices=(
            "role_aware_tiles", "deeper_action_head", "post_action_shape",
        ), required=True,
    )
    parser.add_argument("--seed-start", type=int, default=30_001)
    parser.add_argument("--matches", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
