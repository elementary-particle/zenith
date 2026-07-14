"""Acceptance, capability, reproducibility, and metric-verification commands."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

from ..capabilities import configure, detect
from ..config import load
from ..rewards.curriculum import Curriculum


def _one_update(config, output):
    from .train import run_one_update
    return run_one_update(config, output)


def verify_metrics(run):
    path = Path(run) / "metrics" / "canonical.jsonl"
    previous, keys = -1, set()
    raw = path.read_text()
    forbidden = ("concealed_tile", "priv_wall", "rng_state", "hidden_target")
    if any(term in raw for term in forbidden): raise ValueError("secret payload found in metrics")
    for line in raw.splitlines():
        row = json.loads(line)
        if row["sequence"] <= previous: raise ValueError("metric sequence is not monotonic")
        key = (row["name"], row["axis"], row["step"], row["source"])
        if key in keys: raise ValueError("duplicate metric key")
        previous = row["sequence"]; keys.add(key)
    return {"records": len(keys), "last_sequence": previous}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="zenith-ppo-smoke")
    sub = parser.add_subparsers(dest="command", required=True)
    cap = sub.add_parser("capabilities"); cap.add_argument("--profile", required=True)
    for name in ("run", "curriculum", "population", "reproduce", "resume", "benchmark", "evaluation", "rating", "convergence"):
        command = sub.add_parser(name); command.add_argument("--config", required=True)
        command.add_argument("--profile"); command.add_argument("--output", required=True)
        if name in ("reproduce", "resume"): command.add_argument("--runs", type=int, default=2)
    verify = sub.add_parser("verify-metrics"); verify.add_argument("--run", required=True)
    args = parser.parse_args(argv)
    if args.command == "capabilities": print(json.dumps(detect(args.profile).as_dict(), sort_keys=True)); return 0
    if args.command == "verify-metrics": print(json.dumps(verify_metrics(args.run))); return 0
    config = load(args.config); configure(args.profile or config.values["run"]["profile"])
    output = Path(args.output)
    if args.command == "curriculum":
        schedule = Curriculum(config.values["curriculum"])
        evidence = [asdict(schedule.snapshot(update, update)) for update in range(config.values["curriculum"]["total_updates"] + 1)]
        for row in evidence:
            if abs(sum(row["weights"]) - 1.0) > 1e-12:
                raise RuntimeError("curriculum weights are not convex")
        if tuple(evidence[-1]["weights"]) != (0.0, 0.0, 1.0):
            raise RuntimeError("final curriculum anchor is not rank-only")
        output.mkdir(parents=True, exist_ok=True)
        (output / "curriculum.json").write_text(json.dumps(evidence, default=list, indent=2)); return 0
    if args.command == "population":
        from ..population.registry import CheckpointPool, PoolEntry
        from ..population.sampler import UniformSampler
        from ..seeds import SeedStreams

        pool = CheckpointPool(config.compatibility)
        for index in range(4):
            checkpoint_id = f"historical-{index}"
            pool.admit(PoolEntry(
                checkpoint_id,
                f"checkpoints/{checkpoint_id}",
                config.compatibility,
                "smoke",
                index,
            ))
        snapshot = pool.snapshot("rating-smoke")
        def draw():
            sampler = UniformSampler(
                SeedStreams(config.values["run"]["seed"]),
                learner_seats=config.values["population"]["learner_seats"],
                shortage=config.values["population"]["shortage"],
            )
            return [
                asdict(sampler.sample(
                    snapshot,
                    environment_id=environment_id,
                    generation=1,
                    current_id="current",
                    policy_version=0,
                ))
                for environment_id in range(config.values["env"]["num_envs"])
            ]
        first, second = draw(), draw()
        if first != second:
            raise RuntimeError("seeded population assignment is not repeatable")
        if any(len(set(row["seat_policy_ids"]) - {"current"}) != 2 for row in first):
            raise RuntimeError("population lineup did not select two distinct opponents")
        output.mkdir(parents=True, exist_ok=True)
        (output / "population.json").write_text(
            json.dumps({"pool_snapshot": snapshot.snapshot_id, "lineups": first}, indent=2),
            encoding="utf-8",
        )
        return 0
    if args.command in ("evaluation", "rating"):
        from ..evaluation.ratings import RatingTable
        from ..evaluation.runner import run_series

        checkpoint_ids = ("a", "b", "c", "d")
        seeds = config.values["evaluation"]["held_out_seeds"]
        outcomes = run_series(
            checkpoint_ids,
            seeds,
            lambda lineup, seed, **contract: {
                "ranks": tuple(sorted(range(4), key=lambda seat: lineup[seat])),
                "scores": tuple(25000 + ((seed + seat) % 4) * 100 for seat in range(4)),
                "valid": contract == {
                    "ordinary_view": True, "rank_only": True, "gradients": False
                },
            },
            series_id="smoke",
        )
        table = RatingTable(parameters=config.values["rating"]).update(outcomes)
        output.mkdir(parents=True, exist_ok=True)
        payload = {
            "command": args.command,
            "ordinary_view": True,
            "rank_only": True,
            "outcomes": outcomes,
            "leaderboard": [
                {"checkpoint_id": key, **asdict(rating)}
                for key, rating in table.leaderboard()
            ],
        }
        (output / f"{args.command}.json").write_text(json.dumps(payload, sort_keys=True, indent=2))
        return 0
    if args.command == "convergence":
        from ..evaluation.runner import convergence

        budget = config.values["rollout"]["learner_decisions_per_update"]
        curves = {
            "curriculum": [[(0, 0.0), (budget, 1.0)]] * 3,
            "rank-only": [[(0, 0.0), (budget, 0.4)]] * 3,
        }
        payload = convergence(curves, 0.5)
        output.mkdir(parents=True, exist_ok=True)
        (output / "convergence.json").write_text(json.dumps(payload, indent=2))
        return 0
    if args.command == "reproduce":
        results = [_one_update(config, output / f"run-{index}") for index in range(args.runs)]
        digests = {
            (result["trajectory_digest"], result["parameter_digest_after"])
            for result in results
        }
        if len(digests) != 1: raise RuntimeError("repeated smoke updates diverged")
        return 0
    if args.command == "resume":
        from .train import run_one_update

        base = _one_update(config, output / "base")
        checkpoint = base["checkpoint_path"]
        results = [
            run_one_update(config, output / f"resumed-{index}", resume=checkpoint)
            for index in range(args.runs)
        ]
        digests = {
            (result["trajectory_digest"], result["parameter_digest_after"])
            for result in results
        }
        if len(digests) != 1:
            raise RuntimeError("restored next updates diverged")
        return 0
    if args.command == "run":
        _one_update(config, output)
        return 0
    evidence = _one_update(config, output)
    if args.command == "benchmark": print(json.dumps(evidence, sort_keys=True))
    return 0


if __name__ == "__main__": raise SystemExit(main())
