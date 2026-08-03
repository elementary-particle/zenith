"""Command-line entry point for durable PPO training."""

from __future__ import annotations

import argparse

from ..config import load


def run_one_update(config, output, *, resume=None, weights_only=False):
    """Run one production update for smoke and reproducibility checks."""
    from ..orchestrator import run_training

    return run_training(
        config,
        output,
        resume=resume,
        weights_only=weights_only,
        max_updates=1,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(prog="zenith-ppo-train")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--weights-only", action="store_true")
    parser.add_argument(
        "--initial-checkpoint",
        help="initialize from a completed behavior-cloning checkpoint",
    )
    parser.add_argument(
        "--max-updates",
        type=int,
        help="stop after this many updates in this process",
    )
    parser.add_argument(
        "--profile-stages",
        action="store_true",
        help="record low-overhead training stage timings",
    )
    parser.add_argument(
        "--skip-periodic-evaluation",
        action="store_true",
        help=(
            "defer scheduled arena games without changing the training "
            "configuration or exact-resume identity"
        ),
    )
    args = parser.parse_args(argv)
    if args.initial_checkpoint and args.resume:
        parser.error("--initial-checkpoint is mutually exclusive with --resume")
    if args.weights_only and not args.resume:
        parser.error("--weights-only requires --resume")
    from ..orchestrator import run_training

    run_training(
        load(args.config),
        args.output,
        resume=args.resume,
        weights_only=args.weights_only,
        initial_checkpoint=args.initial_checkpoint,
        max_updates=args.max_updates,
        profile_stages=args.profile_stages,
        skip_periodic_evaluation=args.skip_periodic_evaluation,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
