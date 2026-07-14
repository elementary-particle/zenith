"""Run one synchronized training update and report stage attribution."""

from __future__ import annotations

import argparse
import json

from ..config import load
from ..orchestrator import run_training


def main(argv=None):
    parser = argparse.ArgumentParser(prog="zenith-ppo-profile")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--updates", type=int, default=1)
    args = parser.parse_args(argv)
    if args.updates < 1:
        parser.error("--updates must be positive")
    result = run_training(
        load(args.config), args.output, max_updates=args.updates, profile_stages=True
    )
    print(json.dumps(result["profile"], sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
