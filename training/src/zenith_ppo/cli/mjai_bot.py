"""Load a Zenith checkpoint and play one RiichiLab MJAI game."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import threading

from ..config import load
from ..mjai import load_checkpoint_agent, run_matches


URLS = {
    "validate": "wss://game.riichi.dev/ws/validate",
    "ranked": "wss://game.riichi.dev/ws/ranked",
}
DEFAULT_CONFIG = "training/configs/shepard.toml"


def main(argv=None):
    parser = argparse.ArgumentParser(prog="zenith-mjai-bot")
    parser.add_argument(
        "--config", default=DEFAULT_CONFIG,
        help=f"model configuration (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mode", choices=tuple(URLS), default="validate")
    parser.add_argument("--url", help="override the RiichiLab WebSocket endpoint")
    parser.add_argument("--token-env", default="RIICHI_DEV_BOT_TOKEN")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--backend", default="sdpa")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument(
        "--exact-chi-variants", action="store_true",
        help=(
            "disable the legacy RiichiLab multi-shape chi workaround once "
            "the server honors the consumed field"
        ),
    )
    parser.add_argument(
        "--once", action="store_true",
        help="exit after one completed game instead of reconnecting",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    token = os.environ.get(args.token_env)
    if not token:
        parser.error(f"bot token is missing from environment variable {args.token_env}")
    import torch

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda requested but CUDA is unavailable")
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )
    agent = load_checkpoint_agent(
        load(args.config), args.checkpoint,
        device=device, backend=args.backend, use_bf16=args.bf16,
        legacy_chi_workaround=not args.exact_chi_variants,
    )
    stop_requested = threading.Event()
    previous_sigint = signal.getsignal(signal.SIGINT)

    def request_stop(_signum, _frame):
        if stop_requested.is_set():
            raise KeyboardInterrupt
        stop_requested.set()
        logging.getLogger(__name__).warning(
            "shutdown requested; waiting for the current match to complete "
            "(press Ctrl+C again to force exit)"
        )

    signal.signal(signal.SIGINT, request_stop)
    try:
        asyncio.run(run_matches(
            args.url or URLS[args.mode], token, agent,
            max_games=1 if args.once else None,
            stop_requested=stop_requested.is_set,
        ))
    except KeyboardInterrupt:
        logging.getLogger(__name__).warning("forced shutdown requested")
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
