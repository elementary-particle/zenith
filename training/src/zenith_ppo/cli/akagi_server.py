"""Serve a Zenith checkpoint to remote Akagi relay plugins."""

from __future__ import annotations

import argparse
import asyncio
import logging
import math
import os
import ssl

from ..akagi_server import AkagiWebSocketServer
from ..config import load
from ..mjai import load_checkpoint_agent


DEFAULT_CONFIG = "training/configs/default.toml"


def _temperature(value):
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise argparse.ArgumentTypeError("temperature must be finite and non-negative")
    return result


def _ssl_context(cert: str | None, key: str | None):
    if bool(cert) != bool(key):
        raise ValueError("--tls-cert and --tls-key must be supplied together")
    if cert is None:
        return None
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert, key)
    return context


def _is_loopback(host: str) -> bool:
    return host in {"127.0.0.1", "::1", "localhost"}


async def _serve(args, agent, token):
    from websockets.asyncio.server import serve

    service = AkagiWebSocketServer(agent, token=token)
    tls = _ssl_context(args.tls_cert, args.tls_key)
    scheme = "wss" if tls is not None else "ws"
    async with serve(
        service.handler,
        args.host,
        args.port,
        ssl=tls,
        max_size=args.max_message_bytes,
        ping_interval=20,
        ping_timeout=20,
    ):
        logging.getLogger(__name__).info(
            "Zenith Akagi server listening on %s://%s:%d",
            scheme,
            args.host,
            args.port,
        )
        await asyncio.Future()


def main(argv=None):
    parser = argparse.ArgumentParser(prog="zenith-akagi-server")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--token-env", default="ZENITH_AKAGI_TOKEN")
    parser.add_argument("--tls-cert")
    parser.add_argument("--tls-key")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--backend", default="sdpa")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--temperature", type=_temperature, default=0.0)
    parser.add_argument("--max-message-bytes", type=int, default=2 * 1024 * 1024)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65_535:
        parser.error("--port must be between 1 and 65535")
    if args.max_message_bytes <= 0:
        parser.error("--max-message-bytes must be positive")
    try:
        tls = _ssl_context(args.tls_cert, args.tls_key)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    token = os.environ.get(args.token_env) or None
    if not _is_loopback(args.host) and token is None:
        parser.error(
            f"non-loopback listeners require a bearer token in {args.token_env}"
        )
    if not _is_loopback(args.host) and tls is None:
        logging.getLogger(__name__).warning(
            "serving unencrypted WebSockets on a non-loopback address; "
            "use --tls-cert/--tls-key or a private VPN/tunnel"
        )
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    import torch

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda requested but CUDA is unavailable")
    agent = load_checkpoint_agent(
        load(args.config),
        args.checkpoint,
        device=device,
        backend=args.backend,
        use_bf16=args.bf16,
        legacy_chi_workaround=False,
        temperature=args.temperature,
    )
    try:
        asyncio.run(_serve(args, agent, token))
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
