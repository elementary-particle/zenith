"""Authenticated WebSocket service for remote Akagi clients."""

from __future__ import annotations

import asyncio
import hmac
import json
import logging

from websockets.exceptions import ConnectionClosed

from .akagi import AkagiProtocolError, AkagiSession


LOGGER = logging.getLogger(__name__)
MAX_BATCH_EVENTS = 512


def authorized(headers, token: str | None) -> bool:
    """Check an optional bearer token without leaking it through timing."""
    if token is None:
        return True
    supplied = headers.get("Authorization", "")
    expected = f"Bearer {token}"
    return hmac.compare_digest(str(supplied), expected)


class AkagiWebSocketServer:
    """Create isolated game sessions while sharing one loaded policy."""

    def __init__(self, agent, *, token: str | None = None):
        self.agent = agent
        self.token = token
        # Torch inference on a shared CUDA model is serialized. State mutation
        # happens inside the same critical section, so one connection cannot
        # interleave two batches from its session either.
        self.inference_lock = asyncio.Lock()

    async def handler(self, websocket) -> None:
        if not authorized(websocket.request.headers, self.token):
            await websocket.close(code=1008, reason="unauthorized")
            return
        peer = getattr(websocket, "remote_address", None)
        LOGGER.info("Akagi client connected: %s", peer)
        session = AkagiSession(self.agent.new_session())
        try:
            async for payload in websocket:
                response = await self._react(session, payload)
                await websocket.send(json.dumps(response, separators=(",", ":")))
                if session.ended:
                    await websocket.close(code=1000, reason="end_game")
                    return
        except ConnectionClosed as error:
            # Akagi may terminate its subprocess, hit its reaction timeout, or
            # lose the network before the WebSocket close handshake completes.
            # This ends only this isolated game session and isn't a server
            # handler failure, so don't let websockets emit a traceback.
            LOGGER.warning("Akagi client connection ended: %s (%s)", peer, error)
        finally:
            LOGGER.info("Akagi client disconnected: %s", peer)

    async def _react(self, session: AkagiSession, payload) -> dict:
        if not isinstance(payload, str):
            LOGGER.warning("Akagi client sent a non-text WebSocket message")
            return {"type": "none"}
        events = None
        try:
            events = json.loads(payload)
            if not isinstance(events, list):
                raise AkagiProtocolError("WebSocket messages must be event arrays")
            if len(events) > MAX_BATCH_EVENTS:
                raise AkagiProtocolError(
                    f"event batch exceeds the {MAX_BATCH_EVENTS}-event limit"
                )
            async with self.inference_lock:
                # A single loaded policy is intentionally serialized. Typical
                # inference is far shorter than Akagi's reaction budget, and
                # keeping it on this thread avoids spawning a new CUDA-facing
                # executor worker for every game connection.
                return session.react(events)
        except (AkagiProtocolError, json.JSONDecodeError) as error:
            session.desynchronize()
            tail = events[-3:] if isinstance(events, list) else []
            summary = [
                {
                    key: event[key]
                    for key in ("type", "actor", "target", "pai", "tsumogiri")
                    if isinstance(event, dict) and key in event
                }
                for event in tail
            ]
            LOGGER.warning(
                "invalid Akagi request; ignoring events until the next round: "
                "%s (recent=%s)",
                error,
                json.dumps(summary, separators=(",", ":")),
            )
            return {"type": "none"}
        except Exception:
            LOGGER.exception("Akagi inference failed")
            return {"type": "none"}
