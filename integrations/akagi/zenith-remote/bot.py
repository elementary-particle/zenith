"""Akagi subprocess relay for a remote Zenith WebSocket server."""

from __future__ import annotations

import json
import os
import ssl
import sys

from websockets.sync.client import connect


DEFAULTS = {
    "server_url": "ws://127.0.0.1:8765",
    "api_token": "",
    "connect_timeout_seconds": 4.0,
    "response_timeout_seconds": 4.5,
    "allow_invalid_tls": False,
}


def load_config() -> dict:
    config = dict(DEFAULTS)
    path = os.environ.get("AKAGI_BOT_CONFIG")
    if not path:
        return config
    try:
        with open(path, encoding="utf-8") as source:
            values = json.load(source)
        if not isinstance(values, dict):
            raise ValueError("resolved bot settings must be a JSON object")
        for key in DEFAULTS:
            if key in values:
                config[key] = values[key]
    except Exception as error:
        print(f"zenith relay config error: {error}", file=sys.stderr, flush=True)
    return config


class ZenithRelay:
    def __init__(self, config: dict):
        self.url = str(config["server_url"])
        self.token = str(config["api_token"])
        self.connect_timeout = float(config["connect_timeout_seconds"])
        self.response_timeout = float(config["response_timeout_seconds"])
        self.allow_invalid_tls = bool(config["allow_invalid_tls"])
        self.websocket = None
        self.desynchronized = False

    def _connect(self):
        headers = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        tls = None
        if self.url.startswith("wss://") and self.allow_invalid_tls:
            tls = ssl.create_default_context()
            tls.check_hostname = False
            tls.verify_mode = ssl.CERT_NONE
        self.websocket = connect(
            self.url,
            additional_headers=headers,
            ssl=tls,
            open_timeout=self.connect_timeout,
            close_timeout=1,
            ping_interval=20,
            ping_timeout=20,
            proxy=None,
        )

    def react(self, raw_batch: str) -> str:
        # Parse locally before forwarding so malformed stdin can never desync
        # the one-request/one-response Akagi contract.
        events = json.loads(raw_batch)
        if not isinstance(events, list):
            raise ValueError("Akagi input must be an MJAI event array")
        starts_game = any(event.get("type") == "start_game" for event in events)
        if starts_game:
            self.close()
            self.desynchronized = False
        if self.desynchronized:
            raise ConnectionError("remote session is desynchronized until the next game")
        if self.websocket is None:
            self._connect()
        self.websocket.send(json.dumps(events, separators=(",", ":")))
        payload = self.websocket.recv(timeout=self.response_timeout)
        if not isinstance(payload, str):
            raise ValueError("Zenith server returned a non-text response")
        response = json.loads(payload)
        if not isinstance(response, dict) or not isinstance(response.get("type"), str):
            raise ValueError("Zenith server returned an invalid MJAI reaction")
        return json.dumps(response, separators=(",", ":"))

    def fail(self) -> None:
        # Reconnecting in the middle of a game would create an empty server
        # session and silently feed it only a suffix of the MJAI stream. Fail
        # closed until the next authoritative start_game instead.
        self.close()
        self.desynchronized = True

    def close(self) -> None:
        if self.websocket is None:
            return
        try:
            self.websocket.close()
        except Exception:
            pass
        self.websocket = None


def main() -> int:
    relay = ZenithRelay(load_config())
    try:
        for raw in sys.stdin:
            line = raw.strip()
            if not line:
                continue
            try:
                response = relay.react(line)
            except Exception as error:
                relay.fail()
                print(f"zenith relay error: {error}", file=sys.stderr, flush=True)
                response = '{"type":"none"}'
            sys.stdout.write(response + "\n")
            sys.stdout.flush()
            try:
                events = json.loads(line)
            except Exception:
                continue
            if any(event.get("type") == "end_game" for event in events):
                break
    finally:
        relay.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
