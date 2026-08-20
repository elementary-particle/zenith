import importlib.util
import json
from pathlib import Path


PLUGIN = (
    Path(__file__).resolve().parents[3]
    / "integrations" / "akagi" / "zenith-remote" / "bot.py"
)


def _module():
    spec = importlib.util.spec_from_file_location("zenith_remote_bot", PLUGIN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Socket:
    def __init__(self):
        self.sent = []
        self.closed = False

    def send(self, value):
        self.sent.append(json.loads(value))

    def recv(self, *, timeout):
        assert timeout == 4.5
        return json.dumps({
            "type": "none",
            "meta": {"policy": {"top_actions": [{"rank": 1}]}},
        })

    def close(self):
        self.closed = True


def test_relay_forwards_one_batch_and_parses_response(monkeypatch):
    plugin = _module()
    socket = _Socket()
    monkeypatch.setattr(plugin, "connect", lambda *_args, **_kwargs: socket)
    relay = plugin.ZenithRelay(dict(plugin.DEFAULTS))
    batch = '[{"type":"start_game","id":0}]'

    assert json.loads(relay.react(batch)) == {
        "type": "none",
        "meta": {"policy": {"top_actions": [{"rank": 1}]}},
    }
    assert socket.sent == [[{"type": "start_game", "id": 0}]]


def test_relay_does_not_reconnect_mid_game_after_failure(monkeypatch):
    plugin = _module()
    connections = []

    def connect(*_args, **_kwargs):
        socket = _Socket()
        connections.append(socket)
        return socket

    monkeypatch.setattr(plugin, "connect", connect)
    relay = plugin.ZenithRelay(dict(plugin.DEFAULTS))
    relay.react('[{"type":"start_game","id":0}]')
    relay.fail()

    try:
        relay.react('[{"type":"tsumo","actor":0,"pai":"1m"}]')
    except ConnectionError as error:
        assert "desynchronized" in str(error)
    else:
        raise AssertionError("a partial game must not reconnect as a fresh session")
    assert len(connections) == 1

    relay.react('[{"type":"start_game","id":0}]')
    assert len(connections) == 2
