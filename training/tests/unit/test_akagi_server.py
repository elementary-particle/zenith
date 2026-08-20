import asyncio
import json
from types import SimpleNamespace

from websockets.exceptions import ConnectionClosedError

from zenith_ppo.akagi_server import AkagiWebSocketServer, authorized


class _FactoryAgent:
    def new_session(self):
        return _PassiveAgent()


class _PassiveAgent:
    pending_riichi_tile = None

    def reset(self):
        pass

    def observe_event(self, _event):
        pass

    def act(self, _observation):
        raise AssertionError("boundary events must not invoke inference")


class _FakeWebSocket:
    def __init__(self, messages, authorization="Bearer secret"):
        self.request = SimpleNamespace(headers={"Authorization": authorization})
        self.remote_address = ("127.0.0.1", 12345)
        self.messages = list(messages)
        self.sent = []
        self.closed = None

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.messages:
            raise StopAsyncIteration
        value = self.messages.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value

    async def send(self, value):
        self.sent.append(value)

    async def close(self, *, code, reason):
        self.closed = (code, reason)


def test_bearer_authorization():
    assert authorized({"Authorization": "Bearer secret"}, "secret")
    assert not authorized({"Authorization": "Bearer other"}, "secret")
    assert authorized({}, None)


def test_server_replies_once_per_batch_and_closes_after_end_game():
    websocket = _FakeWebSocket([
        json.dumps([{
            "type": "start_game", "names": ["a", "b", "c", "d"],
            "id": 0, "num_players": 4,
        }]),
        json.dumps([{"type": "end_game"}]),
    ])
    server = AkagiWebSocketServer(_FactoryAgent(), token="secret")

    asyncio.run(server.handler(websocket))

    assert [json.loads(value) for value in websocket.sent] == [
        {"type": "none"}, {"type": "none"},
    ]
    assert websocket.closed == (1000, "end_game")


def test_server_rejects_bad_token_before_reading_messages():
    websocket = _FakeWebSocket([], authorization="Bearer wrong")
    server = AkagiWebSocketServer(_FactoryAgent(), token="secret")

    asyncio.run(server.handler(websocket))

    assert websocket.closed == (1008, "unauthorized")
    assert websocket.sent == []


def test_abrupt_client_disconnect_does_not_escape_handler(caplog):
    websocket = _FakeWebSocket([ConnectionClosedError(None, None, None)])
    server = AkagiWebSocketServer(_FactoryAgent(), token="secret")

    asyncio.run(server.handler(websocket))

    assert "no close frame received or sent" in caplog.text


def test_invalid_stream_logs_once_and_quarantines_current_hand(caplog):
    server = AkagiWebSocketServer(_FactoryAgent(), token="secret")
    from zenith_ppo.akagi import AkagiSession

    session = AkagiSession(_PassiveAgent())
    session.react([{
        "type": "start_game", "names": ["a", "b", "c", "d"],
        "id": 0, "num_players": 4,
    }])
    invalid = json.dumps([{
        "type": "dahai", "actor": 0, "pai": "1m", "tsumogiri": False,
    }])

    assert asyncio.run(server._react(session, invalid)) == {"type": "none"}
    assert session.desynchronized
    assert asyncio.run(server._react(session, invalid)) == {"type": "none"}
    assert caplog.text.count("ignoring events until the next round") == 1
    assert "recent=" in caplog.text
    assert "own hand does not contain 1m" in caplog.text
