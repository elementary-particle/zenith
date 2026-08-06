import asyncio
import json
from types import SimpleNamespace

import pytest

from zenith_ppo.mjai import (
    CheckpointAgent,
    _InferenceContext,
    _decision_phase,
    _event_rows,
    _legacy_chi_compatible_group,
    _public_melds,
    action_response,
    encode_observation,
    load_checkpoint_agent,
    play_connection,
    run_matches,
    validate_possible_action,
)


class _Action:
    def __init__(self, action_type, response):
        self.action_type = action_type
        self.response = response

    def to_mjai(self):
        return json.dumps(self.response)


def _candidate(action_type, response):
    return SimpleNamespace(action=_Action(action_type, response))


def test_legacy_chi_workaround_passes_instead_of_executing_wrong_shape():
    import torch

    encoded = SimpleNamespace(
        native_candidates=(
            _candidate(7, {"type": "none"}),
            _candidate(1, {
                "type": "chi", "pai": "5p", "consumed": ["3p", "4p"],
            }),
            _candidate(1, {
                "type": "chi", "pai": "5p", "consumed": ["4p", "6p"],
            }),
        ),
        action_representatives=(0, 1, 2),
        action_members=((0,), (1,), (2,)),
    )
    probabilities = torch.tensor([0.3354, 0.0050, 0.6596]).log()

    assert _legacy_chi_compatible_group(encoded, probabilities, 2) == 0


def test_legacy_chi_workaround_can_select_first_server_shape():
    import torch

    encoded = SimpleNamespace(
        native_candidates=(
            _candidate(7, {"type": "none"}),
            _candidate(1, {
                "type": "chi", "pai": "2m", "consumed": ["1m", "3m"],
            }),
            _candidate(1, {
                "type": "chi", "pai": "2m", "consumed": ["3m", "4m"],
            }),
        ),
        action_representatives=(0, 1, 2),
        action_members=((0,), (1,), (2,)),
    )
    probabilities = torch.tensor([0.10, 0.35, 0.55]).log()

    assert _legacy_chi_compatible_group(encoded, probabilities, 2) == 1


def test_legacy_chi_workaround_preserves_first_shape_and_non_chi_actions():
    import torch

    encoded = SimpleNamespace(
        native_candidates=(
            _candidate(7, {"type": "none"}),
            _candidate(1, {
                "type": "chi", "pai": "5m", "consumed": ["3m", "4m"],
            }),
        ),
        action_representatives=(0, 1),
        action_members=((0,), (1,)),
    )
    log_probabilities = torch.tensor([0.25, 0.75]).log()

    assert _legacy_chi_compatible_group(encoded, log_probabilities, 0) == 0
    assert _legacy_chi_compatible_group(encoded, log_probabilities, 1) == 1


def test_stateful_mjai_context_preserves_kyoku_progress_and_boundary():
    context = _InferenceContext()
    start = {
        "type": "start_kyoku", "bakaze": "S", "kyoku": 2,
        "honba": 1, "kyotaku": 0, "oya": 1,
        "scores": [28_000, 25_000, 24_000, 23_000],
    }
    for event in (
        start,
        {"type": "tsumo", "actor": 1, "pai": "5m"},
        {"type": "reach", "actor": 1},
        {"type": "dahai", "actor": 1, "pai": "5m", "tsumogiri": True},
        {"type": "reach_accepted", "actor": 1},
        {"type": "tsumo", "actor": 2, "pai": "3p"},
        {"type": "dahai", "actor": 2, "pai": "3p", "tsumogiri": True},
        {
            "type": "pon", "actor": 3, "target": 2, "pai": "3p",
            "consumed": ["3p", "3p"],
        },
    ):
        context.observe_event(event)

    assert context.complete
    assert context.live_wall_remaining == 68
    assert context.current_scores == [28_000, 24_000, 24_000, 23_000]
    assert context.boundary_frame({"scores": tuple(context.current_scores)})["scores"] \
        == tuple(start["scores"])
    rivers = context.public_rivers()
    assert rivers[0]["riichi_declaration"] and not rivers[0]["called"]
    assert rivers[1]["called"]


def test_mjai_call_and_meld_encoding_matches_native_canonical_shape():
    observation = SimpleNamespace(scores=[25_000] * 4, riichi_sticks=0)
    rows, _ = _event_rows(observation, [{
        "type": "chi", "actor": 2, "target": 1, "pai": "7s",
        "consumed": ["6s", "8s"],
    }])
    from zenith_ppo.encoding.events import factorize_event

    categorical, _ = factorize_event(rows[0], observer=2)
    assert categorical[8] == 3  # called tile is the middle of 6s-7s-8s

    meld = SimpleNamespace(meld_type=0, tiles=[96, 92, 100])
    encoded = _public_melds(SimpleNamespace(melds=[(), (), (meld,), ()]))
    assert encoded[0]["tiles"] == (92, 96, 100)


def test_mjai_decision_phase_excludes_automatic_replacement_transition():
    def action(kind):
        return SimpleNamespace(action_type=kind)

    self_turn = SimpleNamespace(legal_actions=lambda: [action(0), action(9)])
    reaction = SimpleNamespace(legal_actions=lambda: [action(7), action(4)])

    assert _decision_phase(self_turn, [{"type": "kakan", "actor": 0}]) == 1
    assert _decision_phase(reaction, [{"type": "dahai", "actor": 0}]) == 2
    assert _decision_phase(reaction, [{"type": "ankan", "actor": 0}]) == 3


def _riichi_observation():
    import riichienv

    env = riichienv.RiichiEnv(seed=6)
    observations = env.reset()
    for _ in range(500):
        selected = {}
        for seat, observation in observations.items():
            legal = observation.legal_actions()
            if any(int(action.action_type) == 5 for action in legal):
                return env, observation
            selected[seat] = (
                next((action for action in legal if int(action.action_type) in (4, 6)), None)
                or next((action for action in legal if int(action.action_type) == 0), None)
                or next((action for action in legal if int(action.action_type) == 7), legal[0])
            )
        observations = env.step(selected)
    pytest.fail("fixture seed did not expose riichi")


class _ChooseLast:
    def eval(self):
        return self

    def forward_actor(self, **inputs):
        import torch

        count = int(inputs["action_offsets"][-1])
        return SimpleNamespace(log_probabilities=torch.arange(count, dtype=torch.float32))


def test_observation_bridge_groups_physical_copies_and_runs_model_batch():
    import riichienv
    from zenith_ppo.encoding.packing import model_batch

    observation = next(iter(riichienv.RiichiEnv(seed=42).reset().values()))
    encoded = encode_observation(observation)
    inputs = model_batch([encoded])

    assert encoded.token_factors.shape[1] == 10
    assert len(encoded.native_candidates) == 14
    assert len(encoded.action_representatives) == 12
    assert inputs["action_offsets"].tolist() == [0, 12]


def test_observation_bridge_emits_exact_current_rivers_and_melds():
    import riichienv
    from zenith_ppo.encoding.schema import TokenKind

    env = riichienv.RiichiEnv(seed=42)
    observations = env.reset()
    for _ in range(500):
        observation = next(iter(observations.values()))
        if any(observation.melds):
            encoded = encode_observation(observation)
            kinds = encoded.token_factors[:, 1]
            assert (kinds == int(TokenKind.RIVER)).any()
            assert (kinds == int(TokenKind.MELD)).any()
            break
        selected = {}
        for seat, row in observations.items():
            legal = row.legal_actions()
            selected[seat] = (
                next((action for action in legal
                      if int(action.action_type) in (1, 2, 3)), None)
                or next((action for action in legal
                         if int(action.action_type) in (4, 6)), None)
                or next((action for action in legal
                         if int(action.action_type) == 0), None)
                or next((action for action in legal
                         if int(action.action_type) == 7), legal[0])
            )
        observations = env.step(selected)
    else:
        pytest.fail("fixture seed did not expose a public meld")


def test_checkpoint_loader_accepts_production_architecture(tmp_path):
    from zenith_ppo.checkpoint import publish
    from zenith_ppo.config import load
    from zenith_ppo.model.actor_critic import ActorCritic

    config = load("training/configs/default.toml")
    model_config = dict(config.values["model"])
    model_config["context_tokens"] = config.values["encoding"]["context_tokens"]
    model = ActorCritic(model_config)
    publish(tmp_path, {
        "model": model.state_dict(),
        "state": {
            "architecture": "shared-shape-rank-v-bc-v1"
        },
    })

    agent = load_checkpoint_agent(config, tmp_path)

    assert agent.model is not None


def test_combined_model_riichi_is_split_across_two_protocol_requests():
    env, observation = _riichi_observation()
    agent = CheckpointAgent(_ChooseLast())
    try:
        reach = agent.act(observation)
        assert json.loads(reach.to_mjai())["type"] == "reach"
        assert agent.pending_riichi_tile == 61

        following = env.step({int(observation.player_id): reach})
        discard_observation = following[int(observation.player_id)]
        discard = agent.act(discard_observation)
        assert int(discard.tile) == 61
        assert action_response(discard, discard_observation)["type"] == "dahai"
        assert agent.pending_riichi_tile is None
    finally:
        del env


def test_possible_action_validation_uses_mjai_action_identity():
    response = {
        "type": "chi", "actor": 1, "pai": "5m",
        "consumed": ["4m", "6m"],
    }
    validate_possible_action(response, [{
        "type": "chi", "pai": "5m", "consumed": ["6m", "4m"],
    }])
    with pytest.raises(RuntimeError, match="possible_actions"):
        validate_possible_action(response, [{
            "type": "pon", "pai": "5m", "consumed": ["5m", "5m"],
        }])


class _FakeAgent:
    def __init__(self, action):
        self.action = action
        self.resets = 0
        self.events = []

    def reset(self):
        self.resets += 1

    def act(self, observation):
        return self.action

    def observe_event(self, event):
        self.events.append(dict(event))


class _FakeSocket:
    def __init__(self, messages):
        self.messages = iter(messages)
        self.sent = []

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self.messages)
        except StopIteration:
            raise StopAsyncIteration

    async def send(self, value):
        self.sent.append(value)


def test_server_loop_only_replies_to_request_action_and_echoes_request_id():
    import riichienv

    observation = next(iter(riichienv.RiichiEnv(seed=42).reset().values()))
    action = observation.legal_actions()[0]
    possible = json.loads(action.to_mjai())
    socket = _FakeSocket([
        json.dumps({"type": "start_game", "id": 0}),
        json.dumps({"type": "tsumo", "actor": 0, "pai": "5m"}),
        json.dumps({
            "type": "request_action", "request_id": 42,
            "observation": observation.serialize_to_base64(),
            "possible_actions": [possible],
        }),
        json.dumps({"type": "action_ack", "request_id": 42, "status": "accepted"}),
        json.dumps({"type": "end_game", "scores": [25_000] * 4}),
    ])
    agent = _FakeAgent(action)

    result = asyncio.run(play_connection(socket, agent))

    assert agent.resets == 1
    assert result["type"] == "end_game"
    assert len(socket.sent) == 1
    assert json.loads(socket.sent[0])["request_id"] == 42
    assert [event["type"] for event in agent.events] == [
        "start_game", "tsumo", "end_game",
    ]


def test_match_runner_reconnects_and_retries_until_game_limit():
    calls = []
    delays = []

    async def connect_one(url, token, agent):
        calls.append((url, token, agent))
        if len(calls) == 1:
            raise ConnectionError("temporary disconnect")
        return {"type": "end_game", "scores": [25_000] * 4}

    async def sleep(delay):
        delays.append(delay)

    agent = object()
    completed = asyncio.run(run_matches(
        "wss://example.test/ws/ranked", "secret", agent,
        max_games=2,
        retry_initial_seconds=0.25,
        retry_max_seconds=1.0,
        connect_one=connect_one,
        sleep=sleep,
    ))

    assert completed == 2
    assert len(calls) == 3
    assert delays == [0.25]


def test_match_runner_finishes_active_match_after_stop_request():
    stop_requested = False
    calls = 0

    async def connect_one(_url, _token, _agent):
        nonlocal calls, stop_requested
        calls += 1
        stop_requested = True
        return {"type": "end_game", "scores": [25_000] * 4}

    completed = asyncio.run(run_matches(
        "wss://example.test/ws/ranked", "secret", object(),
        connect_one=connect_one,
        stop_requested=lambda: stop_requested,
    ))

    assert completed == 1
    assert calls == 1


def test_match_runner_does_not_retry_failed_connection_after_stop_request():
    stop_requested = False
    delays = []

    async def connect_one(_url, _token, _agent):
        nonlocal stop_requested
        stop_requested = True
        raise ConnectionError("connection closed during shutdown")

    completed = asyncio.run(run_matches(
        "wss://example.test/ws/ranked", "secret", object(),
        connect_one=connect_one,
        sleep=delays.append,
        stop_requested=lambda: stop_requested,
    ))

    assert completed == 0
    assert delays == []
