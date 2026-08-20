import json

import pytest

from zenith_ppo.akagi import AkagiProtocolError, AkagiSession, LiveMjaiState


START_GAME = {
    "type": "start_game",
    "names": ["east", "south", "west", "north"],
    "id": 0,
    "num_players": 4,
}
START_KYOKU = {
    "type": "start_kyoku",
    "bakaze": "E",
    "dora_marker": "9s",
    "kyoku": 1,
    "honba": 0,
    "kyotaku": 0,
    "oya": 0,
    "scores": [25_000] * 4,
    "tehais": [
        ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p"],
        ["?"] * 13,
        ["?"] * 13,
        ["?"] * 13,
    ],
}


class _Choose:
    def __init__(self, action_type):
        self.action_type = action_type
        self.pending_riichi_tile = None
        self.observed = []

    def reset(self):
        self.pending_riichi_tile = None
        self.observed.clear()

    def observe_event(self, event):
        self.observed.append(dict(event))

    def cancel_pending_action(self):
        self.pending_riichi_tile = None

    def act(self, observation):
        legal = observation.legal_actions()
        if self.pending_riichi_tile is not None:
            tile = self.pending_riichi_tile
            self.pending_riichi_tile = None
            return next(
                action for action in legal
                if int(action.action_type) == 0 and int(action.tile) == tile
            )
        selected = next(
            action for action in legal
            if int(action.action_type) == self.action_type
        )
        if self.action_type == 5:
            from riichienv import check_riichi_candidates

            self.pending_riichi_tile = int(check_riichi_candidates(observation.hand)[0])
        return selected


class _ChooseWithDiagnostics(_Choose):
    def act(self, observation):
        selected = super().act(observation)
        action = json.loads(selected.to_mjai())
        self.last_diagnostics = {
            "schema_version": 1,
            "source": "policy",
            "temperature": 0.0,
            "legal_action_count": 3,
            "selected_rank": 1,
            "selected_action": {
                "rank": 1,
                "action": action,
                "policy_probability": 0.75,
                "selected": True,
            },
            "top_actions": [{
                "rank": 1,
                "action": action,
                "policy_probability": 0.75,
                "selected": True,
                "prospects": {
                    "hand_outcome_probabilities": {
                        "draw": 0.2,
                        "win": 0.4,
                        "deal_in": 0.1,
                        "other_win": 0.3,
                    },
                    "expected_score_delta": 1200.0,
                },
            }],
        }
        return selected


def test_live_state_builds_riichienv_observation_for_own_draw():
    state = LiveMjaiState()
    for event in (START_GAME, START_KYOKU, {"type": "tsumo", "actor": 0, "pai": "5p"}):
        state.consume(event)

    observation = state.observation({"type": "tsumo", "actor": 0, "pai": "5p"})

    assert observation.player_id == 0
    assert len(observation.hand) == 14
    assert observation.drawn_tile is not None
    assert {int(action.action_type) for action in observation.legal_actions()} >= {0}
    assert [
        (json.loads(value) if isinstance(value, str) else value)["type"]
        for value in observation.events
    ] == [
        "start_game", "start_kyoku", "tsumo",
    ]


def test_hidden_own_draw_skips_inference_and_recovers_on_tsumogiri():
    session = AkagiSession(_Choose(0))
    session.react([START_GAME])

    assert session.react([
        START_KYOKU,
        {"type": "tsumo", "actor": 0, "pai": "?"},
    ]) == {"type": "none"}
    assert session.state.own_draw_hidden
    assert session.state.unknown_own_tiles == 1
    assert len(session.state.hand) == 13

    assert session.react([{
        "type": "dahai", "actor": 0, "pai": "5p", "tsumogiri": True,
    }]) == {"type": "none"}
    assert not session.state.own_draw_hidden
    assert session.state.unknown_own_tiles == 0
    assert len(session.state.hand) == 13
    assert session.state.discards[0][-1] // 4 == 13


def test_hidden_own_draw_with_tedashi_tracks_uncertainty_without_desync():
    session = AkagiSession(_Choose(0))
    session.react([
        START_GAME,
        START_KYOKU,
        {"type": "tsumo", "actor": 0, "pai": "?"},
    ])

    assert session.react([{
        "type": "dahai", "actor": 0, "pai": "1m", "tsumogiri": False,
    }]) == {"type": "none"}
    assert not session.desynchronized
    assert session.state.unknown_own_tiles == 1
    assert len(session.state.hand) == 12

    # Known future draws/discards remain trackable but inference stays off
    # while the concealed hand contains an unresolved tile.
    assert session.react([
        {"type": "tsumo", "actor": 1, "pai": "?"},
        {"type": "dahai", "actor": 1, "pai": "9m", "tsumogiri": True},
    ]) == {"type": "none"}
    assert session.state.unknown_own_tiles == 1


def test_hidden_own_draw_recovers_when_tsumogiri_flag_is_wrong_but_tile_is_new():
    session = AkagiSession(_Choose(0))
    session.react([
        START_GAME,
        START_KYOKU,
        {"type": "tsumo", "actor": 0, "pai": "?"},
    ])

    assert session.react([{
        "type": "dahai", "actor": 0, "pai": "9s", "tsumogiri": False,
    }]) == {"type": "none"}
    assert not session.state.own_draw_hidden
    assert session.state.unknown_own_tiles == 0
    assert len(session.state.hand) == 13


def test_akagi_session_returns_discard_and_tracks_echoed_hand():
    session = AkagiSession(_Choose(0))
    assert session.react([START_GAME]) == {"type": "none"}
    response = session.react([
        START_KYOKU,
        {"type": "tsumo", "actor": 0, "pai": "5p"},
    ])

    assert response == {
        "type": "dahai", "actor": 0, "pai": "1m", "tsumogiri": False,
    }
    assert len(session.state.hand) == 14
    assert session.react([response]) == {"type": "none"}
    assert len(session.state.hand) == 13


def test_akagi_session_renders_policy_diagnostics_in_meta_show():
    session = AkagiSession(_ChooseWithDiagnostics(0))
    session.react([START_GAME])

    response = session.react([
        START_KYOKU,
        {"type": "tsumo", "actor": 0, "pai": "5p"},
    ])

    assert response["meta"]["confidence"] == pytest.approx(0.75)
    assert response["meta"]["policy"]["selected_rank"] == 1
    assert response["meta"]["state"]["live_wall_remaining"] == 69
    assert response["meta"]["show"] == {
        "title": "Zenith policy — top actions",
        "items": [{
            "label": "#1 Discard 1m",
            "value": "75.0%",
            "pais": ["1m"],
            "note": "selected · win 40.0% · deal-in 10.0% · projected delta +1200",
            "color": "#22c55e",
        }],
    }


def test_akagi_session_adds_call_target():
    start = dict(START_KYOKU)
    start["tehais"] = [
        ["5m", "5mr", "1m", "2m", "3m", "1p", "2p", "3p", "1s", "2s", "3s", "E", "E"],
        ["?"] * 13,
        ["?"] * 13,
        ["?"] * 13,
    ]
    session = AkagiSession(_Choose(2))
    session.react([START_GAME])
    session.react([start])

    response = session.react([{
        "type": "dahai", "actor": 1, "pai": "5m", "tsumogiri": False,
    }])

    assert response["type"] == "pon"
    assert response["actor"] == 0
    assert response["target"] == 1
    assert sorted(response["consumed"]) == ["5m", "5mr"]


def test_unconfirmed_riichi_recommendation_is_not_applied_to_later_decision():
    agent = _Choose(7)
    session = AkagiSession(agent)
    session.react([START_GAME, START_KYOKU])
    session.state.consume({"type": "tsumo", "actor": 0, "pai": "5p"})
    agent.pending_riichi_tile = session.state.hand[-1]

    response = session.react([
        {"type": "dahai", "actor": 0, "pai": "5p", "tsumogiri": True},
        {"type": "tsumo", "actor": 1, "pai": "?"},
        {"type": "dahai", "actor": 1, "pai": "9m", "tsumogiri": True},
    ])

    assert response == {"type": "none"}
    assert agent.pending_riichi_tile is None


def test_akagi_riichi_response_is_combined_and_clears_forced_followup():
    agent = _Choose(5)
    session = AkagiSession(agent)
    session.react([START_GAME])

    response = session.react([
        START_KYOKU,
        {"type": "tsumo", "actor": 0, "pai": "5p"},
    ])

    assert response["type"] == "reach"
    assert response["pai"]
    assert agent.pending_riichi_tile is None


def test_desynchronized_hand_recovers_at_next_start_kyoku():
    session = AkagiSession(_Choose(0))
    session.react([START_GAME, START_KYOKU])
    session.desynchronize()

    assert session.react([{
        "type": "dahai", "actor": 1, "pai": "1m", "tsumogiri": False,
    }]) == {"type": "none"}
    assert session.desynchronized

    assert session.react([START_KYOKU]) == {"type": "none"}
    assert not session.desynchronized
    assert len(session.state.hand) == 13


def test_non_four_player_game_is_rejected():
    state = LiveMjaiState()
    with pytest.raises(AkagiProtocolError, match="four-player"):
        state.consume({**START_GAME, "num_players": 3})


def test_live_state_legal_actions_match_riichienv_through_calls():
    import riichienv

    def signature(action):
        value = json.loads(action.to_mjai())
        return (
            value.get("type"), value.get("pai"),
            tuple(sorted(value.get("consumed", ()))),
        )

    def aggressive_action(observation):
        legal = observation.legal_actions()
        for kinds in ((4, 6), (3, 2, 1), (8, 9), (0,), (7,)):
            selected = next(
                (action for action in legal if int(action.action_type) in kinds),
                None,
            )
            if selected is not None:
                return selected
        return legal[0]

    env = riichienv.RiichiEnv(seed=2)
    observations = env.reset()
    state = LiveMjaiState()
    event_offset = 0
    saw_own_call = False
    for _ in range(200):
        event_log = list(env.mjai_log)
        for raw in event_log[event_offset:]:
            event = json.loads(raw) if isinstance(raw, str) else dict(raw)
            if event["type"] == "start_game":
                event.update(id=0, num_players=4, names=["a", "b", "c", "d"])
            saw_own_call |= event["type"] in {"chi", "pon"} and event.get("actor") == 0
            state.consume(event)
        event_offset = len(event_log)
        if state.events[-1]["type"] == "end_game":
            break
        if 0 in observations:
            reconstructed = state.observation(state.events[-1])
            assert reconstructed is not None
            assert set(map(signature, reconstructed.legal_actions())) == set(
                map(signature, observations[0].legal_actions())
            )
        if env.is_done:
            break
        observations = env.step({
            seat: aggressive_action(observation)
            for seat, observation in observations.items()
        })
    assert saw_own_call


def test_checkpoint_agent_session_state_is_independent():
    from zenith_ppo.mjai import CheckpointAgent

    model = object()
    parent = CheckpointAgent(model, temperature=0.7, legacy_chi_workaround=False)
    child = parent.new_session()
    child.pending_riichi_tile = 12

    assert child.model is parent.model
    assert child.temperature == parent.temperature
    assert child.legacy_chi_workaround is False
    assert parent.pending_riichi_tile is None
