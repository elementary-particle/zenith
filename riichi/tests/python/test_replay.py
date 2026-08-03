import pytest

import riichi


def _fixed_hanchan(state):
    return riichi.ReplayHanchan(
        0,
        round_wind=state.round_wind,
        hand_number=state.hand_number,
        dealer=state.dealer,
        honba=state.honba,
        riichi_deposits=state.riichi_deposits,
        scores=state.scores,
        wall=state.hidden.wall,
    )


def _decision_signature(state):
    return tuple(
        (
            decision.seat,
            tuple(
                (int(action.kind), action.primary_tile_type, tuple(action.tiles))
                for action in decision.candidates
            ),
        )
        for decision in state.action_spaces
    )


def test_fixed_wall_load_reproduces_public_decision_and_snapshot_round_trip():
    native = riichi.Env(1, master_seed=73, num_threads=1, privileged=True)
    original = native.reset([0])
    original_state = original.states[0]
    draw = next(event for event in original.events if event.kind_name == "tsumo")

    replay = riichi.Env(1, master_seed=999, num_threads=1, privileged=True)
    hanchan = _fixed_hanchan(original_state)
    assert hanchan.wall == original_state.hidden.wall
    with pytest.raises(AttributeError):
        hanchan.dealer = 3
    loaded = replay.load_hanchan([hanchan])
    assert [event.kind_name for event in loaded.events] == ["start_game", "start_kyoku"]
    assert {16, 52, 88} <= set(original_state.hidden.wall)

    replay_draw = riichi.ReplayEvent(
        0, "tsumo", actor_seat=draw.actor_seat, tile=draw.args[0]
    )
    assert (replay_draw.actor_seat, replay_draw.tile) == (
        draw.actor_seat, draw.args[0]
    )
    with pytest.raises(AttributeError):
        replay_draw.tile = 0
    transition = replay.apply_events([replay_draw])
    replay_state = transition.states[0]
    assert _decision_signature(replay_state) == _decision_signature(original_state)
    assert replay_state.scores == original_state.scores
    assert replay_state.dora_indicators == original_state.dora_indicators
    assert replay_state.hidden.concealed_tile_ids == original_state.hidden.concealed_tile_ids

    snapshot = replay.snapshot([0])[0]
    restored = replay.restore({0: snapshot})
    assert _decision_signature(restored.states[0]) == _decision_signature(replay_state)
    assert replay.snapshot([0])[0] == snapshot
    native.close()
    replay.close()


def test_replay_applies_physical_discard_and_rejects_transactionally():
    source = riichi.Env(1, master_seed=91, num_threads=1, privileged=True)
    original = source.reset([0])
    state = original.states[0]
    draw = next(event for event in original.events if event.kind_name == "tsumo")
    discard = next(
        action for action in state.action_spaces[0].candidates
        if action.kind == riichi.ActionKind.Discard
    )
    discard_selection = discard.select()

    replay = riichi.Env(1, master_seed=0, num_threads=1, privileged=True)
    replay.load_hanchan([_fixed_hanchan(state)])
    replay.apply_events([
        riichi.ReplayEvent(0, "tsumo", actor_seat=draw.actor_seat, tile=draw.args[0])
    ])
    before = replay.snapshot([0])[0]
    concealed_types = {
        tile // 4 for tile in replay.inspect([0], privileged=True)
        .states[0].hidden.concealed_tile_ids[discard_selection.seat]
    }
    absent_type = next(tile_type for tile_type in range(34) if tile_type not in concealed_types)
    with pytest.raises(RuntimeError, match="not legal|not owned"):
        replay.apply_events([
            riichi.ReplayEvent(
                0, "dahai", actor_seat=discard_selection.seat, tile=absent_type * 4
            )
        ])
    assert replay.snapshot([0])[0] == before

    applied = replay.apply_events([
        riichi.ReplayEvent(
            0,
            "dahai",
            actor_seat=discard_selection.seat,
            tile=discard.tiles[0],
            tsumogiri=discard.tiles[0] == draw.args[0],
        )
    ])
    assert applied.applied_selections[0].frame_id == discard_selection.frame_id
    assert applied.applied_selections[0].candidate_index == discard.candidate_index
    arrays = applied.as_numpy()
    assert arrays["applied_candidate_index"].tolist() == [discard.candidate_index]
    assert arrays["applied_selection_seat"].tolist() == [discard_selection.seat]
    source.close()
    replay.close()


def test_omitted_reactions_are_returned_as_applied_passes():
    for seed in range(1, 128):
        source = riichi.Env(1, master_seed=seed, num_threads=1, privileged=True)
        initial = source.reset([0])
        state = initial.states[0]
        snapshot = source.snapshot([0])[0]
        candidate = None
        reaction = None
        for action in state.action_spaces[0].candidates:
            if action.kind != riichi.ActionKind.Discard:
                continue
            reaction = source.step([action.select()])
            if reaction.states[0].phase == 2 and reaction.states[0].action_spaces:
                candidate = action
                break
            source.restore({0: snapshot})
        if candidate is not None:
            break
        source.close()
    else:
        pytest.fail("no deterministic discard exposed a genuine reaction frame")

    passes = [
        next(action for action in decision.candidates if action.kind == riichi.ActionKind.Pass)
        for decision in reaction.states[0].action_spaces
    ]
    resolved = source.step([candidate.select() for candidate in passes])
    draw = next(event for event in resolved.events if event.kind_name == "tsumo")
    initial_draw = next(event for event in initial.events if event.kind_name == "tsumo")

    replay = riichi.Env(1, master_seed=0, num_threads=1, privileged=True)
    replay.load_hanchan([_fixed_hanchan(state)])
    replay.apply_events([
        riichi.ReplayEvent(
            0, "tsumo", actor_seat=initial_draw.actor_seat, tile=initial_draw.args[0]
        )
    ])
    replay.apply_events([
        riichi.ReplayEvent(
            0, "dahai", actor_seat=candidate.select().seat, tile=candidate.tiles[0],
            tsumogiri=candidate.tiles[0] == initial_draw.args[0],
        )
    ])
    inferred = replay.apply_events([
        riichi.ReplayEvent(0, "tsumo", actor_seat=draw.actor_seat, tile=draw.args[0])
    ])
    assert len(inferred.applied_selections) == len(passes)
    assert {
        (selection.seat, selection.candidate_index)
        for selection in inferred.applied_selections
    } == {
        (candidate.select().seat, candidate.candidate_index)
        for candidate in passes
    }
    source.close()
    replay.close()
