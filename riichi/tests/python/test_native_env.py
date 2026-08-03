import pytest

import riichi


def test_native_values_are_immutable_and_survive_later_calls_and_env_drop():
    env = riichi.Env(1, master_seed=7, num_threads=1, privileged=True)
    transition = env.reset([0])
    state = transition.states[0]
    event = transition.events[0]
    decision = state.action_spaces[0]
    action = decision.candidates[0]
    selection = action.select()
    before = (state.scores, event.kind_name, selection.frame_id, action.tiles)
    with pytest.raises(AttributeError):
        state.frame_id = 99
    with pytest.raises(AttributeError):
        action.candidate_index = 99
    env.step([selection])
    env.close()
    del env
    assert (state.scores, event.kind_name, selection.frame_id, action.tiles) == before


def test_action_kind_and_component_types_are_native():
    env = riichi.Env(1, master_seed=3, num_threads=1)
    transition = env.reset([0])
    decision = transition.states[0].action_spaces[0]
    assert isinstance(decision, riichi.ActionSpace)
    assert isinstance(decision.candidates[0], riichi.ActionCandidate)
    assert isinstance(decision.candidates[0].select(), riichi.ActionSelection)
    assert isinstance(decision.candidates[0].kind, riichi.ActionKind)
    assert isinstance(transition.events[0], riichi.Event)


def test_removed_action_api_has_no_compatibility_aliases():
    env = riichi.Env(1, master_seed=3, num_threads=1)
    transition = env.reset([0])
    state = transition.states[0]
    space = state.action_spaces[0]

    assert not hasattr(riichi, "Action")
    assert not hasattr(riichi, "Decision")
    assert not hasattr(state, "decisions")
    assert not hasattr(space, "actions")
    assert not hasattr(transition, "applied_actions")
    env.close()


def test_privileged_search_fork_pairs_wall_particles_and_rebinds_actions():
    env = riichi.Env(4, master_seed=37, num_threads=2, privileged=True)
    env.reset([0])
    transition = env.fork_privileged_wall(0, [(1, 5), (2, 5), (3, 6)])
    states = {int(state.environment_id): state for state in transition.states}

    assert states[1].hidden.wall == states[2].hidden.wall
    assert states[1].hidden.wall != states[3].hidden.wall
    assert not transition.events
    for environment_id, state in states.items():
        assert state.action_spaces
        assert all(
            candidate.select().environment_id == environment_id
            for space in state.action_spaces
            for candidate in space.candidates
        )
    env.close()


def test_search_state_fork_preserves_an_existing_wall_particle():
    env = riichi.Env(4, master_seed=43, num_threads=2, privileged=True)
    env.reset([0])
    sampled = env.fork_privileged_wall(0, [(1, 7)]).states[0]
    transition = env.fork_search_state(1, [2, 3])

    assert not transition.events
    for state in transition.states:
        assert state.hidden.wall == sampled.hidden.wall
        assert state.action_spaces
        assert all(
            candidate.select().environment_id == state.environment_id
            for space in state.action_spaces
            for candidate in space.candidates
        )
    env.close()


def test_public_information_fork_pairs_hidden_state_but_not_observer_hand():
    env = riichi.Env(4, master_seed=53, num_threads=2, privileged=True)
    root = env.reset([0]).states[0]
    observer = root.hidden.current_seat
    observer_hand = root.hidden.concealed_tile_ids[observer]
    transition = env.fork_public_information(
        0, observer, [(1, 11), (2, 11), (3, 12)]
    )
    states = {int(state.environment_id): state for state in transition.states}

    assert states[1].hidden.wall == states[2].hidden.wall
    assert states[1].hidden.concealed_tile_ids == states[2].hidden.concealed_tile_ids
    assert states[1].hidden.wall != states[3].hidden.wall
    for state in states.values():
        assert state.hidden.concealed_tile_ids[observer] == observer_hand
    env.close()


def test_every_projected_decision_retains_its_frame_local_actions():
    env = riichi.Env(32, master_seed=29, num_threads=4)
    queue = [env.reset(list(range(32)))]

    for _ in range(256):
        transition = queue.pop(0)
        decisions = [
            decision
            for state in transition.states
            for decision in state.action_spaces
        ]
        for decision in decisions:
            assert decision.candidates
            assert [action.candidate_index for action in decision.candidates] == list(
                range(len(decision.candidates))
            )
            assert all(action.select().frame_id == decision.frame_id for action in decision.candidates)
            assert all(action.select().seat == decision.seat for action in decision.candidates)

        automatic = [
            state.environment_id for state in transition.states
            if not state.action_spaces and int(state.lifecycle) != 3
        ]
        if decisions:
            queue.append(env.step([
                decision.candidates[0].select() for decision in decisions
            ]))
        if automatic:
            queue.append(env.advance(automatic))
        completed = [
            state.environment_id
            for state in transition.states
            if state.lifecycle == 3
        ]
        if completed:
            queue.append(env.reset(completed))

    env.close()


def test_strategic_action_branches_expose_and_advance_automatic_states():
    """Prefer wins, riichi, and kans to cover policy-dependent control flow."""
    priority = {
        int(riichi.ActionKind.Ron): 0,
        int(riichi.ActionKind.Tsumo): 0,
        int(riichi.ActionKind.RiichiDiscard): 1,
        int(riichi.ActionKind.ClosedKan): 2,
        int(riichi.ActionKind.AddedKan): 2,
        int(riichi.ActionKind.OpenKan): 3,
        int(riichi.ActionKind.Pon): 4,
        int(riichi.ActionKind.Chi): 5,
        int(riichi.ActionKind.AbortiveDeclaration): 6,
        int(riichi.ActionKind.Discard): 7,
        int(riichi.ActionKind.Pass): 8,
    }
    env = riichi.Env(32, master_seed=717171, num_threads=4, privileged=True)
    queue = [env.reset(range(32))]
    pending = set(range(32))
    saw_automatic = False

    for _ in range(100_000):
        transition = queue.pop(0)
        actions = []
        automatic = []
        for state in transition.states:
            if int(state.lifecycle) == 3:
                pending.discard(int(state.environment_id))
                continue
            if not state.action_spaces:
                saw_automatic = True
                automatic.append(int(state.environment_id))
                continue
            for decision in state.action_spaces:
                actions.append(min(
                    decision.candidates,
                    key=lambda action: (priority[int(action.kind)], action.candidate_index),
                ))
        if not pending:
            break
        if actions:
            queue.append(env.step([candidate.select() for candidate in actions]))
        if automatic:
            queue.append(env.advance(automatic))
        assert queue
    else:
        pytest.fail(f"strategic native matches did not complete: {sorted(pending)}")

    assert saw_automatic
    env.close()
