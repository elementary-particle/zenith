import pytest

import riichi


def test_native_values_are_immutable_and_survive_later_calls_and_env_drop():
    env = riichi.Env(1, master_seed=7, num_threads=1, privileged=True)
    transition = env.reset([0])
    state = transition.states[0]
    event = transition.events[0]
    decision = state.decisions[0]
    action = decision.actions[0]
    before = (state.scores, event.kind_name, action.frame_id, action.tiles)
    with pytest.raises(AttributeError):
        state.frame_id = 99
    with pytest.raises(AttributeError):
        action.action_index = 99
    env.step([action])
    env.close()
    del env
    assert (state.scores, event.kind_name, action.frame_id, action.tiles) == before


def test_action_kind_and_component_types_are_native():
    env = riichi.Env(1, master_seed=3, num_threads=1)
    transition = env.reset([0])
    decision = transition.states[0].decisions[0]
    assert isinstance(decision, riichi.Decision)
    assert isinstance(decision.actions[0], riichi.Action)
    assert isinstance(decision.actions[0].kind, riichi.ActionKind)
    assert isinstance(transition.events[0], riichi.Event)


def test_every_projected_decision_retains_its_frame_local_actions():
    env = riichi.Env(32, master_seed=29, num_threads=4)
    transition = env.reset(list(range(32)))

    for _ in range(256):
        decisions = [
            decision
            for state in transition.states
            for decision in state.decisions
        ]
        assert decisions
        for decision in decisions:
            assert decision.actions
            assert [action.action_index for action in decision.actions] == list(
                range(len(decision.actions))
            )
            assert all(action.frame_id == decision.frame_id for action in decision.actions)
            assert all(action.seat == decision.seat for action in decision.actions)

        transition = env.step([decision.actions[0] for decision in decisions])
        completed = [
            state.environment_id
            for state in transition.states
            if state.lifecycle == 3
        ]
        if completed:
            restarted = env.reset(completed)
            live_states = [state for state in transition.states if state.lifecycle != 3]
            transition_states = (*live_states, *restarted.states)
            decisions = [
                decision
                for state in transition_states
                for decision in state.decisions
            ]
            transition = env.inspect(
                sorted({decision.environment_id for decision in decisions})
            )

    env.close()


def test_strategic_action_branches_never_return_an_active_state_without_a_decision():
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
    transition = env.reset(range(32))
    pending = set(range(32))

    for _ in range(10_000):
        actions = []
        for state in transition.states:
            if int(state.lifecycle) == 3:
                pending.discard(int(state.environment_id))
                continue
            assert state.decisions, (
                state.environment_id,
                state.episode_generation,
                state.lifecycle,
                state.phase,
                state.frame_id,
            )
            for decision in state.decisions:
                actions.append(min(
                    decision.actions,
                    key=lambda action: (priority[int(action.kind)], action.action_index),
                ))
        if not pending:
            break
        assert actions
        transition = env.step(actions)
    else:
        pytest.fail(f"strategic native matches did not complete: {sorted(pending)}")

    env.close()
