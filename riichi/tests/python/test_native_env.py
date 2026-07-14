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
