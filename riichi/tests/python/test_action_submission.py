import pytest

import riichi


def snapshot_bytes(env, ids):
    return {key: bytes(value) for key, value in env.snapshot(ids).items()}


def first_actions(transition):
    return [
        space.candidates[0].select()
        for state in transition.states
        for space in state.action_spaces
    ]


def test_stale_duplicate_incomplete_and_cross_frame_sets_are_atomic():
    env = riichi.Env(2, master_seed=17, num_threads=2)
    initial = env.reset([0, 1])
    old = first_actions(initial)
    reaction = env.step(old)
    before = snapshot_bytes(env, [0, 1])

    duplicate = reaction.states[0].action_spaces[0].candidates[0].select()
    with pytest.raises(ValueError, match="DuplicateSeat|duplicate"):
        env.step([duplicate, duplicate])
    assert snapshot_bytes(env, [0, 1]) == before

    current_env1 = [space.candidates[0].select() for space in reaction.states[1].action_spaces]
    with pytest.raises(ValueError, match="StaleFrame|mismatch"):
        env.step([old[0], *current_env1])
    assert snapshot_bytes(env, [0, 1]) == before


def test_mixed_queryable_reaction_frame_requires_every_model_seat():
    env = riichi.Env(16, master_seed=1, num_threads=1)
    queue = [env.reset(list(range(16)))]
    mixed = None
    for _ in range(128):
        transition = queue.pop(0)
        mixed = next((state for state in transition.states
                      if len(state.action_spaces) >= 2), None)
        if mixed is not None:
            break
        actions = first_actions(transition)
        automatic = [
            state.environment_id for state in transition.states
            if not state.action_spaces and int(state.lifecycle) != 3
        ]
        if actions:
            queue.append(env.step(actions))
        if automatic:
            queue.append(env.advance(automatic))
    assert mixed is not None
    before = snapshot_bytes(env, [mixed.environment_id])
    with pytest.raises(ValueError, match="IncompleteActionSet|incomplete"):
        env.step([mixed.action_spaces[0].candidates[0].select()])
    assert snapshot_bytes(env, [mixed.environment_id]) == before


def test_out_of_range_environment_action_fails_before_mutation():
    env = riichi.Env(2, master_seed=23, num_threads=1)
    env.reset([0, 1])
    before = snapshot_bytes(env, [0, 1])

    source = riichi.Env(3, master_seed=23, num_threads=1)
    foreign = source.reset([2]).states[0].action_spaces[0].candidates[0].select()
    with pytest.raises(ValueError, match="out of range"):
        env.step([foreign])
    assert snapshot_bytes(env, [0, 1]) == before
    source.close()
    env.close()
