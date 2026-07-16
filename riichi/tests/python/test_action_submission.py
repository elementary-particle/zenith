import pytest

import riichi


def snapshot_bytes(env, ids):
    return {key: bytes(value) for key, value in env.snapshot(ids).items()}


def first_actions(transition):
    return [decision.actions[0] for state in transition.states for decision in state.decisions]


def test_stale_duplicate_incomplete_and_cross_frame_sets_are_atomic():
    env = riichi.Env(2, master_seed=17, num_threads=2)
    initial = env.reset([0, 1])
    old = first_actions(initial)
    reaction = env.step(old)
    before = snapshot_bytes(env, [0, 1])

    duplicate = reaction.states[0].decisions[0].actions[0]
    with pytest.raises(ValueError, match="DuplicateSeat|duplicate"):
        env.step([duplicate, duplicate])
    assert snapshot_bytes(env, [0, 1]) == before

    current_env1 = [d.actions[0] for d in reaction.states[1].decisions]
    with pytest.raises(ValueError, match="StaleFrame|mismatch"):
        env.step([old[0], *current_env1])
    assert snapshot_bytes(env, [0, 1]) == before


def test_mixed_queryable_reaction_frame_requires_every_model_seat():
    env = riichi.Env(16, master_seed=1, num_threads=1)
    transition = env.reset(list(range(16)))
    mixed = None
    for _ in range(32):
        mixed = next((state for state in transition.states
                      if len(state.decisions) >= 2), None)
        if mixed is not None:
            break
        transition = env.step(first_actions(transition))
    assert mixed is not None
    before = snapshot_bytes(env, [mixed.environment_id])
    with pytest.raises(ValueError, match="IncompleteActionSet|incomplete"):
        env.step([mixed.decisions[0].actions[0]])
    assert snapshot_bytes(env, [mixed.environment_id]) == before

def test_ineligible_and_foreign_environment_actions_fail_before_mutation():
    env = riichi.Env(2, master_seed=23, num_threads=1)
    transition = env.reset([0, 1])
    before = snapshot_bytes(env, [0, 1])
    action = transition.states[0].decisions[0].actions[0]
    with pytest.raises(ValueError):
        env.step([action, action])
    assert snapshot_bytes(env, [0, 1]) == before
