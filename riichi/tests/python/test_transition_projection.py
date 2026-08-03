import gc

import numpy as np

import riichi


def test_native_build_profile_is_exposed():
    assert riichi.NATIVE_BUILD_PROFILE in {"debug", "release"}


def test_projection_is_cached_read_only_aligned_contiguous_and_logical():
    env = riichi.Env(4, master_seed=11, num_threads=2, privileged=True)
    transition = env.reset([3, 1, 2, 0])
    first = transition.as_numpy()
    second = transition.as_numpy()
    assert first is second
    assert first["state_environment_id"].tolist() == [0, 1, 2, 3]
    assert first["state_scores"].shape == (4, 4)
    assert first["hidden_live_wall_counts"].shape == (4, 34)
    np.testing.assert_array_equal(
        first["hidden_live_wall_counts"].sum(axis=1),
        first["state_live_wall_remaining"],
    )
    for index, state in enumerate(transition.states):
        assert tuple(first["hidden_live_wall_counts"][index]) == tuple(
            state.hidden.live_wall_counts
        )
    assert first["state_action_space_offsets"][-1] == len(first["action_space_seat"])
    assert first["action_space_candidate_offsets"][-1] == len(first["candidate_kind"])
    for value in first.values():
        assert isinstance(value, np.ndarray)
        assert value.flags.aligned and value.flags.c_contiguous
        assert not value.flags.writeable


def test_projection_arrays_outlive_transition_and_env():
    env = riichi.Env(1, master_seed=5, num_threads=1, privileged=True)
    transition = env.reset([0])
    scores = transition.as_numpy()["state_scores"]
    expected = scores.copy()
    del transition
    env.close()
    del env
    gc.collect()
    np.testing.assert_array_equal(scores, expected)


def test_only_genuine_choices_are_projected_and_native_metrics_split_resolution():
    env = riichi.Env(1, master_seed=11, num_threads=1)
    transition = env.reset([0])
    assert all(len(decision.candidates) >= 2 for state in transition.states
               for decision in state.action_spaces)
    env.metrics(reset=True)
    transition = env.step([
        space.candidates[0].select()
        for state in transition.states
        for space in state.action_spaces
    ])
    assert all(len(decision.candidates) >= 2 for state in transition.states
               for decision in state.action_spaces)
    metrics = env.metrics()
    assert metrics["model_queries"] == 1
    assert metrics["rust_resolved_decisions"] >= 3
