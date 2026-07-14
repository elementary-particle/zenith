import gc

import numpy as np

import riichi


def test_projection_is_cached_read_only_aligned_contiguous_and_logical():
    env = riichi.Env(4, master_seed=11, num_threads=2, privileged=True)
    transition = env.reset([3, 1, 2, 0])
    first = transition.as_numpy()
    second = transition.as_numpy()
    assert first is second
    assert first["state_environment_id"].tolist() == [0, 1, 2, 3]
    assert first["state_scores"].shape == (4, 4)
    assert first["state_decision_offsets"][-1] == len(first["decision_seat"])
    assert first["decision_action_offsets"][-1] == len(first["action_kind"])
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
