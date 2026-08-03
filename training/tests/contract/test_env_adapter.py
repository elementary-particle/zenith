import numpy as np
import riichi
from zenith_ppo.encoding.packing import encode_native_batch
from zenith_ppo.env.adapter import EnvAdapter, EnvBatch


def test_adapter_preserves_bindings_and_gap_free_histories():
    adapter = EnvAdapter(riichi.Env(2, master_seed=1, num_threads=1))
    batch = adapter.reset(np.arange(2, dtype=np.uint32))
    assert len(batch.bindings) == len(batch.action_spaces)
    assert all(adapter.histories.get(b.environment_id, b.episode_generation).next_sequence > 0 for b in batch.bindings)
    assert [b.environment_id for b in batch.bindings] == sorted(b.environment_id for b in batch.bindings)
    selected = adapter.select(batch, [0] * len(batch.action_spaces))
    next_batch = adapter.step(selected)
    assert next_batch.transition.transition_id > batch.transition.transition_id


def test_disjoint_native_results_merge_into_one_sorted_frame():
    env = riichi.Env(2, master_seed=3, num_threads=1)
    adapter = EnvAdapter(env)
    adapter.reset([0, 1])
    left = adapter.inspect([0])
    right = adapter.inspect([1])

    merged = EnvBatch.merge(right, left)

    assert [int(state.environment_id) for state in merged.transition.states] == [0, 1]
    assert len(merged.action_spaces) == sum(
        len(batch.action_spaces) for batch in (left, right)
    )
    env.close()


def test_privileged_search_fork_preserves_public_actor_observation():
    env = riichi.Env(3, master_seed=41, num_threads=1, privileged=True)
    adapter = EnvAdapter(env)
    root = adapter.reset([0])
    root_encoded = encode_native_batch(root, adapter.histories)
    branches = adapter.fork_privileged_wall(0, ((1, 13), (2, 13)))
    branch_encoded = encode_native_batch(branches, adapter.histories)

    assert len(root_encoded) == 1
    assert len(branch_encoded) == 2
    for row in branch_encoded:
        np.testing.assert_array_equal(
            row.token_factors, root_encoded[0].token_factors
        )
        np.testing.assert_array_equal(
            row.token_numeric, root_encoded[0].token_numeric
        )
        np.testing.assert_array_equal(
            row.action_factors, root_encoded[0].action_factors
        )
    env.close()


def test_exact_search_fork_clones_public_history_at_later_node():
    env = riichi.Env(4, master_seed=47, num_threads=1, privileged=True)
    adapter = EnvAdapter(env)
    adapter.reset([0])
    sampled = adapter.fork_privileged_wall(0, ((1, 17),))
    source_encoded = encode_native_batch(sampled, adapter.histories)
    branches = adapter.fork_search_state(1, (2, 3))
    branch_encoded = encode_native_batch(branches, adapter.histories)

    assert len(source_encoded) == 1
    assert len(branch_encoded) == 2
    for row in branch_encoded:
        np.testing.assert_array_equal(
            row.token_factors, source_encoded[0].token_factors
        )
        np.testing.assert_array_equal(
            row.token_numeric, source_encoded[0].token_numeric
        )
        np.testing.assert_array_equal(
            row.action_factors, source_encoded[0].action_factors
        )
    env.close()


def test_public_information_fork_preserves_observer_actor_input():
    env = riichi.Env(3, master_seed=59, num_threads=1, privileged=True)
    adapter = EnvAdapter(env)
    root = adapter.reset([0])
    root_encoded = encode_native_batch(root, adapter.histories)
    observer = int(root_encoded[0].binding.seat)
    branches = adapter.fork_public_information(
        0, observer, ((1, 19), (2, 20))
    )
    branch_encoded = encode_native_batch(branches, adapter.histories)

    assert len(root_encoded) == 1
    assert len(branch_encoded) == 2
    for row in branch_encoded:
        np.testing.assert_array_equal(
            row.token_factors, root_encoded[0].token_factors
        )
        np.testing.assert_array_equal(
            row.token_numeric, root_encoded[0].token_numeric
        )
        np.testing.assert_array_equal(
            row.action_factors, root_encoded[0].action_factors
        )
    env.close()
