import numpy as np
import riichi
from zenith_ppo.env.adapter import EnvAdapter


def test_adapter_preserves_bindings_and_gap_free_histories():
    adapter = EnvAdapter(riichi.Env(2, master_seed=1, num_threads=1))
    batch = adapter.reset(np.arange(2, dtype=np.uint32))
    assert len(batch.bindings) == len(batch.decisions)
    assert all(adapter.histories.get(b.environment_id, b.episode_generation).next_sequence > 0 for b in batch.bindings)
    assert [b.environment_id for b in batch.bindings] == sorted(b.environment_id for b in batch.bindings)
    selected = adapter.select(batch, [0] * len(batch.decisions))
    next_batch = adapter.step(selected)
    assert next_batch.transition.transition_id > batch.transition.transition_id
