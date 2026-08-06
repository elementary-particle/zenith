import numpy as np
import riichi

from zenith_ppo.encoding.packing import encode_native_batch
from zenith_ppo.env.adapter import EnvAdapter


def _binding_tuple(binding):
    return (
        int(binding.environment_id),
        int(binding.episode_generation),
        int(binding.frame_id),
        int(binding.seat),
    )


def test_native_engine_matches_python_encoder_across_live_frames():
    seed = 17
    adapter = EnvAdapter(riichi.Env(1, master_seed=seed, num_threads=1))
    batch = adapter.reset([0])
    reference = {
        _binding_tuple(row.binding): row
        for row in encode_native_batch(batch, adapter.histories)
    }

    engine = riichi.RolloutEngine(
        1,
        master_seed=seed,
        num_threads=1,
        context_tokens=2048,
        token_budget=65536,
    )
    matches = engine.reset_chunk(1)
    engine.register_lineups(matches, [(0, 0, 0, 0)], [15])

    seen = set()
    decisions = 0
    while not engine.complete and decisions < 64:
        request = engine.next_request()
        assert request is not None
        for index in range(request.row_count):
            binding = (
                int(request.environment_ids[index]),
                int(request.episode_generations[index]),
                int(request.frame_ids[index]),
                int(request.seats[index]),
            )
            row = reference[binding]
            length = int(request.lengths[index])
            actions = int(request.action_lengths[index])
            np.testing.assert_array_equal(
                row.token_factors,
                request.token_factors[index, :length],
            )
            np.testing.assert_allclose(
                row.token_numeric,
                request.token_numeric[index, :length],
                rtol=1e-6,
                atol=1e-6,
            )
            np.testing.assert_array_equal(
                row.action_factors,
                request.action_factors[index, :actions],
            )
            np.testing.assert_allclose(
                row.rank_boundary_features,
                request.rank_boundary_features[index],
                rtol=1e-7,
                atol=1e-7,
            )
            assert row.actor_query_index == int(request.query_offsets[index])
            seen.add(binding)
        engine.submit(
            request.request_id,
            [0] * request.row_count,
            [0.0] * request.row_count,
        )
        decisions += request.row_count

        if len(seen) == len(reference):
            batch = adapter.step([
                row.native_candidates[0].select()
                for row in reference.values()
            ])
            while all(
                not state.action_spaces and int(state.lifecycle) not in (3, 4)
                for state in batch.transition.states
            ):
                batch = adapter.advance([0])
            reference = {
                _binding_tuple(row.binding): row
                for row in encode_native_batch(batch, adapter.histories)
            }
            seen.clear()

    assert decisions == 64
