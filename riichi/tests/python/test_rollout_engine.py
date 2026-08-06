import numpy as np
import pytest

import riichi


def _engine(seed=7, *, inference_only=False):
    return riichi.RolloutEngine(
        2,
        master_seed=seed,
        num_threads=1,
        context_tokens=2048,
        token_budget=4096,
        inference_only=inference_only,
    )


def test_request_is_pre_padded_and_submission_is_strict():
    engine = _engine()
    matches = engine.reset_chunk(2)
    engine.register_lineups(matches, [(0, 0, 0, 0)] * 2, [15, 15])

    request = engine.next_request()
    assert request.row_count > 0
    assert request.token_factors.shape == (
        request.row_count, request.sequence_bucket, 10,
    )
    assert request.token_numeric.shape[-1] == 8
    assert request.action_factors.shape == (
        request.row_count, request.action_bucket, 15,
    )
    assert np.all(request.lengths <= request.sequence_bucket)
    assert np.all(request.action_lengths <= request.action_bucket)
    assert not request.token_factors.flags.writeable

    with pytest.raises(ValueError, match="stale inference request"):
        engine.submit(
            request.request_id + 1,
            [0] * request.row_count,
            [0.0] * request.row_count,
        )
    with pytest.raises(ValueError, match="invalid for request row"):
        engine.submit(
            request.request_id,
            [999] + [0] * (request.row_count - 1),
            [0.0] * request.row_count,
        )
    engine.submit(
        request.request_id,
        [0] * request.row_count,
        [0.0] * request.row_count,
    )


def test_native_bot_rollout_is_reproducible_and_columnar():
    chunks = []
    for _ in range(2):
        engine = _engine(seed=31)
        matches = engine.reset_chunk(2)
        engine.register_lineups(
            matches,
            [(9, 9, 9, 9)] * 2,
            [0, 0],
            bot_policy_slots=[9],
        )
        assert engine.complete
        chunk = engine.take_chunk()
        assert chunk.match_completions == 2
        arrays = chunk.as_numpy()
        assert arrays["token_offsets"].shape == (chunk.row_count + 1,)
        assert arrays["action_offsets"].shape == (chunk.row_count + 1,)
        assert arrays["terminal_scores"].shape == (2, 4)
        assert arrays["terminal_ranks"].shape == (2, 4)
        assert chunk.row_count == 0
        chunks.append(arrays)

    for name in chunks[0]:
        np.testing.assert_array_equal(chunks[0][name], chunks[1][name])


def test_inference_only_mode_does_not_record_trajectory_rows():
    engine = _engine(inference_only=True)
    matches = engine.reset_chunk(1)
    engine.register_lineups(
        matches, [(7, 7, 7, 7)], [0], bot_policy_slots=[7]
    )
    chunk = engine.take_chunk()
    assert chunk.row_count == 0
    assert chunk.match_completions == 1


def test_native_boundary_groups_and_advantages_are_bulk_materialized():
    engine = _engine(seed=9)
    matches = engine.reset_chunk(1)
    engine.register_lineups(
        matches,
        [(0, 9, 9, 9)],
        [1],
        bot_policy_slots=[9],
    )
    while not engine.complete:
        request = engine.next_request()
        engine.submit(
            request.request_id,
            [0] * request.row_count,
            [0.0] * request.row_count,
        )

    chunk = engine.take_chunk()
    boundary = chunk.boundary_batch()
    group_ids = boundary["boundary_group_ids"].tolist()
    assert len(group_ids) < chunk.row_count
    chunk.set_boundary_values(group_ids, [0.0] * len(group_ids))
    chunk.finish_targets()
    arrays = chunk.as_numpy()
    eligible = arrays["eligibility"].astype(bool)
    assert np.isfinite(arrays["advantages"][eligible]).all()
    assert np.isfinite(arrays["normalized_advantages"][eligible]).all()
    assert arrays["rank_boundary_supervision"].sum() == len(group_ids)
    assert set(arrays["terminal_placements"]) <= {0, 1, 2, 3}
