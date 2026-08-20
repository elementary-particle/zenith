import numpy as np
import pytest
import torch

import riichi
from zenith_ppo.capabilities import configure
from zenith_ppo.config import load
from zenith_ppo.model.factory import build_actor_critic
from zenith_ppo.rollout.native import NativeInferenceRunner


def _model(seed):
    configure("cpu-smoke")
    config = load("training/configs/smoke.toml")
    model_config = dict(config.values["model"])
    model_config["context_tokens"] = config.values["encoding"]["context_tokens"]
    torch.manual_seed(seed)
    return build_actor_critic(model_config), model_config["context_tokens"]


def _collect_one(seed=23, gae_lambda=1.0):
    model, context_tokens = _model(seed)
    engine = riichi.RolloutEngine(
        1, master_seed=seed, num_threads=1,
        context_tokens=context_tokens, token_budget=16_384,
    )
    matches = engine.reset_chunk(1)
    engine.register_lineups(matches, [(0, 0, 0, 0)], [0b1111])
    runner = NativeInferenceRunner(
        {0: model}, backend="eager",
        generator=torch.Generator().manual_seed(seed),
        gae_lambda=gae_lambda,
    )
    chunk = runner.run_chunk(engine)
    runner.prepare_training_chunk(chunk)
    return engine, model, chunk


def test_native_rollout_stores_replayable_behavior_log_probabilities():
    _, model, chunk = _collect_one()
    columns = chunk.columns()
    action_counts = np.diff(columns["action_offsets"])
    multi_action = np.flatnonzero(action_counts > 1)
    assert len(multi_action)
    assert np.count_nonzero(columns["old_logp"][multi_action] < -1e-6) \
        > len(multi_action) / 2

    replayed = []
    stored = []
    with torch.inference_mode():
        for batch in chunk.actor_minibatches(
            0, 16_384, max_padding_fraction=0.5, backend="eager"
        ):
            assert np.asarray(batch["raw_advantages"]).shape == np.asarray(
                batch["advantages"]
            ).shape
            assert np.isfinite(np.asarray(batch["raw_advantages"])).all()
            inputs = {
                name: value if name == "backend" else torch.as_tensor(value)
                for name, value in batch["model_inputs"].items()
            }
            output = model.forward_actor(**inputs, compute_entropy=False)
            selected = torch.as_tensor(batch["selected"], dtype=torch.long)
            replayed.extend(output.log_probabilities[selected].tolist())
            stored.extend(np.asarray(batch["old_logp"]).tolist())
    np.testing.assert_allclose(replayed, stored, atol=2e-5, rtol=2e-5)


def test_complete_match_keeps_rollout_critic_value_and_native_targets():
    from zenith_ppo.ppo.current_kyoku import RANK_UTILITIES

    _, _, chunk = _collect_one()
    columns = chunk.columns()
    eligible = np.asarray(columns["eligibility"], dtype=bool)
    boundaries = np.asarray(columns["rank_boundary_supervision"], dtype=bool)
    match_boundaries = np.asarray(chunk.as_numpy()["match_boundary"], dtype=bool)
    assert chunk.match_completions == 1
    assert int(columns["terminal_completed_kyoku"][0]) > 0
    assert eligible.any() and match_boundaries[eligible].any()
    assert np.all(np.asarray(columns["terminal_placements"])[eligible] >= 0)
    assert np.isfinite(np.asarray(columns["advantages"])[eligible]).all()
    assert np.isfinite(np.asarray(columns["normalized_advantages"])[eligible]).all()
    state_values = np.asarray(columns["old_state_values"])
    value_targets = np.asarray(columns["value_targets"])
    assert np.isfinite(state_values[eligible]).all()
    assert np.isfinite(value_targets[eligible]).all()
    np.testing.assert_allclose(
        np.asarray(columns["advantages"])[eligible],
        value_targets[eligible] - state_values[eligible],
        atol=1e-6,
    )
    predictions = np.asarray(columns["old_boundary_values"])
    placements = np.asarray(columns["terminal_placements"])
    terminal_rows = eligible & match_boundaries
    assert np.any(
        np.abs(predictions[terminal_rows] - RANK_UTILITIES[placements[terminal_rows]])
        > 1e-5
    )
    assert len(chunk.boundary_batch()["boundary_group_ids"]) == len(np.unique(
        np.asarray(columns["boundary_group_ids"])[eligible]
    ))
    assert boundaries.sum() == int(columns["terminal_completed_kyoku"][0])


def test_lambda_one_recovers_boundary_return_and_lower_lambda_changes_trace():
    _, _, boundary = _collect_one(seed=37, gae_lambda=1.0)
    _, _, discounted = _collect_one(seed=37, gae_lambda=0.9)
    boundary_columns = boundary.columns()
    discounted_columns = discounted.columns()
    eligible = np.asarray(boundary_columns["eligibility"], dtype=bool)
    np.testing.assert_allclose(
        np.asarray(boundary_columns["advantages"])[eligible],
        (
            np.asarray(boundary_columns["value_targets"])
            - np.asarray(boundary_columns["old_state_values"])
        )[eligible],
        atol=1e-6,
    )
    assert np.max(np.abs(
        np.asarray(boundary_columns["advantages"])[eligible]
        - np.asarray(discounted_columns["advantages"])[eligible]
    )) > 1e-5


def test_native_engine_reuses_slot_with_new_match_generation():
    engine, _, first = _collect_one(seed=29)
    first_generation = int(first.columns()["terminal_episode_generations"][0])
    matches = engine.reset_chunk(1)
    engine.register_lineups(matches, [(9, 9, 9, 9)], [0], bot_policy_slots=[9])
    second = engine.take_chunk()
    second_generation = int(second.columns()["terminal_episode_generations"][0])
    assert second_generation == first_generation + 1


def test_native_engine_rejects_reset_of_incomplete_chunk():
    engine = riichi.RolloutEngine(
        1, master_seed=31, num_threads=1,
        context_tokens=2048, token_budget=4096,
    )
    matches = engine.reset_chunk(1)
    engine.register_lineups(matches, [(0, 0, 0, 0)], [15])
    with pytest.raises(ValueError, match="incomplete rollout chunk"):
        engine.reset_chunk(1)
