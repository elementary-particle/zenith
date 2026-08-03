import pytest


def _collect_one(
    seed=23, *, target_matches=1, verify_policy_replay=False,
    diagnostic_dir=None,
):
    import riichi
    import torch
    from zenith_ppo.config import load
    from zenith_ppo.capabilities import configure
    from zenith_ppo.env.adapter import EnvAdapter
    from zenith_ppo.model.actor_critic import ActorCritic
    from zenith_ppo.rewards.curriculum import Curriculum
    from zenith_ppo.rollout.collector import Collector
    from zenith_ppo.seeds import SeedStreams

    configure("cpu-smoke")
    config = load("training/configs/smoke.toml")
    model_config = dict(config.values["model"])
    model_config["context_tokens"] = config.values["encoding"]["context_tokens"]
    torch.manual_seed(seed)
    model = ActorCritic(model_config)
    model.verify_rollout_policy_replay = verify_policy_replay
    env = riichi.Env(1, master_seed=seed, num_threads=1, privileged=True)
    adapter = EnvAdapter(env)
    streams = SeedStreams(seed)
    result = Collector(
        adapter, model, streams.torch_generator("action", "cpu"),
        diagnostic_dir=diagnostic_dir,
    ).collect(
        adapter.reset([0]), target_matches=target_matches,
        curriculum=Curriculum(config.values["curriculum"]).snapshot(0, 0),
        streams=streams, max_env_calls=4096,
    )
    env.close()
    return result


def test_batched_rollout_stores_each_rows_sampled_log_probability(tmp_path):
    result = _collect_one(
        verify_policy_replay=True, diagnostic_dir=tmp_path,
    )
    # The collector's replay invariant compares every stored behavior log
    # probability with a fresh forward pass of the unchanged policy.  Keep a
    # direct sanity check as well: legal multi-action rows are not initialized
    # placeholders.
    multi_action = [
        sample for sample in result.samples
        if len(sample.encoded.action_representatives) > 1
    ]
    assert multi_action
    assert sum(
        sample.old_log_probability < -1e-6 for sample in multi_action
    ) > len(multi_action) / 2


def test_complete_match_keeps_public_value_on_terminal_pre_action_frame():
    from zenith_ppo.ppo.current_kyoku import RANK_UTILITIES, compute

    result = _collect_one()
    assert result.match_completions == len(result.match_outcomes) == 1
    assert result.match_outcomes[0].completed_kyoku > 0
    learners = [sample for sample in result.samples if sample.ppo_eligible]
    assert learners
    assert not any(sample.truncated for sample in learners)
    assert all(
        result.frames[sample.frame_index].genuine_action for sample in learners
    )
    assert all(
        result.frames[sample.frame_index].successor is not None
        or result.frames[sample.frame_index].match_boundary
        for sample in learners
    )
    assert all(0 <= sample.terminal_placement < 4 for sample in learners)
    assert any(frame.match_boundary for frame in result.frames if frame.ppo_eligible)
    assert len(result.frames) == len(result.samples)
    # `terminal` is attached retroactively to the last pre-action frame.  Its
    # rollout-policy critic prediction must not be replaced by exact utility,
    # or the final action's current-kyoku residual is silently erased.
    terminal_frames = [
        frame for frame in result.frames
        if frame.ppo_eligible and frame.match_boundary
    ]
    assert terminal_frames
    assert all(
        max(frame.old_boundary_rank_probabilities) < 1.0
        for frame in terminal_frames
    )
    assert any(
        abs(
            frame.old_boundary_rank_value
            - float(RANK_UTILITIES[frame.terminal_placement])
        ) > 1e-5
        for frame in terminal_frames
    )
    advantages = compute(result.samples, result.frames)
    terminal_frame_indices = {
        index for index, frame in enumerate(result.frames)
        if frame.ppo_eligible and frame.match_boundary
    }
    terminal_action_rows = [
        row for row, sample_index in enumerate(advantages.indices)
        if result.samples[int(sample_index)].frame_index in terminal_frame_indices
    ]
    assert terminal_action_rows
    assert any(
        abs(float(advantages.advantages[row])) > 1e-5
        for row in terminal_action_rows
    )


def test_collector_refills_a_bounded_environment_slot_across_match_generations():
    result = _collect_one(target_matches=2)
    generations = {
        outcome.match_id[1] for outcome in result.match_outcomes
    }
    assert result.match_completions == len(result.match_outcomes) == 2
    assert len(generations) == 2


def test_collector_rejects_completed_input_instead_of_refilling_it():
    import riichi
    from zenith_ppo.env.adapter import EnvAdapter

    env = riichi.Env(1, master_seed=31, num_threads=1, privileged=True)
    adapter = EnvAdapter(env)
    batch = adapter.reset([0])
    while int(batch.transition.states[0].lifecycle) != 3:
        batch = (
            adapter.step([
                space.candidates[0].select() for space in batch.action_spaces
            ])
            if batch.action_spaces else adapter.advance([0])
        )
    # Collection owns a fixed launch set and therefore cannot silently refill.
    with pytest.raises(ValueError, match="between one and target_matches"):
        # The validation happens before model use, so placeholders are sufficient.
        from zenith_ppo.rollout.collector import Collector
        Collector(adapter, None, None).collect(
            batch, target_matches=1, curriculum=None, streams=None
        )
    env.close()
