import math
import pytest


def _collect_one(seed=23):
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
    env = riichi.Env(1, master_seed=seed, num_threads=1, privileged=True)
    adapter = EnvAdapter(env)
    streams = SeedStreams(seed)
    result = Collector(
        adapter, model, streams.torch_generator("action", "cpu")
    ).collect(
        adapter.reset([0]), target_matches=1,
        curriculum=Curriculum(config.values["curriculum"]).snapshot(0, 0),
        streams=streams, max_env_calls=4096,
    )
    env.close()
    return result


def test_complete_match_has_terminal_rank_reward_and_no_bootstrap_samples():
    result = _collect_one()
    assert result.match_completions == len(result.match_outcomes) == 1
    assert result.match_outcomes[0].completed_kyoku > 0
    learners = [sample for sample in result.samples if sample.ppo_eligible]
    assert learners
    assert not any(sample.truncated for sample in learners)
    assert not any(math.isfinite(sample.bootstrap_score_value) or
                   math.isfinite(sample.bootstrap_rank_value) for sample in learners)
    assert all(sample.successor is not None or sample.terminal for sample in learners)
    assert all(0 <= sample.terminal_placement < 4 for sample in learners)
    assert any(sample.terminal and sample.reward.rank_reward != 0 for sample in learners)


def test_collector_rejects_completed_input_instead_of_refilling_it():
    import riichi
    from zenith_ppo.env.adapter import EnvAdapter

    env = riichi.Env(1, master_seed=31, num_threads=1, privileged=True)
    adapter = EnvAdapter(env)
    batch = adapter.reset([0])
    while int(batch.transition.states[0].lifecycle) != 3:
        batch = adapter.step([decision.actions[0] for decision in batch.decisions])
    # Collection owns a fixed launch set and therefore cannot silently refill.
    with pytest.raises(ValueError, match="exactly 1 fresh matches"):
        # The validation happens before model use, so placeholders are sufficient.
        from zenith_ppo.rollout.collector import Collector
        Collector(adapter, None, None).collect(
            batch, target_matches=1, curriculum=None, streams=None
        )
    env.close()
