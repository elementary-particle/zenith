def test_collector_continues_until_every_environment_crosses_kyoku():
    import riichi
    import torch

    from zenith_ppo.config import load
    from zenith_ppo.env.adapter import EnvAdapter
    from zenith_ppo.model.actor_critic import ActorCritic
    from zenith_ppo.rewards.curriculum import Curriculum
    from zenith_ppo.rollout.collector import Collector
    from zenith_ppo.seeds import SeedStreams

    config = load("training/configs/smoke.toml")
    model_config = dict(config.values["model"])
    model_config["context_tokens"] = config.values["encoding"]["context_tokens"]
    torch.manual_seed(23)
    model = ActorCritic(model_config)
    env = riichi.Env(2, master_seed=23, num_threads=1, privileged=True)
    try:
        adapter = EnvAdapter(env)
        streams = SeedStreams(23)
        result = Collector(
            adapter,
            model,
            streams.torch_generator("action", "cpu"),
        ).collect(
            adapter.reset([0, 1]),
            target_decisions=1,
            curriculum=Curriculum(config.values["curriculum"]).snapshot(0, 0),
            streams=streams,
            complete_kyoku_per_env=True,
            max_env_calls=1000,
        )
    finally:
        env.close()

    assert result.boundary_aligned
    assert result.kyoku_environment_coverage == 1.0
    assert result.kyoku_completions >= 2
    assert result.decisions > 1
