from zenith_ppo.types import MatchLineup


def test_fixed_lineup_and_current_only_ownership():
    lineup = MatchLineup((0, 1), ("current", "a", "current", "b"), 0b0101, 3, "pool")
    assert [seat for seat in range(4) if lineup.learner_mask & (1 << seat)] == [0, 2]
    assert lineup.seat_policy_ids == ("current", "a", "current", "b")


def test_native_rollout_keeps_lineup_and_marks_only_learner_rows():
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
    torch.manual_seed(4)
    model = ActorCritic(model_config)
    env = riichi.Env(2, master_seed=4, num_threads=1, privileged=True)
    adapter = EnvAdapter(env)
    initial = adapter.reset([0, 1])
    lineups = {
        (environment_id, 1): MatchLineup(
            (environment_id, 1),
            ("current", "a", "current", "b"),
            0b0101,
            0,
            "pool",
        )
        for environment_id in (0, 1)
    }
    streams = SeedStreams(4)
    result = Collector(
        adapter,
        model,
        streams.torch_generator("action", "cpu"),
        lineups=lineups,
        policy_models={"a": model, "b": model},
    ).collect(
        initial,
        target_decisions=32,
        curriculum=Curriculum(config.values["curriculum"]).snapshot(0, 0),
        streams=streams,
    )
    env.close()
    assert {sample.checkpoint_id for sample in result.samples} == {"current", "a", "b"}
    assert all(
        sample.ppo_eligible == (sample.binding.seat in {0, 2})
        for sample in result.samples
    )
