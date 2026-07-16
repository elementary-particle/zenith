from zenith_ppo.types import MatchLineup


def test_self_play_lineup_owns_all_four_seats():
    lineup = MatchLineup((0, 1), ("current",) * 4, 0b1111, 3, "self-play")
    assert [seat for seat in range(4) if lineup.learner_mask & (1 << seat)] == [0, 1, 2, 3]
    assert lineup.seat_policy_ids == ("current",) * 4


def test_native_self_play_rollout_marks_every_policy_row_eligible():
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
    torch.manual_seed(4)
    model = ActorCritic(model_config)
    env = riichi.Env(2, master_seed=4, num_threads=1, privileged=True)
    adapter = EnvAdapter(env)
    initial = adapter.reset([0, 1])
    lineups = {
        (environment_id, 1): MatchLineup(
            (environment_id, 1),
            ("current",) * 4,
            0b1111,
            0,
            "self-play",
        )
        for environment_id in (0, 1)
    }
    streams = SeedStreams(4)
    result = Collector(
        adapter,
        model,
        streams.torch_generator("action", "cpu"),
        lineups=lineups,
    ).collect(
        initial,
        target_matches=2,
        curriculum=Curriculum(config.values["curriculum"]).snapshot(0, 0),
        streams=streams,
    )
    env.close()
    assert {sample.checkpoint_id for sample in result.samples} == {"current"}
    assert all(sample.ppo_eligible for sample in result.samples)


def test_bot_rows_bypass_models_and_are_never_ppo_eligible():
    import riichi
    import torch

    from zenith_ppo.config import load
    from zenith_ppo.capabilities import configure
    from zenith_ppo.env.adapter import EnvAdapter
    from zenith_ppo.inference import CONSERVATIVE_BOT_ID
    from zenith_ppo.model.actor_critic import ActorCritic
    from zenith_ppo.rewards.curriculum import Curriculum
    from zenith_ppo.rollout.collector import Collector
    from zenith_ppo.seeds import SeedStreams

    configure("cpu-smoke")
    config = load("training/configs/smoke.toml")
    model_config = dict(config.values["model"])
    model_config["context_tokens"] = config.values["encoding"]["context_tokens"]
    model = ActorCritic(model_config)
    env = riichi.Env(1, master_seed=8, num_threads=1, privileged=True)
    adapter = EnvAdapter(env)
    initial = adapter.reset([0])
    lineup = MatchLineup(
        (0, 1), ("current", CONSERVATIVE_BOT_ID, "current", CONSERVATIVE_BOT_ID),
        0b0101, 0, "pool",
    )
    streams = SeedStreams(8)
    result = Collector(
        adapter, model, streams.torch_generator("action", "cpu"),
        lineups={(0, 1): lineup}, teacher_config=config.values["teacher"],
    ).collect(
        initial, target_matches=1,
        curriculum=Curriculum(config.values["curriculum"]).snapshot(0, 0),
        streams=streams,
    )
    env.close()
    bot_rows = [sample for sample in result.samples if sample.checkpoint_id == CONSERVATIVE_BOT_ID]
    assert bot_rows and not any(sample.ppo_eligible for sample in bot_rows)
