import json
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from zenith_ppo.checkpoint import publish, resolve_latest, restore
from zenith_ppo.config import load
from zenith_ppo.model.factory import build_actor_critic
from zenith_ppo.orchestrator import run_training


def _smoke_source():
    source = Path("training/configs/smoke.toml").read_text(encoding="utf-8")
    default = Path("training/configs/default.toml").resolve()
    return source.replace(
        'extends = "default.toml"', f'extends = "{default}"'
    )


def _tiny_config(tmp_path: Path, *, total_updates: int):
    source = _smoke_source()
    source = source.replace("total_matches = 1", f"total_matches = {total_updates}")
    source = source.replace(
        "[metrics.tensorboard]\nenabled = true",
        "[metrics.tensorboard]\nenabled = false",
    )
    path = tmp_path / "tiny.toml"
    path.write_text(source, encoding="utf-8")
    return load(path)


def test_complete_driver_runs_all_updates_and_preserves_environment(tmp_path):
    output = tmp_path / "run"
    config = _tiny_config(tmp_path, total_updates=2)
    model_config = dict(config.values["model"])
    model_config["context_tokens"] = config.values["encoding"]["context_tokens"]
    bc_root = tmp_path / "bc-checkpoints"
    publish(bc_root, {
        "model": build_actor_critic(model_config).state_dict(),
        "state": {
            "architecture": "verified-public-state-value-ppo-v1",
            "phase": "behavior_cloning_complete",
            "selected_epoch": 1,
        },
    })
    report = run_training(
        config, output, profile_stages=True, initial_checkpoint=bc_root,
    )

    manifest = json.loads((output / "run.json").read_text())
    rows = [json.loads(line) for line in (output / "metrics/canonical.jsonl").read_text().splitlines()]
    assert report["status"] == "completed"
    assert report["update"] == 2
    assert list(report["last_update"]["league_updates"]) == ["self-play"]
    assert all(
        update["committed"]
        for update in report["last_update"]["league_updates"].values()
    )
    assert manifest["status"] == "completed"
    assert {row["step"] for row in rows if row["axis"] == "match"} == {1, 2}
    names = {row["name"] for row in rows}
    assert "league/arena_size" in names
    assert "league/pairwise_games" in names
    assert not any(name.startswith("rollout_rating/") for name in names)
    assert "ppo/magnet_kl" in names
    assert "ppo/magnet_parameter_rms_distance" in names
    assert "rollout_rating/win_rate_vs_conservative_bot" not in names
    assert (output / "checkpoints/latest").is_file()
    baseline_evaluation = json.loads(
        (output / "evaluations/bc-baseline.json").read_text()
    )
    assert baseline_evaluation["status"] == "completed"
    assert baseline_evaluation["protocol"] == (
        "one learner versus three frozen BC seats"
    )
    assert baseline_evaluation["learner_seats_per_game"] == 1
    assert baseline_evaluation["baseline_seats_per_game"] == 3
    assert baseline_evaluation["valid_games"] > 0
    assert 0 <= baseline_evaluation["pairwise_win_rate"] <= 1
    assert baseline_evaluation["paired_bootstrap"][
        "pairwise_win_rate"
    ]["paired_seeds"] == 4
    exploitability = json.loads(
        (output / "evaluations/exploitability-000000001.json").read_text()
    )
    assert exploitability["protocol"] == "restricted unilateral deviation"
    assert exploitability["primary_stat"] == "rank_advantage"
    assert exploitability["matches_per_witness"] == 16
    assert {row["kind"] for row in exploitability["witnesses"]} == {
        "current-greedy", "bc",
    }
    assert exploitability["maximum_rank_witness"]["rank_advantage"] == max(
        (
            row["paired_bootstrap"]["rank_advantage"]
            for row in exploitability["witnesses"]
        ),
        key=lambda value: value["mean"],
    )
    assert (
        output / "evaluations/exploitability-000000001.outcomes.jsonl"
    ).is_file()
    assert (output / "profile.json").is_file()
    assert {row["stage"] for row in report["profile"]["stages"]} >= {
            "rollout.encoding",
            "rollout.actor_candidate_processing",
            "rollout.frame_critic_forward",
        "ppo.actor_optimization",
        "ppo.state_critic_optimization",
        "ppo.boundary_critic_optimization",
        "checkpoint.publish",
    }


def test_bounded_run_can_resume_to_configured_completion(tmp_path):
    output = tmp_path / "resume"
    config = _tiny_config(tmp_path, total_updates=3)

    first = run_training(config, output, max_updates=1)
    second = run_training(config, output, resume=output / "checkpoints")

    assert first["status"] == "stopped"
    assert first["update"] == 1
    assert second["status"] == "completed"
    assert second["update"] == 3
    restored = restore(resolve_latest(output / "checkpoints"))
    league = restored["trainer"]["population"]["league"]
    assert league["version"] == 1
    assert league["kind"] == "pure_self_play"
    assert league["policy_ids"] == ["self-play"]
    assert league["games"] == 3
    ema_magnet = restored["optimizer"]["ema_magnet"]
    assert ema_magnet["version"] == 1
    assert ema_magnet["updates"] == 3
    assert ema_magnet["completed_matches"] == 3
    assert restored["optimizer"]["adaptive_kl"]["version"] == 1


def test_exact_resume_ignores_uninitialized_native_environment_capacity(tmp_path):
    source = _smoke_source()
    source = source.replace("total_matches = 1", "total_matches = 2")
    source = source.replace("num_envs = 1", "num_envs = 4")
    source = source.replace(
        "[metrics.tensorboard]\nenabled = true",
        "[metrics.tensorboard]\nenabled = false",
    )
    path = tmp_path / "resume-unused-capacity.toml"
    path.write_text(source, encoding="utf-8")
    config = load(path)
    output = tmp_path / "resume-unused-capacity"

    run_training(config, output, max_updates=1)
    checkpoint = restore(resolve_latest(output / "checkpoints"))
    snapshots = checkpoint["trainer"]["env"]["snapshots"]
    assert set(map(int, snapshots)) == {0}

    resumed = run_training(config, output, resume=output / "checkpoints")
    assert resumed["status"] == "completed"
    assert resumed["update"] == 2


def test_streaming_update_reuses_one_slot_for_multiple_gradient_chunks(tmp_path):
    source = _smoke_source()
    source = source.replace("total_matches = 1", "total_matches = 2")
    source = source.replace("matches_per_update = 1", "matches_per_update = 2")
    source = source.replace("[ppo]\n", "[ppo]\nepochs = 1\n", 1)
    source = source.replace(
        "[metrics.tensorboard]\nenabled = true",
        "[metrics.tensorboard]\nenabled = false",
    )
    path = tmp_path / "streaming-chunks.toml"
    path.write_text(source, encoding="utf-8")

    report = run_training(load(path), tmp_path / "streaming-chunks")

    assert report["status"] == "completed"
    assert report["update"] == 1
    assert report["policy_version"] == 1
    assert report["last_update"]["rollout_chunk_matches"] == 1
    assert report["last_update"]["rollout_chunks"] == 2
    assert report["last_update"]["learner_decisions"] > 0


def test_resume_from_older_checkpoint_replaces_metric_and_tensorboard_tail(tmp_path):
    source = _smoke_source()
    source = source.replace("total_matches = 1", "total_matches = 4")
    source = source.replace("matches_per_update = 1", "matches_per_update = 2")
    source = source.replace("num_envs = 1", "num_envs = 2")
    source = source.replace("keep = 2", "keep = 10")
    path = tmp_path / "resume-tail.toml"
    path.write_text(source, encoding="utf-8")
    config = load(path)
    output = tmp_path / "resume-tail"

    first = run_training(config, output, max_updates=1)
    checkpoint = first["last_update"]["checkpoint_path"]
    run_training(config, output, resume=output / "checkpoints", max_updates=1)
    replayed = run_training(config, output, resume=checkpoint)

    rows = [
        json.loads(line)
        for line in (output / "metrics/canonical.jsonl").read_text().splitlines()
    ]
    policy_rows = [row for row in rows if row["name"] == "ppo/policy_loss"]
    assert replayed["status"] == "completed"
    assert [row["step"] for row in policy_rows] == [2, 4]
    assert len({row["sequence"] for row in rows}) == len(rows)

    events = EventAccumulator(str(output / "tensorboard"))
    events.Reload()
    assert [event.step for event in events.Tensors("ppo/policy_loss")] == [2, 4]


def test_periodic_checkpoints_follow_completed_match_cadence(tmp_path):
    source = _smoke_source()
    source = source.replace("total_matches = 1", "total_matches = 3")
    source = source.replace("checkpoint_matches = [1]", "checkpoint_matches = [999]")
    source = source.replace(
        "[checkpoint]\ncadence_matches = 1",
        "[checkpoint]\ncadence_matches = 2",
    )
    source = source.replace(
        "[metrics.tensorboard]\nenabled = true",
        "[metrics.tensorboard]\nenabled = false",
    )
    path = tmp_path / "checkpoint-cadence.toml"
    path.write_text(source, encoding="utf-8")
    output = tmp_path / "checkpoint-cadence"

    run_training(load(path), output)

    checkpoint_updates = {
        json.loads(state.read_text(encoding="utf-8"))["update"]
        for state in (output / "checkpoints").glob("*/state.json")
    }
    assert 1 not in checkpoint_updates
    assert {0, 2, 3} <= checkpoint_updates


def test_periodic_evaluation_can_be_deferred_without_changing_training(tmp_path):
    output = tmp_path / "deferred-evaluation"
    config = _tiny_config(tmp_path, total_updates=1)

    report = run_training(
        config, output, skip_periodic_evaluation=True,
    )

    assert report["status"] == "completed"
    assert report["update"] == 1
    assert "evaluation" not in report["last_update"]
    assert not (output / "evaluations").exists()
    assert (output / "checkpoints/latest").is_file()


def test_checkpoint_league_evaluates_held_out_frozen_response(tmp_path):
    target_root = tmp_path / "target-checkpoints"
    base_config = _tiny_config(tmp_path, total_updates=1)
    model_config = dict(base_config.values["model"])
    model_config["context_tokens"] = base_config.values["encoding"][
        "context_tokens"
    ]
    target_id = publish(target_root, {
        "model": build_actor_critic(model_config).state_dict(),
        "state": {"architecture": "verified-public-state-value-ppo-v1"},
    })
    source = _smoke_source().replace(
        "matches_per_update = 1",
        "matches_per_update = 1\n"
        "training_mode = \"checkpoint_league\"\n"
        f"league_checkpoints = [\"{target_root}\"]\n"
        "league_learner_seats = 1\n"
        "league_minimum_games = 1\n"
        "league_uniform_fraction = 1.0",
    )
    source = source.replace(
        "[metrics.tensorboard]\nenabled = true",
        "[metrics.tensorboard]\nenabled = false",
    )
    path = tmp_path / "checkpoint-league.toml"
    path.write_text(source, encoding="utf-8")
    output = tmp_path / "checkpoint-league"

    report = run_training(load(path), output)

    response = report["last_update"]["evaluation"][
        "frozen_checkpoint_responses"
    ]
    assert response["protocol"] == (
        "one learner versus three frozen target seats"
    )
    assert response["claim"] == (
        "held-out achieved response gain, not a certified best response"
    )
    assert response["learner_seats_per_game"] == 1
    assert response["target_seats_per_game"] == 3
    assert response["matches_per_target"] == 16
    assert [row["target_checkpoint_id"] for row in response["targets"]] == [
        target_id
    ]
    assert response["targets"][0]["paired_bootstrap"]["rank_advantage"][
        "paired_seeds"
    ] == 4
    assert response["minimum_rank_target"]["target_checkpoint_id"] == target_id
    stem = output / "evaluations/frozen-response-000000001"
    assert stem.with_suffix(".json").is_file()
    assert Path(f"{stem}.outcomes.jsonl").is_file()
    assert not (
        output / "evaluations/exploitability-000000001.json"
    ).exists()


def test_weights_only_starts_fresh(tmp_path):
    source_output = tmp_path / "source"
    config = _tiny_config(tmp_path, total_updates=1)
    run_training(config, source_output)
    source_checkpoint = source_output / "checkpoints" / (
        source_output / "checkpoints/latest"
    ).read_text(encoding="ascii").strip()
    output = tmp_path / "weights-only"
    report = run_training(
        config,
        output,
        resume=source_checkpoint,
        weights_only=True,
        max_updates=1,
    )

    assert report["status"] == "completed"
    assert report["update"] == 1
    manifest = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert manifest["update"] == 1


def test_branch_resume_preserves_state_and_records_source(tmp_path):
    source_output = tmp_path / "branch-source"
    config = _tiny_config(tmp_path, total_updates=2)
    first = run_training(
        config,
        source_output,
        max_updates=1,
        skip_periodic_evaluation=True,
    )
    source_checkpoint = resolve_latest(source_output / "checkpoints")
    source_id = restore(source_checkpoint)["manifest"]["checkpoint_id"]

    output = tmp_path / "branch"
    second = run_training(
        config,
        output,
        resume=source_checkpoint,
        branch_resume=True,
        max_updates=1,
        skip_periodic_evaluation=True,
    )

    assert first["update"] == 1
    assert second["update"] == 2
    branched = restore(resolve_latest(output / "checkpoints"))
    metadata = branched["manifest"]["metadata"]
    assert metadata["reproducibility_level"] == "numerical_compatible"
    assert metadata["branch_source"] == {
        "checkpoint_id": source_id,
        "checkpoint_path": str(source_checkpoint.resolve()),
    }
    assert branched["optimizer"]["ema_magnet"]["updates"] == 2


def test_ema_self_play_updates_and_restores_dynamic_opponent(tmp_path):
    source = _smoke_source().replace("total_matches = 1", "total_matches = 2")
    source = source.replace(
        "matches_per_update = 1",
        "matches_per_update = 1\ntraining_mode = \"ema_self_play\"\n"
        "ema_opponent_half_life_matches = 4",
    )
    source = source.replace(
        "[metrics.tensorboard]\nenabled = true",
        "[metrics.tensorboard]\nenabled = false",
    )
    path = tmp_path / "ema-self-play.toml"
    path.write_text(source, encoding="utf-8")
    config = load(path)
    output = tmp_path / "ema-self-play"

    first = run_training(
        config, output, max_updates=1, skip_periodic_evaluation=True
    )
    first_state = restore(resolve_latest(output / "checkpoints"))
    ema_state = first_state["trainer"]["population"]["opponent_ema"]
    assert first["last_update"]["rollout_opponents"] == "ema-self-play"
    assert first_state["trainer"]["population"]["league"]["games"] == 1
    assert ema_state["updates"] == 1
    assert ema_state["completed_matches"] == 1

    second = run_training(
        config,
        output,
        resume=output / "checkpoints",
        skip_periodic_evaluation=True,
    )
    second_state = restore(resolve_latest(output / "checkpoints"))
    assert second["status"] == "completed"
    assert second_state["trainer"]["population"]["league"]["games"] == 2
    assert second_state["trainer"]["population"]["opponent_ema"]["updates"] == 2


def test_weights_only_bc_resume_records_evaluation_baseline(tmp_path):
    output = tmp_path / "weights-only-bc"
    config = _tiny_config(tmp_path, total_updates=1)
    model_config = dict(config.values["model"])
    model_config["context_tokens"] = config.values["encoding"]["context_tokens"]
    bc_root = tmp_path / "bc-checkpoints"
    checkpoint_id = publish(bc_root, {
        "model": build_actor_critic(model_config).state_dict(),
        "state": {
            "architecture": "verified-public-state-value-ppo-v1",
            "phase": "behavior_cloning_complete",
            "selected_epoch": 3,
        },
    })

    report = run_training(
        config, output, resume=bc_root, weights_only=True, max_updates=0,
    )

    assert report["status"] == "stopped"
    manifest = json.loads((output / "run.json").read_text())
    assert manifest["initial_policy"]["checkpoint_id"] == checkpoint_id
    baseline = Path(manifest["initial_policy"]["checkpoint_path"])
    assert baseline.parent == output / "baselines"
    assert restore(baseline)["manifest"]["checkpoint_id"] == checkpoint_id


def test_evaluation_rng_guard_restores_all_training_generators():
    import random
    import numpy as np
    import torch

    from zenith_ppo.orchestrator import _preserve_training_rng_state

    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    expected = (random.random(), np.random.random(), torch.rand(()).item())
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    with _preserve_training_rng_state():
        random.random()
        np.random.random()
        torch.rand(100)
    actual = (random.random(), np.random.random(), torch.rand(()).item())

    assert actual == expected
