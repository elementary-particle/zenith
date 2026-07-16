import json
from pathlib import Path

from zenith_ppo.config import load
from zenith_ppo.orchestrator import run_training


def _tiny_config(tmp_path: Path, *, total_updates: int):
    source = Path("training/configs/smoke.toml").read_text(encoding="utf-8")
    source = source.replace("total_matches = 1", f"total_matches = {total_updates}")
    source = source.replace("enabled = true", "enabled = false")
    path = tmp_path / "tiny.toml"
    path.write_text(source, encoding="utf-8")
    return load(path)


def test_complete_driver_runs_all_updates_and_preserves_environment(tmp_path):
    output = tmp_path / "run"
    report = run_training(
        _tiny_config(tmp_path, total_updates=2), output, profile_stages=True
    )

    manifest = json.loads((output / "run.json").read_text())
    rows = [json.loads(line) for line in (output / "metrics/canonical.jsonl").read_text().splitlines()]
    assert report["status"] == "completed"
    assert report["update"] == 2
    assert manifest["status"] == "completed"
    assert {row["step"] for row in rows if row["axis"] == "match"} == {1, 2}
    assert (output / "checkpoints/latest").is_file()
    assert (output / "profile.json").is_file()
    assert {row["stage"] for row in report["profile"]["stages"]} >= {
        "rollout.encoding",
        "rollout.actor_candidate_processing",
        "rollout.oracle_encoding",
        "rollout.oracle_forward",
        "ppo.actor_optimization",
        "ppo.critic_optimization",
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


def test_periodic_checkpoints_follow_completed_match_cadence(tmp_path):
    source = Path("training/configs/smoke.toml").read_text(encoding="utf-8")
    source = source.replace("total_matches = 1", "total_matches = 3")
    source = source.replace("checkpoint_matches = [1]", "checkpoint_matches = [999]")
    source = source.replace(
        "[checkpoint]\ncadence_matches = 1",
        "[checkpoint]\ncadence_matches = 2",
    )
    source = source.replace("enabled = true", "enabled = false")
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
