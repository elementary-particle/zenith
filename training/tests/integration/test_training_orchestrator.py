import json
from pathlib import Path

from zenith_ppo.config import load
from zenith_ppo.orchestrator import run_training


def _tiny_config(tmp_path: Path, *, total_updates: int):
    source = Path("training/configs/smoke.toml").read_text(encoding="utf-8")
    source = source.replace("num_envs = 64", "num_envs = 4")
    source = source.replace("num_threads = 4", "num_threads = 1")
    source = source.replace(
        "learner_decisions_per_update = 2048",
        "learner_decisions_per_update = 16",
    )
    source = source.replace("total_updates = 1", f"total_updates = {total_updates}")
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
    assert {row["step"] for row in rows if row["axis"] == "update"} == {1, 2}
    assert (output / "checkpoints/latest").is_file()
    assert (output / "profile.json").is_file()
    assert {row["stage"] for row in report["profile"]["stages"]} >= {
        "rollout.encoding",
        "rollout.inference_and_copy",
        "ppo.optimization",
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
