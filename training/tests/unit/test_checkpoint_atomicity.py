import os
import pytest
from zenith_ppo.checkpoint import CheckpointError, publish, resolve_latest, restore, validate
from zenith_ppo.cli.evaluate import _checkpoint_argument_path


def test_atomic_checkpoint_round_trip(tmp_path):
    checkpoint = publish(tmp_path, {"model": {"x": 1}, "state": {"update": 1}})
    assert resolve_latest(tmp_path).name == checkpoint
    assert restore(tmp_path / checkpoint)["state"]["update"] == 1


def test_evaluator_accepts_checkpoint_collection_root(tmp_path):
    checkpoint = tmp_path / "checkpoint-id"
    checkpoint.mkdir()
    (tmp_path / "latest").write_text(checkpoint.name, encoding="ascii")

    assert _checkpoint_argument_path(tmp_path) == checkpoint
    assert _checkpoint_argument_path(checkpoint) == checkpoint


def test_corruption_forgery_and_failed_publish_preserve_previous_latest(tmp_path, monkeypatch):
    first = publish(tmp_path, {"model": {"x": 1}, "state": {}})
    corrupt = tmp_path / first / "state.json"
    corrupt.write_text('{"corrupt":true}')
    with pytest.raises(CheckpointError, match="checksum"):
        validate(tmp_path / first)
    # Restore the first checkpoint, then inject interruption before the next atomic rename.
    first = publish(tmp_path, {"model": {"x": 1}, "state": {}}, metadata={"revision": 2})
    real_replace = os.replace

    def fail(source, destination):
        if ".checkpoint-staging-" in str(source):
            raise OSError("injected interruption")
        return real_replace(source, destination)
    monkeypatch.setattr(os, "replace", fail)
    with pytest.raises(OSError, match="interruption"):
        publish(tmp_path, {"model": {"x": 2}, "state": {}})
    assert resolve_latest(tmp_path).name == first
