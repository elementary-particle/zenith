import json
import pytest
from zenith_ppo.metrics import CanonicalMetrics
from zenith_ppo.types import MetricPoint


def test_sorted_jsonl_is_monotonic_and_duplicate_safe(tmp_path):
    metrics = CanonicalMetrics(tmp_path / "m.jsonl", run_id="r", writer_session="s")
    records = metrics.commit([MetricPoint("ppo/policy_loss", "match", 1, 0.5, "scalar", "batch", "mean", "m1")])
    assert records[0]["sequence"] == 0
    assert list(json.loads((tmp_path / "m.jsonl").read_text())) != []


def test_resume_truncates_records_after_checkpoint_cursor(tmp_path):
    path = tmp_path / "m.jsonl"
    metrics = CanonicalMetrics(path, run_id="r", writer_session="first")
    metrics.commit([
        MetricPoint("ppo/policy_loss", "match", 1, 0.5,
                    "scalar", "batch", "mean", "update"),
        MetricPoint("ppo/policy_loss", "match", 2, 9.9,
                    "scalar", "batch", "mean", "update"),
    ])

    resumed = CanonicalMetrics(
        path, run_id="r", writer_session="second", resume_sequence=1
    )
    resumed.commit([
        MetricPoint("ppo/policy_loss", "match", 2, 0.4,
                    "scalar", "batch", "mean", "update"),
    ])

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [(row["sequence"], row["step"], row["value"]) for row in rows] == [
        (0, 1, 0.5),
        (1, 2, 0.4),
    ]
    assert rows[-1]["writer_session"] == "second"


def test_resume_rejects_checkpoint_cursor_past_canonical_log(tmp_path):
    path = tmp_path / "m.jsonl"
    CanonicalMetrics(path, run_id="r", writer_session="first").commit([
        MetricPoint("ppo/policy_loss", "match", 1, 0.5,
                    "scalar", "batch", "mean", "update"),
    ])

    with pytest.raises(ValueError, match="checkpoint metric cursor 2"):
        CanonicalMetrics(
            path, run_id="r", writer_session="second", resume_sequence=2
        )
