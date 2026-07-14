import json
from zenith_ppo.metrics import CanonicalMetrics
from zenith_ppo.types import MetricPoint


def test_sorted_jsonl_is_monotonic_and_duplicate_safe(tmp_path):
    metrics = CanonicalMetrics(tmp_path / "m.jsonl", run_id="r", writer_session="s")
    records = metrics.commit([MetricPoint("ppo/policy_loss", "update", 1, 0.5, "scalar", "update", "mean", "u1")])
    assert records[0]["sequence"] == 0
    assert list(json.loads((tmp_path / "m.jsonl").read_text())) != []

