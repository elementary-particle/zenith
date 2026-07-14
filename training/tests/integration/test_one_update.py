from pathlib import Path
from zenith_ppo.cli.smoke import _one_update
from zenith_ppo.config import load


def test_parameter_changing_cpu_update(tmp_path):
    result = _one_update(load("training/configs/smoke.toml"), tmp_path)
    assert result["status"] == "passed"
    assert result["parameter_digest_before"] != result["parameter_digest_after"]

