from zenith_ppo.cli.smoke import _one_update
from zenith_ppo.config import load


def test_parameter_changing_cpu_update(tmp_path):
    result = _one_update(load("training/configs/smoke.toml"), tmp_path)
    assert result["status"] == "completed"
    update = result["last_update"]
    assert update["status"] == "passed"
    assert update["parameter_digest_before"] != update["parameter_digest_after"]
    assert update["ppo"]["actor_optimization_fraction"] == 1
    assert update["ppo"]["critic_optimization_fraction"] == 1
    assert update["ppo"]["post_update_approximate_kl"] >= 0
