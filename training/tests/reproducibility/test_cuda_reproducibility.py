import pytest
import torch


@pytest.mark.cuda
def test_cuda_strict_repeatability():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(1)
    first = torch.randn(32, device="cuda")
    torch.manual_seed(1)
    second = torch.randn(32, device="cuda")
    assert torch.equal(first, second)


@pytest.mark.cuda
def test_cuda_strict_one_update_and_restored_next_update(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from copy import deepcopy
    from dataclasses import replace

    from zenith_ppo.cli.train import run_one_update
    from zenith_ppo.config import load

    base = load("training/configs/smoke.toml")
    values = deepcopy(base.values)
    values["run"]["profile"] = "cuda-strict"
    values["curriculum"]["total_matches"] = 2
    config = replace(base, values=values)
    first = run_one_update(config, tmp_path / "first")
    second = run_one_update(config, tmp_path / "second")
    assert (first["trajectory_digest"], first["parameter_digest_after"]) == (
        second["trajectory_digest"], second["parameter_digest_after"]
    )
    resumed_a = run_one_update(config, tmp_path / "resume-a", resume=first["checkpoint_path"])
    resumed_b = run_one_update(config, tmp_path / "resume-b", resume=first["checkpoint_path"])
    assert (resumed_a["trajectory_digest"], resumed_a["parameter_digest_after"]) == (
        resumed_b["trajectory_digest"], resumed_b["parameter_digest_after"]
    )
