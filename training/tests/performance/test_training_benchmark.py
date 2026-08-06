import os
import pytest
import resource
import time
import torch

from zenith_ppo.encoding.packing import pack
from zenith_ppo.metrics import TensorBoardProjector


@pytest.mark.performance
def test_benchmark_is_explicitly_opt_in(tmp_path):
    if os.environ.get("ZENITH_RUN_BENCHMARKS") != "1":
        pytest.skip("set ZENITH_RUN_BENCHMARKS=1")
    started = time.perf_counter()
    packed = pack(([64, 128, 256, 512] * 1024), 8192)
    elapsed = time.perf_counter() - started
    tensorboard_started = time.perf_counter()
    projector = TensorBoardProjector(tmp_path / "tensorboard", run_id="benchmark", writer_session="0")
    projector.enqueue([{"name": "ppo/policy_loss", "value": 0.1, "step": 1}])
    projector.close()
    tensorboard_elapsed = time.perf_counter() - tensorboard_started
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tensor = torch.randn(1024, 256, device=device)
    memory = torch.cuda.max_memory_allocated() if device == "cuda" else tensor.numel() * tensor.element_size()
    assert packed.padded_tokens >= packed.total_tokens
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > 0 and elapsed < 5
    assert memory > 0 and tensorboard_elapsed < 5
