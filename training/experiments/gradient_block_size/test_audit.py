import numpy as np
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from audit import (
    GradientMoments,
    _trainer_compatible_batches,
    cosine,
    gradient_report,
)


def test_gradient_report_recovers_signal_and_noise_scale():
    moments = GradientMoments()
    for gradient in (
        np.asarray([2.0, 0.0]),
        np.asarray([0.0, 0.0]),
        np.asarray([2.0, 0.0]),
        np.asarray([0.0, 0.0]),
    ):
        moments.add(gradient)
    report = gradient_report(moments, 8)
    assert report["blocks"] == 4
    assert report["total_matches"] == 32
    assert report["mean_gradient_norm_squared"] == pytest.approx(1.0)
    assert report["gradient_noise_trace_per_match"] == pytest.approx(32 / 3)
    assert report["debiased_gradient_signal_squared"] == pytest.approx(2 / 3)
    assert report["critical_batch_matches"] == pytest.approx(16.0)
    assert report["snr_at_block_size"] == pytest.approx(2 ** -0.5)


def test_cosine_handles_orthogonal_and_zero_vectors():
    assert cosine(np.asarray([1.0, 0.0]), np.asarray([0.0, 1.0])) == 0.0
    assert cosine(np.zeros(2), np.ones(2)) is None


def test_trainer_compatibility_tensorizes_only_entropy_prepass_fields():
    selected = np.asarray([0], dtype=np.int64)
    batch = {
        "selected": selected,
        "model_inputs": {
            "action_factors": np.zeros((1, 1, 15), dtype=np.int32),
            "action_lengths": np.ones(1, dtype=np.int64),
            "token_factors": np.zeros((1, 1, 10), dtype=np.int32),
        },
    }
    converted = _trainer_compatible_batches((batch,))[0]
    assert converted["model_inputs"]["action_factors"].shape == (1, 1, 15)
    assert converted["model_inputs"]["action_lengths"].device.type == "cpu"
    assert isinstance(converted["model_inputs"]["token_factors"], np.ndarray)
    assert converted["selected"] is selected
