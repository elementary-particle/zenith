import pytest
import torch
from zenith_ppo.encoding.fourier import DOMAINS, features


def test_features_are_finite_distinct_and_bounded():
    value = features(torch.tensor([0.0, 1.0]), DOMAINS["progress"])
    assert value.dtype == torch.float32 and torch.isfinite(value).all()
    assert not torch.equal(value[0], value[1])
    with pytest.raises(ValueError):
        features(torch.tensor([2.0]), DOMAINS["progress"])
