import numpy as np
from zenith_ppo.metrics import histogram_sample


def test_histogram_budget_and_determinism():
    values = np.arange(1000, dtype=np.float32)
    first = histogram_sample(values, max_elements=10, max_bytes=40, rng=np.random.default_rng(3))
    second = histogram_sample(values, max_elements=10, max_bytes=40, rng=np.random.default_rng(3))
    assert np.array_equal(first, second) and first.nbytes <= 40

