import pytest

from zenith_ppo.ppo.current_kyoku import explained_variance, rank_explained_variance


def test_rank_explained_variance_uses_placement_utility():
    assert rank_explained_variance([1.0, -1.0], [0, 3]) == pytest.approx(1.0)


def test_explained_variance_validates_shapes_and_degenerate_targets():
    assert explained_variance([], []) == 0.0
    assert explained_variance([0.0, 0.0], [1.0, 1.0]) == 0.0
    with pytest.raises(ValueError, match="same shape"):
        explained_variance([0.0], [0.0, 1.0])
