from types import SimpleNamespace

import numpy as np
import pytest

from experiments.gae_falls_short.current_kyoku_qboost import (
    kyoku_segments,
    lambda_one_advantages,
)


def test_lambda_one_reverse_control_variate_resets_at_boundary():
    result = lambda_one_advantages(
        np.asarray([0.8, 0.8, -0.4]),
        np.asarray([0.1, 0.5, 9.0]),
        np.asarray([0.2, 0.3, 1.0]),
        ((0, 1), (2,)),
    )
    # First row removes the next sampled-action residual; the second segment
    # cannot affect it.  The last row is simply G - V at its own boundary.
    assert np.allclose(result, [0.4, 0.5, -1.4])


def test_lambda_one_rejects_incomplete_segments():
    with pytest.raises(ValueError, match="do not cover"):
        lambda_one_advantages(
            np.ones(2), np.zeros(2), np.zeros(2), ((0,),)
        )


def test_collector_rows_split_by_seat_and_kyoku():
    def sample(frame, seat, boundary=False):
        return SimpleNamespace(
            binding=SimpleNamespace(
                environment_id=0,
                episode_generation=2,
                frame_id=frame,
                seat=seat,
            ),
            checkpoint_id="current",
            kyoku_boundary=boundary,
        )

    samples = (
        sample(1, 0),
        sample(2, 1, True),
        sample(3, 0, True),
        sample(4, 0, True),
    )
    assert kyoku_segments(samples, range(4)) == ((0, 2), (3,), (1,))
