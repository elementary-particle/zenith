"""Start-of-kyoku match features for the boundary-rank critic."""

from __future__ import annotations

from itertools import permutations


BOUNDARY_RANK_FEATURES = 28
POINTS_PER_REWARD = 24_000.0
RANK_ORDERS = tuple(permutations(range(4)))
RANK_ORDER_INDEX = {order: index for index, order in enumerate(RANK_ORDERS)}


def encode_rank_boundary(frame):
    """Encode score, dealer, round, counters, and remaining match structure."""
    import numpy as np

    dealer = int(frame.get("dealer", 0))
    round_index = max(0, min(
        15,
        int(frame.get("round_wind", 0)) * 4
        + int(frame.get("hand_number", 0)),
    ))

    def one_hot(value, size):
        return [float(index == int(value)) for index in range(int(size))]

    scores = frame.get("scores", (0, 0, 0, 0))
    result = np.asarray([
        *(
            float(scores[(dealer + offset) % 4]) / POINTS_PER_REWARD
            for offset in range(4)
        ),
        *one_hot(dealer, 4),
        *one_hot(round_index, 16),
        min(float(frame.get("honba", 0)) / 5.0, 2.0),
        min(float(frame.get("riichi_deposits", 0)) / 5.0, 2.0),
        max(0, 7 - round_index) / 8.0,
        float(round_index >= 8),
    ], dtype=np.float32)
    if result.shape != (BOUNDARY_RANK_FEATURES,):
        raise AssertionError("rank boundary feature contract changed")
    return np.ascontiguousarray(result)
