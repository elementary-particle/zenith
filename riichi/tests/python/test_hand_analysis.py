import numpy as np
import pytest

import riichi


def valid_hands():
    counts = np.zeros((2, 34), np.uint8)
    counts[0, [0, 1, 2, 9, 10, 11, 18, 19, 20]] = 1
    counts[0, 27] = 2
    counts[0, 28] = 2
    counts[1] = counts[0]
    return counts


def test_hand_analysis_v2_is_ordered_read_only_and_env_independent():
    counts = valid_hands()
    result = riichi.analyze_hands(counts, np.zeros(2, np.uint8))
    assert result.analysis_version == 2
    assert result.row_count == 2
    assert result.shanten.shape == (2, 4)
    assert result.improving_type_mask.shape == (2,)
    assert not result.shanten.flags.writeable
    assert not result.improving_type_mask.flags.writeable
    np.testing.assert_array_equal(result.shanten[0], result.shanten[1])
    assert result.improving_type_mask[0] == result.improving_type_mask[1]


@pytest.mark.parametrize(
    "counts, melds",
    [
        (np.zeros((1, 33), np.uint8), np.zeros(1, np.uint8)),
        (np.full((1, 34), 5, np.uint8), np.zeros(1, np.uint8)),
        (np.zeros((1, 34), np.uint8), np.full(1, 5, np.uint8)),
    ],
)
def test_invalid_hand_analysis_inputs_fail(counts, melds):
    with pytest.raises(ValueError):
        riichi.analyze_hands(counts, melds)
