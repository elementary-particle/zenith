import pytest
from zenith_ppo.compatibility import CompatibilityError, CompatibilitySet


def test_axes_are_independent_and_diagnostic():
    expected = CompatibilitySet()
    with pytest.raises(CompatibilityError, match="event_schema"):
        expected.require(CompatibilitySet(event_schema=3))
    assert expected.digest != CompatibilitySet(event_schema=3).digest
