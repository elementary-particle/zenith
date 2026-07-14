import pytest
from zenith_ppo.checkpoint import publish, restore
from zenith_ppo.compatibility import CompatibilityError, CompatibilitySet


def test_mismatch_rejects_before_live_state(tmp_path):
    key = publish(tmp_path, {"model": {}, "state": {}}, compatibility=CompatibilitySet())
    with pytest.raises(CompatibilityError, match="token_schema"):
        restore(tmp_path / key, expected=CompatibilitySet(token_schema=4))


@pytest.mark.parametrize("axis", CompatibilitySet.__dataclass_fields__)
def test_every_axis_is_independently_checked(tmp_path, axis):
    key = publish(tmp_path, {"model": {}, "state": {}}, compatibility=CompatibilitySet())
    values = {axis: getattr(CompatibilitySet(), axis) + 1}
    with pytest.raises(CompatibilityError, match=axis):
        restore(tmp_path / key, expected=CompatibilitySet(**values))
