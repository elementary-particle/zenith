from zenith_ppo.checkpoint import publish, restore
from zenith_ppo.compatibility import CompatibilitySet


def test_curriculum_population_state_round_trip(tmp_path):
    state = {"model": {}, "state": {"curriculum": {"progress": .3}, "pool": ["a", "b"]}}
    key = publish(tmp_path, state, compatibility=CompatibilitySet())
    assert restore(tmp_path / key, expected=CompatibilitySet())["state"] == state["state"]


def test_every_anchor_and_blend_position_round_trips(tmp_path):
    for index, progress in enumerate((0, .2, .275, .35, .6, .675, .75, 1)):
        root = tmp_path / str(index)
        state = {"model": {}, "state": {"progress": progress, "pool": ["a"], "rating_cursor": index}}
        key = publish(root, state, compatibility=CompatibilitySet())
        assert restore(root / key, expected=CompatibilitySet())["state"] == state["state"]
