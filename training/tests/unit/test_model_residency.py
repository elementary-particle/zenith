from zenith_ppo.population.residency import ModelResidency


def test_count_bounded_lru_does_not_reload_hits():
    cache = ModelResidency(max_models=1, max_bytes=10)
    cache.get("a", lambda key: (key, 4)); cache.get("a", lambda key: (_ for _ in ()).throw(AssertionError()))
    cache.get("b", lambda key: (key, 4))
    assert list(cache.models) == ["b"] and cache.loads == 2 and cache.evictions == 1

