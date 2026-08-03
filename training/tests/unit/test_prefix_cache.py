from zenith_ppo.rollout.prefix_cache import PrefixCache


def test_cache_reuse_eviction_and_invalidation():
    cache = PrefixCache(5)
    cache.put(("a", 1), 2, object(), 3)
    assert cache.get(("a", 1), 3).event_end == 2
    cache.put(("b", 1), 1, object(), 3)
    assert cache.get(("a", 1), 3) is None
    cache.invalidate()
    assert cache.bytes == 0
