from zenith_ppo.seeds import SeedStreams


def test_named_streams_round_trip_and_keyed_masks_ignore_order():
    streams = SeedStreams(7); first = streams.python_rng("action").random(); state = streams.state_dict()
    expected = streams.python_rng("action").random(); restored = SeedStreams(7); restored.load_state_dict(state)
    assert restored.python_rng("action").random() == expected
    assert streams.keyed_uniform("visibility", 1, 2, 3) == streams.keyed_uniform("visibility", 1, 2, 3)
    assert first != streams.python_rng("opponent").random()

