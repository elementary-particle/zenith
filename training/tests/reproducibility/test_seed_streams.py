from zenith_ppo.seeds import SeedStreams


def test_named_streams_round_trip_and_keyed_masks_ignore_order():
    streams = SeedStreams(7)
    first = streams.python_rng("action").random()
    state = streams.state_dict()
    expected = streams.python_rng("action").random()
    restored = SeedStreams(7)
    restored.load_state_dict(state)
    assert restored.python_rng("action").random() == expected
    assert streams.keyed_uniform("visibility", 1, 2, 3) == streams.keyed_uniform("visibility", 1, 2, 3)
    assert first != streams.python_rng("opponent").random()


def test_actor_epoch_draws_do_not_advance_critic_minibatches():
    streams = SeedStreams(7)
    expected = SeedStreams(7).python_rng("critic_minibatch").random()
    for _ in range(20):
        streams.python_rng("actor_minibatch").random()
    assert streams.python_rng("critic_minibatch").random() == expected
