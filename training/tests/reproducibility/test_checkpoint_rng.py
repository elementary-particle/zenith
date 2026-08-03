from zenith_ppo.seeds import SeedStreams
import random
import numpy as np
import torch


def test_next_draw_matches_after_round_trip():
    streams = SeedStreams(10)
    state = streams.state_dict()
    expected = streams.python_rng("minibatch").random()
    restored = SeedStreams(10)
    restored.load_state_dict(state)
    assert restored.python_rng("minibatch").random() == expected


def test_global_python_numpy_torch_cpu_and_cuda_states_round_trip():
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    states = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
              torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
    expected = (random.random(), np.random.random(), torch.rand(1))
    random.setstate(states[0])
    np.random.set_state(states[1])
    torch.set_rng_state(states[2])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(states[3])
    actual = (random.random(), np.random.random(), torch.rand(1))
    assert expected[0] == actual[0] and expected[1] == actual[1] and torch.equal(expected[2], actual[2])
