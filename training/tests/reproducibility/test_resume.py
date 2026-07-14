from zenith_ppo.seeds import SeedStreams
import torch


def test_restored_named_stream_reproduces_next_rollout_draw():
    first = SeedStreams(11); state = first.state_dict(); expected = first.python_rng("opponent").random()
    second = SeedStreams(11); second.load_state_dict(state)
    assert second.python_rng("opponent").random() == expected


def test_model_optimizer_counter_population_and_metric_cursor_restore():
    model = torch.nn.Linear(2, 1); optimizer = torch.optim.AdamW(model.parameters())
    inputs = torch.ones(1, 2); loss = model(inputs).sum(); loss.backward(); optimizer.step()
    state = {"model": {k: v.clone() for k, v in model.state_dict().items()},
             "optimizer": optimizer.state_dict(), "counter": 4, "pool": ("a", "b"), "metric": 12}
    restored = torch.nn.Linear(2, 1); restored.load_state_dict(state["model"])
    assert torch.equal(model(inputs), restored(inputs)) and state["counter"] == 4
    assert state["pool"] == ("a", "b") and state["metric"] == 12
