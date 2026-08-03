from types import SimpleNamespace

import torch

from zenith_ppo.cli.evaluate import _play_games
from zenith_ppo.evaluation.runner import series_requests


class _UniformPolicy:
    def __init__(self, log_probability=0.0):
        self.batch_sizes = []
        self.log_probability = float(log_probability)

    def forward_actor(self, **inputs):
        self.batch_sizes.append(int(inputs["lengths"].numel()))
        action_count = int(inputs["action_offsets"][-1])
        return SimpleNamespace(
            log_probabilities=torch.full(
                (action_count,), self.log_probability,
                device=inputs["action_offsets"].device,
            )
        )


def test_batched_evaluator_is_reproducible_and_batches_neural_rows():
    requests = series_requests(("uniform",) * 4, (901, 902))
    first_policy = _UniformPolicy()
    first = _play_games(
        {"uniform": first_policy}, {}, requests,
        device="cpu", token_budget=16_384,
    )
    second_policy = _UniformPolicy()
    second = _play_games(
        {"uniform": second_policy}, {}, requests,
        device="cpu", token_budget=16_384,
    )

    assert first == second
    assert all(not isinstance(result, BaseException) for result in first)
    assert all(result["valid"] for result in first)
    assert all(result["gameplay_counts"]["uniform"]["player_matches"] == 4
               for result in first)
    assert all(result["gameplay_counts"]["uniform"]["player_kyoku"] > 0
               for result in first)
    assert max(first_policy.batch_sizes) > 1


def test_batched_evaluator_supports_greedy_action_selection():
    requests = series_requests(("uniform",) * 4, (903,))
    results = _play_games(
        {"uniform": _UniformPolicy()}, {}, requests,
        device="cpu", token_budget=16_384, greedy=True,
    )

    assert all(not isinstance(result, BaseException) for result in results)
    assert all(result["valid"] for result in results)


def test_batched_evaluator_supports_policy_selective_greedy_actions(
    monkeypatch,
):
    from zenith_ppo.encoding import actions

    calls = []
    segmented_sample = actions.segmented_sample

    def record_decoding(log_probabilities, offsets, **kwargs):
        calls.append((
            float(log_probabilities[0]), bool(kwargs.get("deterministic")),
        ))
        return segmented_sample(log_probabilities, offsets, **kwargs)

    monkeypatch.setattr(actions, "segmented_sample", record_decoding)
    requests = series_requests(("sampled", "greedy", "greedy", "greedy"), (904,))
    results = _play_games(
        {
            "sampled": _UniformPolicy(log_probability=-1.0),
            "greedy": _UniformPolicy(log_probability=-2.0),
        }, {}, requests,
        device="cpu", token_budget=16_384,
        greedy_checkpoint_ids={"greedy"},
    )

    assert all(not isinstance(result, BaseException) for result in results)
    assert all(result["valid"] for result in results)
    assert all(set(result["gameplay_counts"]) == {"sampled", "greedy"}
               for result in results)
    assert calls
    assert all(deterministic == (log_probability == -2.0)
               for log_probability, deterministic in calls)
