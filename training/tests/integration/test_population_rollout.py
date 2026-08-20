import numpy as np
import torch

import riichi
from zenith_ppo.capabilities import configure
from zenith_ppo.config import load
from zenith_ppo.model.factory import build_actor_critic
from zenith_ppo.rollout.native import NativeInferenceRunner
from zenith_ppo.types import MatchLineup


def _model(seed=4):
    configure("cpu-smoke")
    config = load("training/configs/smoke.toml")
    model_config = dict(config.values["model"])
    model_config["context_tokens"] = config.values["encoding"]["context_tokens"]
    torch.manual_seed(seed)
    return build_actor_critic(model_config), int(model_config["context_tokens"])


def test_self_play_lineup_owns_all_four_seats():
    lineup = MatchLineup((0, 1), ("current",) * 4, 0b1111, 3, "self-play")
    assert [seat for seat in range(4) if lineup.learner_mask & (1 << seat)] \
        == [0, 1, 2, 3]


def test_native_self_play_rollout_marks_every_policy_row_eligible():
    model, context_tokens = _model()
    engine = riichi.RolloutEngine(
        2, master_seed=4, num_threads=1,
        context_tokens=context_tokens, token_budget=16_384,
    )
    matches = engine.reset_chunk(2)
    engine.register_lineups(matches, [(0, 0, 0, 0)] * 2, [15, 15])
    chunk = NativeInferenceRunner(
        {0: model}, backend="eager",
        generator=torch.Generator().manual_seed(4),
    ).run_chunk(engine)
    columns = chunk.columns()
    assert set(map(int, columns["policy_slots"])) == {0}
    assert np.asarray(columns["eligibility"], dtype=bool).all()


def test_native_scheduler_coalesces_independent_environment_decisions():
    model, context_tokens = _model(43)
    engine = riichi.RolloutEngine(
        4, master_seed=43, num_threads=1,
        context_tokens=context_tokens, token_budget=65_536,
    )
    matches = engine.reset_chunk(4)
    engine.register_lineups(matches, [(0, 0, 0, 0)] * 4, [15] * 4)
    first = engine.next_request()
    assert 1 <= first.row_count <= 4
    runner = NativeInferenceRunner(
        {0: model}, backend="eager",
        generator=torch.Generator().manual_seed(43),
    )
    selected, old_logp, old_state_values = runner.infer(first)
    engine.submit(
        first.request_id, selected, old_logp, old_state_values,
    )
    chunk = runner.run_chunk(engine)
    stats = runner.stats(engine)
    assert chunk.match_completions == 4
    assert stats.rows / stats.requests > 1.5


def test_native_bot_rows_bypass_models_and_are_never_eligible():
    model, context_tokens = _model(8)
    engine = riichi.RolloutEngine(
        1, master_seed=8, num_threads=1,
        context_tokens=context_tokens, token_budget=16_384,
    )
    matches = engine.reset_chunk(1)
    engine.register_lineups(
        matches, [(0, 9, 0, 9)], [0b0101], bot_policy_slots=[9]
    )
    chunk = NativeInferenceRunner(
        {0: model}, backend="eager",
        generator=torch.Generator().manual_seed(8),
    ).run_chunk(engine)
    columns = chunk.columns()
    assert not (np.asarray(columns["policy_slots"]) == 9).any()
    assert int(engine.metrics()["native_bot_rows"]) > 0


def test_frozen_neural_policy_can_be_greedy_without_consuming_action_rng():
    model, context_tokens = _model(19)
    generator = torch.Generator().manual_seed(19)
    before = generator.get_state().clone()
    engine = riichi.RolloutEngine(
        1, master_seed=19, num_threads=1,
        context_tokens=context_tokens, token_budget=16_384,
    )
    matches = engine.reset_chunk(1)
    engine.register_lineups(matches, [(3, 3, 3, 3)], [0])
    runner = NativeInferenceRunner(
        {3: model}, backend="eager", generator=generator,
        deterministic_policy_slots={3},
    )
    chunk = runner.run_chunk(engine)
    assert torch.equal(generator.get_state(), before)
    assert chunk.row_count > 0
    assert not np.asarray(chunk.columns()["eligibility"], dtype=bool).any()
