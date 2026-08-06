from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import torch

from zenith_ppo.encoding.packing import EncodedActionSpace
from zenith_ppo.encoding.schema import Segment, TokenKind, numeric_value_features
from zenith_ppo.model.actor_critic import ActionMemoryCore

sys.path.insert(0, str(Path(__file__).parent))
from audit import (
    SummaryActionMemoryCore,
    event_prefix_encoded,
    event_prefix_tensors,
    stable_boundary_encoded,
    stable_boundary_tensors,
)


def _encoded():
    factors = np.asarray([
        (Segment.MATCH_STATE, TokenKind.SCORE, 0, 0, 0, 0, 0, 0, 0, 0),
        (Segment.MATCH_SUMMARY, TokenKind.QUERY, 0, 0, 0, 0, 0, 0, 0, 0),
        (Segment.EVENT, TokenKind.EVENT, 2, 0, 0, 0, 0, 0, 0, 0),
        (Segment.EVENT, TokenKind.EVENT, 4, 0, 0, 0, 0, 0, 0, 0),
        (Segment.KYOKU_STATE, TokenKind.COUNTER, 0, 0, 0, 0, 0, 0, 0, 0),
        (Segment.KYOKU_SUMMARY, TokenKind.QUERY, 0, 0, 0, 0, 0, 0, 0, 0),
        (Segment.ACTOR_QUERY, TokenKind.QUERY, 0, 0, 0, 0, 0, 0, 0, 0),
    ], dtype=np.uint8)
    return EncodedActionSpace(
        binding=None,
        token_factors=factors,
        token_numeric=np.arange(56, dtype=np.float32).reshape(7, 8),
        actor_query_index=6,
        rank_boundary_features=np.zeros(28, dtype=np.float32),
        decision_seat=0,
        action_factors=np.zeros((1, 15), dtype=np.uint8),
        action_representatives=(0,),
        action_members=((0,),),
        native_candidates=(),
    )


def test_event_prefix_is_stable_and_keeps_query_last():
    source = _encoded()
    result = event_prefix_encoded(source)
    assert result.actor_query_index == 6
    assert result.token_factors[:2, 0].tolist() == [Segment.EVENT, Segment.EVENT]
    assert result.token_factors[:2, 1].tolist() == [TokenKind.EVENT, TokenKind.EVENT]
    assert result.token_factors[4, 1] != TokenKind.EVENT
    assert result.token_factors[-1, 0] == Segment.ACTOR_QUERY
    assert result.token_numeric[:2, 0].tolist() == [16.0, 24.0]

    factors, numeric, queries = event_prefix_tensors(
        torch.as_tensor(source.token_factors)[None],
        torch.as_tensor(source.token_numeric)[None],
        torch.tensor([len(source.token_factors)]),
        torch.tensor([source.actor_query_index]),
    )
    assert np.array_equal(factors[0].numpy(), result.token_factors)
    assert np.array_equal(numeric[0].numpy(), result.token_numeric)
    assert queries.tolist() == [result.actor_query_index]


def test_summary_memory_has_fixed_38_tokens_and_same_parameters():
    baseline = ActionMemoryCore(
        16, 1, 32, 1, concealed_shape_channels=4,
        concealed_shape_blocks=1,
    )
    summary = SummaryActionMemoryCore(
        16, 1, 32, 1, concealed_shape_channels=4,
        concealed_shape_blocks=1,
    )
    assert set(baseline.state_dict()) == set(summary.state_dict())
    factors = torch.zeros((2, 12, 10), dtype=torch.long)
    factors[:, (2, 7, 11), 1] = int(TokenKind.QUERY)
    history = torch.randn(2, 12, 16)
    state, memory, valid = summary.memory(
        torch.randn(2, 16), torch.randn(2, 28), factors,
        torch.tensor([12, 12]), history, torch.ones(2, 12, dtype=torch.bool),
        torch.randn(34, 16),
    )
    assert state.shape == (2, 16)
    assert memory.shape == (2, 38, 16)
    assert valid.shape == (2, 38)
    assert bool(valid.all())


def test_stable_boundary_reconstructs_match_score_in_observer_order():
    source = _encoded()
    factors = source.token_factors.copy()
    factors[0, 1] = int(TokenKind.SCORE)
    factors[0, 3] = 1  # observer's own score
    boundary = np.zeros(28, dtype=np.float32)
    boundary[:4] = np.asarray([1.0, 2.0, 3.0, 4.0])
    boundary[8 + 6] = 1.0
    source = replace(
        source,
        token_factors=factors,
        rank_boundary_features=boundary,
        decision_seat=2,
    )
    result = stable_boundary_encoded(source)
    assert np.allclose(
        result.token_numeric[0], numeric_value_features(1, 72_000.0)
    )
    assert np.array_equal(result.token_factors, source.token_factors)
    tensor_numeric = stable_boundary_tensors(
        torch.as_tensor(source.token_factors)[None],
        torch.as_tensor(source.token_numeric)[None],
        torch.as_tensor(source.rank_boundary_features)[None],
        torch.tensor([source.decision_seat]),
    )
    assert np.allclose(
        tensor_numeric[0, 0].numpy(), result.token_numeric[0], atol=1e-5
    )
