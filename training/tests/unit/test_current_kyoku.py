from types import SimpleNamespace

import numpy as np
import pytest

from zenith_ppo.ppo.current_kyoku import (
    action_family_diagnostics,
    compute,
    rank_explained_variance,
)


def _binding(frame, *, seat=0, environment=0):
    return SimpleNamespace(
        environment_id=environment,
        episode_generation=1,
        frame_id=frame,
        seat=seat,
    )


def _frame(
    frame, boundary, value, *, placement=0, seat=0, environment=0,
    terminal=False, eligible=True,
):
    features = np.zeros(28, dtype=np.float32)
    features[0] = float(boundary)
    return SimpleNamespace(
        binding=_binding(frame, seat=seat, environment=environment),
        checkpoint_id="learner",
        ppo_eligible=eligible,
        terminal=terminal,
        terminal_placement=placement,
        old_boundary_rank_value=value,
        encoded=SimpleNamespace(rank_boundary_features=features),
    )


def _sample(
    frame_index, *, seat=0, environment=0, eligible=True,
    factors=((1,) + (0,) * 14,), selected=0,
):
    return SimpleNamespace(
        binding=_binding(frame_index, seat=seat, environment=environment),
        checkpoint_id="learner",
        ppo_eligible=eligible,
        frame_index=frame_index,
        selected_group=selected,
        encoded=SimpleNamespace(action_factors=np.asarray(factors)),
    )


def test_every_action_receives_its_current_kyoku_potential_change():
    frames = (
        _frame(0, 0, 0.20),
        _frame(1, 0, 0.20),
        _frame(2, 1, 0.50),
        _frame(3, 1, 0.50, terminal=True),
    )
    result = compute((_sample(0), _sample(1), _sample(2)), frames)

    assert result.start_values.tolist() == pytest.approx([0.20, 0.20, 0.50])
    assert result.end_values.tolist() == pytest.approx([0.50, 0.50, 1.00])
    assert result.advantages.tolist() == pytest.approx([0.30, 0.30, 0.50])


def test_actionless_kyoku_does_not_leak_credit_to_an_adjacent_action():
    frames = (
        _frame(0, 0, 0.20),
        _frame(1, 1, 0.40),  # No policy action in this kyoku.
        _frame(2, 2, 0.10),
        _frame(3, 2, 0.10, terminal=True),
    )
    result = compute((_sample(0), _sample(2)), frames)

    assert result.advantages.tolist() == pytest.approx([0.20, 0.90])


def test_terminal_boundary_uses_exact_rank_utility_for_each_seat():
    frames = (
        _frame(0, 0, -0.10, placement=3, seat=3),
        _frame(1, 0, -0.10, placement=3, seat=3, terminal=True),
    )
    result = compute((_sample(0, seat=3),), frames)

    assert result.advantages.tolist() == pytest.approx([-0.90])


def test_policy_advantages_are_standardized_and_ineligible_rows_excluded():
    frames = (
        _frame(0, 0, 0.00),
        _frame(1, 0, 0.00, terminal=True),
        _frame(2, 0, 0.00, placement=3, seat=3, environment=1),
        _frame(
            3, 0, 0.00, placement=3, seat=3, environment=1,
            terminal=True,
        ),
    )
    samples = (
        _sample(0),
        _sample(0, eligible=False),
        _sample(2, seat=3, environment=1),
    )
    result = compute(samples, frames)

    assert result.indices.tolist() == [0, 2]
    assert result.advantages.tolist() == pytest.approx([1.0, -1.0])
    assert float(result.normalized.mean()) == pytest.approx(0.0)
    assert float(result.normalized.std()) == pytest.approx(1.0)


def test_invalid_frame_values_and_trajectory_bindings_are_rejected():
    frames = (
        _frame(0, 0, float("nan")),
        _frame(1, 0, float("nan"), terminal=True),
    )
    with pytest.raises(FloatingPointError, match="non-finite"):
        compute((_sample(0),), frames)

    valid = (
        _frame(0, 0, 0.0),
        _frame(1, 0, 0.0, terminal=True),
    )
    with pytest.raises(ValueError, match="trajectory mismatch"):
        compute((_sample(0, seat=1),), valid)


def test_rank_explained_variance_uses_placement_utility():
    assert rank_explained_variance([1.0, -1.0], [0, 3]) == pytest.approx(1.0)


def test_action_family_diagnostics_report_behavior_only():
    call = (3,) + (0,) * 14
    pass_action = (0,) + (0,) * 14
    samples = (
        _sample(0, factors=(pass_action, call), selected=1),
        _sample(1, factors=(pass_action, call), selected=0),
    )
    advantages = SimpleNamespace(indices=np.asarray((0, 1)))

    metrics = action_family_diagnostics(samples, advantages)
    assert metrics["rollout/call_opportunity_selected_call_rate"] == 0.5
