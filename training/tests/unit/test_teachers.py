from types import SimpleNamespace

import pytest
import torch

from zenith_ppo.inference import ConservativeBot
from zenith_ppo.teachers import (
    DiscardTarget,
    ReactionTarget,
    RiichiTarget,
    TeacherTargets,
    _guaranteed_yaku,
    _reaction_target,
    _riichi_target,
    auxiliary_losses,
    coefficients,
    pack_targets,
    rollout_metrics,
)


def action(kind, tile=None, source=None):
    return SimpleNamespace(
        kind=kind, tiles=() if tile is None else (tile,), source_seat=source
    )


def test_teacher_coefficients_reach_zero_at_twenty_percent():
    config = {
        "discard_coefficient": .10, "reaction_coefficient": .03,
        "riichi_coefficient": .02, "reaction_entropy_coefficient": .005,
        "anneal_end": .20,
    }
    assert coefficients(config, 0)["discard"] == .10
    assert coefficients(config, .10)["discard"] == pytest.approx(.05)
    assert coefficients(config, .20) == {
        "discard": 0, "reaction": 0, "riichi": 0, "reaction_entropy": 0,
    }


def test_default_teacher_coefficients_reach_zero_at_fifty_percent():
    assert coefficients(None, .25)["discard"] == pytest.approx(.05)
    assert coefficients(None, .50) == {
        "discard": 0, "reaction": 0, "riichi": 0, "reaction_entropy": 0,
    }


def test_riichi_pairing_aggregates_same_tile_categories():
    target = _riichi_target(
        (action(1, 16), action(1, 17), action(2, 16), action(1, 20)),
        {"dama_target": .2, "riichi_target": .8},
    )
    assert target.pairs == (((0,), (2,)),)
    assert target.probabilities == (.2, .8)


def test_self_draw_win_gets_unconditional_reaction_target_without_pass():
    actions = (action(1, 4), action(1, 8), action(9, 8))
    target = _reaction_target(
        SimpleNamespace(), SimpleNamespace(), actions, {}
    )

    assert target is not None
    assert target.probabilities == (0.0, 0.0, 1.0)
    assert target.pass_group is None

    logits = torch.tensor([2.0, 1.0, -1.0], requires_grad=True)
    result = auxiliary_losses(
        logits, logits.log_softmax(0), torch.tensor([0, 3]),
        (TeacherTargets(reaction=target),),
        {"discard": 0.0, "reaction": 1.0, "riichi": 0.0,
         "reaction_entropy": 0.0},
    )
    result.total.backward()
    assert logits.grad[2] < 0
    assert logits.grad[:2].gt(0).all()


def test_zero_applicable_rows_produce_differentiable_zero_losses():
    logits = torch.tensor([1.0, -1.0], requires_grad=True)
    logp = logits.log_softmax(0)
    result = auxiliary_losses(
        logits, logp, torch.tensor([0, 2]), (TeacherTargets(),),
        {"discard": .1, "reaction": .03, "riichi": .02, "reaction_entropy": .005},
    )
    assert result.total.item() == 0
    result.total.backward()
    assert torch.equal(logits.grad, torch.zeros_like(logits))


def test_vectorized_auxiliary_losses_match_direct_formulas():
    # The fourth action is an unrelated category and must not affect discard CE.
    logits = torch.tensor([1.0, 0.0, -1.0, 5.0, 0.2, -0.2], requires_grad=True)
    full_first = logits[:4].log_softmax(0)
    second = logits[4:].log_softmax(0)
    logp = torch.cat((full_first, second))
    targets = (
        TeacherTargets(
            discard=DiscardTarget(
                ((0, 1), (2,)), (.25, .75), (1, 2), (8, 3), (0.0, 4.0),
                (0, 0, 1),
            ),
            riichi=RiichiTarget((((0,), (1,)),), (.2, .8)),
        ),
        TeacherTargets(reaction=ReactionTarget((.9, .1), 0, None, (False, False))),
    )
    packed = pack_targets(targets, (4, 2))
    coefficients = {
        "discard": .1, "reaction": .03, "riichi": .02,
        "reaction_entropy": .005,
    }
    result = auxiliary_losses(
        logits, logp, torch.tensor([0, 4, 6]), packed, coefficients
    )
    discard_aggregated = torch.stack((
        torch.logsumexp(logits[:2], 0), logits[2]
    ))
    expected_discard = -(
        torch.tensor((.25, .75)) * discard_aggregated.log_softmax(0)
    ).sum()
    expected_reaction = -(torch.tensor((.9, .1)) * second).sum()
    expected_riichi = -(
        torch.tensor((.2, .8)) * logits[:2].log_softmax(0)
    ).sum()
    expected_entropy = -(second.exp() * second).sum()
    expected = (
        .1 * expected_discard + .03 * expected_reaction + .02 * expected_riichi
        - .005 * expected_entropy
    )
    assert result.discard_loss.detach().item() == pytest.approx(expected_discard.detach().item())
    assert result.reaction_loss.detach().item() == pytest.approx(expected_reaction.detach().item())
    assert result.riichi_loss.detach().item() == pytest.approx(expected_riichi.detach().item())
    assert result.total.detach().item() == pytest.approx(expected.detach().item())


def test_supported_yaku_is_conservative():
    state = SimpleNamespace(dealer=0, round_wind=0, melds=())
    decision = SimpleNamespace(seat=0)
    east_pon = SimpleNamespace(kind=4, tiles=(108, 109, 110))
    simples = [0] * 34
    simples[1] = 2
    assert _guaranteed_yaku(state, decision, east_pon, simples, ("yakuhai",))
    terminal_call = SimpleNamespace(kind=3, tiles=(0, 4, 8))
    assert not _guaranteed_yaku(
        state, decision, terminal_call, simples, ("open_tanyao",)
    )


def test_bot_uses_genbutsu_then_declares_riichi_for_selected_tile():
    actions = (action(1, 4), action(1, 8), action(2, 8))
    encoded = SimpleNamespace(
        binding=SimpleNamespace(seat=0),
        action_representatives=(0, 1, 2),
        native_actions=actions,
        teachers=TeacherTargets(discard=DiscardTarget(
            ((0,), (1, 2)), (.8, .2), (1, 1), (10, 5), (0.0, .1), (0, 1, 1)
        )),
    )
    state = SimpleNamespace(
        seat_flags=(0, 2, 0, 0),
        rivers=(SimpleNamespace(seat=1, tile=8),),
    )
    assert ConservativeBot().select_group(encoded, state=state) == 2


def test_target_encoding_does_not_read_privileged_hidden_state():
    import riichi
    from zenith_ppo.encoding.actions import encode_native_actions
    from zenith_ppo.env.adapter import EnvAdapter
    from zenith_ppo.teachers import build_targets

    env = riichi.Env(1, master_seed=91, num_threads=1, privileged=True)
    adapter = EnvAdapter(env)
    batch = adapter.reset([0])
    state = batch.transition.states[0]
    decision = state.decisions[0]

    class OrdinaryView:
        def __getattr__(self, name):
            if name == "hidden":
                raise AssertionError("teacher attempted to consume privileged critic data")
            return getattr(state, name)

    targets = build_targets(
        OrdinaryView(), decision,
        encode_native_actions(decision.actions, observer=decision.seat),
    )
    env.close()
    assert targets.discard is not None


def test_action_frequencies_use_completed_kyoku_as_explicit_denominator():
    def sample(kind, *, reaction=None, riichi=None):
        encoded = SimpleNamespace(
            teachers=TeacherTargets(reaction=reaction, riichi=riichi),
            native_actions=(action(kind),),
            action_representatives=(0,),
        )
        return SimpleNamespace(ppo_eligible=True, encoded=encoded, selected_group=0)

    call = sample(3, reaction=ReactionTarget((.75,), 0, 0, (True,)))
    declaration = sample(
        2, riichi=RiichiTarget((((0,), (0,)),), (.2, .8))
    )

    unavailable = rollout_metrics((call, declaration), 0)
    assert "teacher/reaction_calls_per_kyoku" not in unavailable
    assert "teacher/riichi_declarations_per_kyoku" not in unavailable

    metrics = rollout_metrics((call, declaration), 2)
    assert metrics["teacher/reaction_calls_per_kyoku"] == 0.5
    assert metrics["teacher/reaction_improving_calls_per_kyoku"] == 0.5
    assert metrics["teacher/riichi_legal_opportunities_per_kyoku"] == 0.5
    assert metrics["teacher/riichi_declarations_per_kyoku"] == 0.5


def test_rollout_discard_agreement_accepts_all_tied_best_options():
    target = DiscardTarget(
        ((0,), (1,), (2,)), (.45, .45, .10),
        (1, 1, 2), (8, 8, 2), (0.0, 0.0, 4.0), (0, 1, 2),
    )
    encoded = SimpleNamespace(
        teachers=TeacherTargets(discard=target),
        native_actions=(action(1, 4), action(1, 8), action(1, 12)),
        action_representatives=(0, 1, 2),
    )
    sample = SimpleNamespace(
        ppo_eligible=True, encoded=encoded, selected_group=1
    )

    assert rollout_metrics((sample,), 0)["teacher/discard_agreement"] == 1.0


def test_riichi_conversion_counts_decisions_not_discard_pairs():
    target = RiichiTarget(
        (((0,), (1,)), ((2,), (3,)), ((4,), (5,))), (.2, .8)
    )
    encoded = SimpleNamespace(
        teachers=TeacherTargets(riichi=target),
        native_actions=tuple(action(kind) for kind in (1, 2, 1, 2, 1, 2)),
        action_representatives=tuple(range(6)),
    )
    sample = SimpleNamespace(
        ppo_eligible=True, encoded=encoded, selected_group=3
    )

    metrics = rollout_metrics((sample,), 2)
    assert metrics["teacher/riichi_conversion_rate"] == 1.0
    assert metrics["teacher/riichi_legal_opportunities_per_kyoku"] == 0.5
