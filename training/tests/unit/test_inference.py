from types import SimpleNamespace

from zenith_ppo.inference import ConservativeBot


def _action(kind, tile=None):
    return SimpleNamespace(kind=kind, tiles=() if tile is None else (tile,))


def test_conservative_bot_uses_public_genbutsu_state():
    encoded = SimpleNamespace(
        binding=SimpleNamespace(seat=0),
        action_representatives=(0, 1, 2),
        native_candidates=(_action(1, 4), _action(1, 8), _action(2, 8)),
    )
    state = SimpleNamespace(
        seat_flags=(0, 2, 0, 0),
        rivers=(SimpleNamespace(seat=1, tile=8),),
    )

    assert ConservativeBot().select_group(encoded, state=state) == 1


def test_conservative_bot_takes_wins_and_passes_reactions():
    win = SimpleNamespace(
        binding=SimpleNamespace(seat=0),
        action_representatives=(0, 1),
        native_candidates=(_action(0), _action(8)),
    )
    reaction = SimpleNamespace(
        binding=SimpleNamespace(seat=0),
        action_representatives=(0, 1),
        native_candidates=(_action(3), _action(0)),
    )

    assert ConservativeBot().select_group(win) == 1
    assert ConservativeBot().select_group(reaction) == 1
