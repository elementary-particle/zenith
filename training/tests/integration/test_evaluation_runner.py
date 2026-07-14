from zenith_ppo.evaluation.runner import cyclic_lineups


def test_cyclic_block_balances_every_initial_seat():
    blocks = cyclic_lineups(("a", "b", "c", "d"))
    assert all(sorted(block[seat] for block in blocks) == ["a", "b", "c", "d"] for seat in range(4))

