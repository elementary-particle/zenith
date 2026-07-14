from zenith_ppo.encoding.state import encode_state


def test_masked_values_and_cardinality_do_not_change_actor_tokens():
    decision = {"concealed_counts": [0] * 34}
    frame_a = {"scores": [25000] * 4, "priv_concealed_tile_ids": [[0] * 14 for _ in range(4)]}
    frame_b = {"scores": [25000] * 4, "priv_concealed_tile_ids": [[135] * 14 for _ in range(4)]}
    assert encode_state(frame_a, decision, observer=0) == encode_state(
        frame_b, decision, observer=0
    )
