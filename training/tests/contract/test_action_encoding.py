from zenith_ppo.encoding.actions import encode_actions


def test_equivalent_native_rows_group_to_canonical_representative():
    row = {"kind": 1, "primary_tile_type": 0, "source_seat": 255, "tiles": [0], "aux": 0, "flags": 0}
    encoded = encode_actions([row, dict(row)], observer=0)
    assert encoded.representatives == (0,)
    assert encoded.members == ((0, 1),)


def test_u16_action_metadata_uses_lossless_byte_factors():
    row = {"kind": 7, "primary_tile_type": 4, "source_seat": 255,
           "tiles": [16], "aux": 0x1234, "flags": 0xABCD}
    factors = encode_actions([row], observer=0).factors[0]
    assert len(factors) == 15
    assert factors[-4:] == (0x34, 0x12, 0xCD, 0xAB)
