import torch
import pytest
from zenith_ppo.model.transformer import Decoder


def test_eager_and_sdpa_match():
    torch.manual_seed(1); model = Decoder(layers=1, d_model=32, query_heads=2, kv_heads=1, head_dim=16, ffn_dim=64)
    values = torch.randn(2, 7, 32)
    assert torch.allclose(model(values, backend="eager"), model(values, backend="sdpa"), atol=2e-5, rtol=2e-5)


def test_bucketed_variable_lengths_match_individual_sequences_and_zero_padding():
    torch.manual_seed(2)
    model = Decoder(layers=1, d_model=32, query_heads=2, kv_heads=1, head_dim=16, ffn_dim=64)
    values = torch.randn(2, 7, 32)
    lengths = torch.tensor([7, 4])
    packed = model(values, lengths=lengths, backend="sdpa")
    individual = model(values[1:2, :4], backend="sdpa")
    assert torch.allclose(packed[1, :4], individual[0], atol=2e-5, rtol=2e-5)
    assert torch.count_nonzero(packed[1, 4:]) == 0
    with pytest.raises(ValueError, match="backend"):
        model(values, backend="unknown")
