"""Summed atomic-factor embeddings with active-count scaling."""

from __future__ import annotations

import math
import torch
from torch import nn


class FactorEmbedding(nn.Module):
    def __init__(self, cardinalities, d_model: int, numeric_dim: int = 0):
        super().__init__()
        cardinalities = tuple(int(size) for size in cardinalities)
        offsets, end = [], 0
        for size in cardinalities:
            if size < 1:
                raise ValueError("factor cardinalities must be positive")
            offsets.append(end)
            end += size - 1
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long), persistent=False)
        self.table = nn.Embedding(end + 1, d_model, padding_idx=0)
        self.numeric = nn.Linear(numeric_dim, d_model, bias=False) if numeric_dim else None
        self.norm = nn.RMSNorm(d_model)
        nn.init.normal_(self.table.weight, std=1.0 / math.sqrt(d_model))
        with torch.no_grad():
            self.table.weight[0].zero_()

    def forward(self, factors, numeric=None):
        if factors.shape[-1] != self.offsets.numel():
            raise ValueError("factor width does not match model schema")
        indices = torch.where(factors == 0, 0, factors + self.offsets)
        active = (factors != 0).sum(dim=-1, keepdim=True).clamp_min(1)
        value = self.table(indices).sum(dim=-2) / active.sqrt()
        if self.numeric is not None:
            if numeric is None:
                numeric = torch.zeros(*factors.shape[:-1], self.numeric.in_features,
                                      dtype=torch.float32, device=factors.device)
            value = value + self.numeric(numeric.float())
        return self.norm(value)
