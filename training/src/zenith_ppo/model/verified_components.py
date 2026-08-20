"""Shared components of the single verified Mahjong policy."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


SCORE_DELTA_ATOMS = (
    -24_000, -12_000, -8_000, -4_000, -2_000,
    0,
    2_000, 4_000, 8_000, 12_000, 24_000,
)


@dataclass(frozen=True)
class VerifiedActorOutput:
    logits: torch.Tensor
    log_probabilities: torch.Tensor
    entropy: torch.Tensor | None
    actor_states: torch.Tensor
    action_states: torch.Tensor
    hand_outcome_logits: torch.Tensor | None
    score_delta_logits: torch.Tensor | None
    placement_logits: torch.Tensor | None
    state_values: torch.Tensor | None


def tile_player_relations():
    """Rule-level relations for the 34 tile nodes."""
    result = torch.empty(34, 34, dtype=torch.long)
    for left in range(34):
        for right in range(34):
            if left == right:
                relation = 0
            elif left < 27 and right < 27:
                left_suit, left_rank = divmod(left, 9)
                right_suit, right_rank = divmod(right, 9)
                if left_suit == right_suit:
                    relation = 1 + right_rank - left_rank + 8
                elif left_rank == right_rank:
                    relation = 18
                else:
                    relation = 19
            elif left >= 27 and right >= 27:
                relation = 20
            else:
                relation = 21
            result[left, right] = relation
    return result


class StrategicMemory(nn.Module):
    """Four actor-relative player tokens and one match-horizon token."""

    def __init__(self, d_model):
        super().__init__()
        self.player = nn.Sequential(
            nn.Linear(6, d_model, bias=False), nn.SiLU(), nn.RMSNorm(d_model),
        )
        self.match = nn.Sequential(
            nn.Linear(20, d_model, bias=False), nn.SiLU(), nn.RMSNorm(d_model),
        )

    def forward(self, features):
        features = features.float()
        batch = features.shape[0]
        relative = torch.eye(4, device=features.device)[None].expand(
            batch, -1, -1,
        )
        player_features = torch.cat((
            features[:, :4, None],
            features[:, 4:8, None],
            relative,
        ), -1)
        players = self.player(player_features)
        match = self.match(features[:, 8:])[:, None]
        return torch.cat((players, match), 1)


class CandidateBlock(nn.Module):
    """Permutation-equivariant legal-action reasoning over public memory."""

    def __init__(self, d_model, heads, ffn_dim):
        super().__init__()
        from .actor_critic import SwiGLUResidual

        self.query_norm = nn.RMSNorm(d_model)
        self.memory_norm = nn.RMSNorm(d_model)
        self.memory_attention = nn.MultiheadAttention(
            d_model, heads, batch_first=True, bias=False,
        )
        self.set_norm = nn.RMSNorm(d_model)
        self.set_attention = nn.MultiheadAttention(
            d_model, heads, batch_first=True, bias=False,
        )
        self.ffn = SwiGLUResidual(d_model, ffn_dim)

    def forward(self, actions, action_valid, memory, memory_valid):
        residual, _ = self.memory_attention(
            self.query_norm(actions), self.memory_norm(memory), memory,
            key_padding_mask=~memory_valid, need_weights=False,
        )
        actions = actions + residual
        residual, _ = self.set_attention(
            self.set_norm(actions), actions, actions,
            key_padding_mask=~action_valid, need_weights=False,
        )
        actions = self.ffn(actions + residual)
        return torch.where(
            action_valid[..., None], actions, torch.zeros_like(actions),
        )
