"""Small shared primitives for the sole production actor-critic."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from ..encoding.actions import segment_layout, segmented_sample
from ..encoding.critic import BOUNDARY_RANK_FEATURES
from .transformer import Decoder


class SwiGLUResidual(nn.Module):
    def __init__(self, d_model: int, ffn_dim: int):
        super().__init__()
        self.norm = nn.RMSNorm(d_model)
        self.up = nn.Linear(d_model, 2 * ffn_dim, bias=False)
        self.down = nn.Linear(ffn_dim, d_model, bias=False)

    def forward(self, value):
        gate, content = self.up(self.norm(value)).chunk(2, dim=-1)
        return value + self.down(F.silu(gate) * content)


def _normalization_groups(channels: int) -> int:
    groups = min(4, int(channels))
    while int(channels) % groups:
        groups -= 1
    return groups


class MultiScaleTileBlock(nn.Module):
    """Cheap suit-local filters for adjacent and one-gap tile relations."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(_normalization_groups(channels), channels)
        self.adjacent = nn.Conv2d(
            channels, channels, (1, 3), padding=(0, 1),
            groups=channels, bias=False,
        )
        self.gapped = nn.Conv2d(
            channels, channels, (1, 3), padding=(0, 2), dilation=(1, 2),
            groups=channels, bias=False,
        )
        self.mix = nn.Conv2d(2 * channels, channels, 1, bias=False)
        self.output = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, value):
        normalized = F.silu(self.norm(value))
        local = torch.cat((
            self.adjacent(normalized), self.gapped(normalized),
        ), dim=1)
        return value + self.output(F.silu(self.mix(local)))


class ConcealedShapeEncoder(nn.Module):
    """Encode concealed suit geometry without handcrafted shanten features."""

    def __init__(self, d_model: int, channels: int, blocks: int):
        super().__init__()
        self.stem = nn.Conv2d(1, channels, 1, bias=False)
        self.blocks = nn.ModuleList(
            MultiScaleTileBlock(channels) for _ in range(blocks)
        )
        self.projection = nn.Conv2d(channels, d_model, 1, bias=False)
        self.norm = nn.RMSNorm(d_model)

    def forward(self, planes):
        value = self.stem(planes[:, :1, :3] * 0.25)
        for block in self.blocks:
            value = block(value)
        suits = self.norm(
            self.projection(value).permute(0, 2, 3, 1).reshape(
                planes.shape[0], 27, -1,
            )
        )
        honors = suits.new_zeros((suits.shape[0], 7, suits.shape[-1]))
        return torch.cat((suits, honors), 1)


@dataclass(frozen=True)
class MatchBoundaryCriticOutput:
    rank_order_logits: torch.Tensor
    rank_marginals: torch.Tensor
    rank_logits: torch.Tensor
    rank_probabilities: torch.Tensor
    rank_values: torch.Tensor
    seat_values: torch.Tensor


class PolicyScaffold(nn.Module):
    """Causal trunk and interface utilities used by the verified policy."""

    def __init__(self, config, **_):
        super().__init__()
        d_model = int(config["d_model"])
        self.context_tokens = int(config.get("context_tokens", 4096))
        self.policy_temperature = float(config.get("policy_temperature", 1.0))
        if self.policy_temperature <= 0:
            raise ValueError("policy temperature must be positive")
        self.token_tile_merge_norm = nn.RMSNorm(d_model)
        self.backbone = Decoder(
            layers=int(config["layers"]), d_model=d_model,
            query_heads=int(config["query_heads"]),
            kv_heads=int(config["kv_heads"]),
            head_dim=int(config["head_dim"]),
            ffn_dim=int(config["ffn_dim"]),
            context_tokens=self.context_tokens,
        )

    def actor_parameters(self):
        return (parameter for _, parameter in self.actor_named_parameters())

    @staticmethod
    def _valid(lengths, maximum, device):
        return torch.arange(maximum, device=device)[None] < lengths[:, None]

    @staticmethod
    def _padded_actions(action_factors, action_lengths, action_offsets):
        if action_factors.ndim == 3:
            if action_lengths is None:
                _, action_lengths, _ = segment_layout(
                    action_offsets,
                    total=sum(
                        int(end) - int(start)
                        for start, end in zip(
                            action_offsets[:-1], action_offsets[1:], strict=True,
                        )
                    ),
                    device=action_factors.device,
                )
            return action_factors, action_lengths
        if action_factors.ndim != 2:
            raise ValueError("action factors must be padded [batch, actions, 15]")
        offsets, lengths, _ = segment_layout(
            action_offsets, total=int(action_factors.shape[0]),
            device=action_factors.device,
        )
        maximum = int(lengths.max())
        padded = action_factors.new_zeros(lengths.numel(), maximum, 15)
        for row, (start, end) in enumerate(
            zip(offsets[:-1], offsets[1:], strict=True)
        ):
            padded[row, :int(end - start)] = action_factors[start:end]
        return padded, lengths

    @staticmethod
    def _strategic_features(rank_boundary_features, decision_seats):
        boundary = rank_boundary_features.float()
        if boundary.shape[-1] != BOUNDARY_RANK_FEATURES:
            raise ValueError(
                f"rank boundary features must have width {BOUNDARY_RANK_FEATURES}"
            )
        seats = decision_seats.long()
        offsets = torch.arange(4, device=boundary.device)[None]
        score_indices = (seats[:, None] + offsets) % 4
        scores = boundary[:, :4].gather(1, score_indices)
        dealer = F.one_hot((-seats) % 4, num_classes=4).to(boundary.dtype)
        return torch.cat((scores, dealer, boundary[:, 8:]), -1)

    def forward_critic(
        self, decision_seats=None, rank_boundary_features=None, **_,
    ):
        return self.match_boundary_critic(
            rank_boundary_features, decision_seats,
        )

    @torch.no_grad()
    def sample(self, output, action_offsets, *, generator=None, deterministic=False):
        return segmented_sample(
            output.log_probabilities,
            action_offsets,
            generator=generator,
            deterministic=deterministic,
        )
