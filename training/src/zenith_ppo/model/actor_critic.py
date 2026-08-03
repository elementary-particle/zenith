"""Shared tile/action-memory policy with a match-boundary rank critic."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations

import torch
from torch import nn
from torch.nn import functional as F

from ..encoding.actions import (
    segment_layout,
    segmented_entropy,
    segmented_log_softmax,
    segmented_sample,
)
from ..encoding.critic import BOUNDARY_RANK_FEATURES
from .embeddings import FactorEmbedding
from .tile import TILE_PLANE_CHANNELS, current_public_tile_count_planes
from .transformer import Decoder


@dataclass(frozen=True)
class ActorOutput:
    logits: torch.Tensor
    log_probabilities: torch.Tensor
    entropy: torch.Tensor | None
    actor_states: torch.Tensor
    action_states: torch.Tensor


@dataclass(frozen=True)
class ActorCriticOutput:
    logits: torch.Tensor
    log_probabilities: torch.Tensor
    rank_logits: torch.Tensor
    rank_probabilities: torch.Tensor
    rank_values: torch.Tensor
    entropy: torch.Tensor


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


class CanonicalTileEmbedding(nn.Module):
    """One tile identity space shared by history, hand shape, and actions."""

    def __init__(self, d_model: int):
        super().__init__()
        self.table = nn.Embedding(35, d_model, padding_idx=34)
        self.norm = nn.RMSNorm(d_model)
        nn.init.normal_(self.table.weight, std=d_model ** -0.5)
        with torch.no_grad():
            self.table.weight[34].zero_()

    def _embed(self, indices, valid):
        indices = torch.where(valid, indices, torch.full_like(indices, 34))
        return self.norm(self.table(indices.long()))

    def tokens(self, factors):
        suit, rank = factors[..., 4].long(), factors[..., 5].long()
        valid = suit.ge(1) & suit.le(4) & rank.ge(1) \
            & torch.where(suit.eq(4), rank.le(7), rank.le(9))
        indices = torch.where(
            suit.eq(4), 27 + rank - 1, (suit - 1) * 9 + rank - 1
        )
        return self._embed(indices, valid)

    def actions(self, factors):
        primary = factors[..., 1].long()
        return self._embed(primary, primary.ge(0) & primary.lt(34))

    def action_tiles(self, factors):
        semantic = factors[..., 7:11].long()
        valid = semantic.gt(0) & semantic.lt(136)
        tile_types = semantic.div(4, rounding_mode="floor")
        embedded = self._embed(tile_types, valid)
        count = valid.sum(-1, keepdim=True).clamp_min(1)
        return (embedded * valid[..., None]).sum(-2) / count.sqrt()

    def all_tiles(self):
        indices = torch.arange(34, device=self.table.weight.device)
        return self.norm(self.table(indices))


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
        # Numbered suits have geometry; honors retain count and identity only.
        value = self.stem(planes[:, :1, :3] * 0.25)
        for block in self.blocks:
            value = block(value)
        suits = self.norm(
            self.projection(value).permute(0, 2, 3, 1).reshape(
                planes.shape[0], 27, -1
            )
        )
        honors = suits.new_zeros((suits.shape[0], 7, suits.shape[-1]))
        return torch.cat((suits, honors), 1)


class ActionMemoryBlock(nn.Module):
    """One standard cross-attention, candidate-attention, and MLP block."""

    def __init__(self, d_model: int, heads: int, ffn_dim: int):
        super().__init__()
        self.memory_norm = nn.RMSNorm(d_model)
        self.memory_attention = nn.MultiheadAttention(
            d_model, heads, batch_first=True, bias=False
        )
        self.candidate_norm = nn.RMSNorm(d_model)
        self.candidate_attention = nn.MultiheadAttention(
            d_model, heads, batch_first=True, bias=False
        )
        self.ffn = SwiGLUResidual(d_model, ffn_dim)

    def forward(self, actions, memory, memory_valid, action_valid):
        residual, _ = self.memory_attention(
            self.memory_norm(actions), memory, memory,
            key_padding_mask=~memory_valid, need_weights=False,
        )
        actions = actions + residual
        residual, _ = self.candidate_attention(
            self.candidate_norm(actions), actions, actions,
            key_padding_mask=~action_valid, need_weights=False,
        )
        actions = self.ffn(actions + residual)
        return torch.where(
            action_valid[..., None], actions, torch.zeros_like(actions)
        )


class ActionMemoryCore(nn.Module):
    """Repeatedly reason from every legal action over one public memory."""

    def __init__(
        self, d_model: int, heads: int, ffn_dim: int, layers: int, *,
        concealed_shape_channels: int, concealed_shape_blocks: int,
    ):
        super().__init__()
        if layers < 1:
            raise ValueError("action-memory reasoning needs at least one layer")
        self.tile_counts = nn.Sequential(
            nn.Linear(TILE_PLANE_CHANNELS, d_model, bias=False),
            nn.RMSNorm(d_model),
        )
        self.tile_input_norm = nn.RMSNorm(d_model)
        self.concealed_shape = ConcealedShapeEncoder(
            d_model, concealed_shape_channels, concealed_shape_blocks,
        )
        self.strategy = nn.Sequential(
            nn.Linear(28, d_model, bias=False), nn.SiLU(), nn.RMSNorm(d_model),
        )
        self.state = nn.Sequential(
            nn.RMSNorm(2 * d_model),
            nn.Linear(2 * d_model, d_model, bias=False),
            nn.SiLU(),
            nn.RMSNorm(d_model),
        )
        self.blocks = nn.ModuleList(
            ActionMemoryBlock(d_model, heads, ffn_dim)
            for _ in range(layers)
        )

    def memory(
        self, actor_state, strategic_features, token_factors, public_lengths,
        history, history_valid, tile_identity,
    ):
        planes = current_public_tile_count_planes(token_factors, public_lengths)
        counts = planes.flatten(2)[:, :, :34].transpose(1, 2) * 0.25
        tiles = self.tile_counts(counts).to(actor_state.dtype)
        tiles = tiles + tile_identity[None].to(actor_state.dtype)
        tiles = tiles + self.concealed_shape(planes).to(actor_state.dtype)
        tiles = self.tile_input_norm(tiles)
        strategy = self.strategy(strategic_features.float()).to(actor_state.dtype)
        state = self.state(torch.cat((actor_state, strategy), -1))
        memory = torch.cat((state[:, None], strategy[:, None], tiles, history), 1)
        prefix_valid = torch.ones(
            history.shape[0], 36, dtype=torch.bool, device=history.device
        )
        return state, memory, torch.cat((prefix_valid, history_valid), 1)

    def forward(self, actions, memory, memory_valid, action_valid):
        for block in self.blocks:
            actions = block(actions, memory, memory_valid, action_valid)
        return actions


@dataclass(frozen=True)
class MatchBoundaryCriticOutput:
    rank_order_logits: torch.Tensor
    rank_marginals: torch.Tensor
    rank_logits: torch.Tensor
    rank_probabilities: torch.Tensor
    rank_values: torch.Tensor
    seat_values: torch.Tensor


class MatchBoundaryCritic(nn.Module):
    """Predict final rank order from start-of-kyoku match state alone."""

    def __init__(self, width: int = 64):
        super().__init__()
        self.rank_tower = nn.Sequential(
            nn.Linear(BOUNDARY_RANK_FEATURES, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.RMSNorm(width),
            nn.Linear(width, 24),
        )
        orders = tuple(permutations(range(4)))
        order_to_marginals = torch.zeros(len(orders), 4, 4)
        for order_index, order in enumerate(orders):
            for rank, seat in enumerate(order):
                order_to_marginals[order_index, seat, rank] = 1.0
        self.register_buffer(
            "order_to_marginals", order_to_marginals, persistent=False
        )
        self.register_buffer(
            "rank_utilities",
            torch.tensor((1.0, 1 / 3, -1 / 3, -1.0)),
            persistent=False,
        )

    def forward(self, rank_boundary_features, decision_seats):
        if rank_boundary_features is None or decision_seats is None:
            raise ValueError("rank boundary critic requires state and seat")
        order_logits = self.rank_tower(rank_boundary_features.float()).float()
        marginals = torch.einsum(
            "bo,osr->bsr", order_logits.softmax(-1), self.order_to_marginals,
        )
        probabilities = marginals.gather(
            1, decision_seats.long()[:, None, None].expand(-1, 1, 4),
        ).squeeze(1)
        values = probabilities @ self.rank_utilities
        seat_values = marginals @ self.rank_utilities
        return MatchBoundaryCriticOutput(
            order_logits,
            marginals,
            probabilities.clamp_min(1e-8).log(),
            probabilities,
            values,
            seat_values,
        )


class ActorCritic(nn.Module):
    """Production policy and disjoint match-boundary rank critic."""

    def __init__(
        self, config, *,
        token_cardinalities=(6, 32, 256, 8, 8, 16, 4, 16, 256, 8),
        action_cardinalities=(
            16, 256, 8, 8, 16, 4, 8,
            256, 256, 256, 256, 256, 256, 256, 256,
        ),
    ):
        super().__init__()
        d_model = int(config["d_model"])
        heads = int(config["query_heads"])
        self.context_tokens = int(config.get("context_tokens", 4096))
        self.policy_temperature = float(config.get("policy_temperature", 1.0))
        if self.policy_temperature <= 0:
            raise ValueError("policy temperature must be positive")
        self.token_embedding = FactorEmbedding(
            token_cardinalities, d_model, numeric_dim=8
        )
        self.action_embedding = FactorEmbedding(action_cardinalities, d_model)
        self.canonical_tile_embedding = CanonicalTileEmbedding(d_model)
        self.token_tile_merge_norm = nn.RMSNorm(d_model)
        self.action_tile_merge_norm = nn.RMSNorm(d_model)
        self.backbone = Decoder(
            layers=int(config["layers"]), d_model=d_model,
            query_heads=heads, kv_heads=int(config["kv_heads"]),
            head_dim=int(config["head_dim"]), ffn_dim=int(config["ffn_dim"]),
            context_tokens=self.context_tokens,
        )
        self.action_memory = ActionMemoryCore(
            d_model,
            heads,
            int(config["action_memory_ffn_dim"]),
            int(config["action_memory_layers"]),
            concealed_shape_channels=int(config["concealed_shape_channels"]),
            concealed_shape_blocks=int(config["concealed_shape_blocks"]),
        )
        self.share_all_action_tiles = bool(config["share_all_action_tiles"])
        self.policy_head = nn.Sequential(
            nn.RMSNorm(d_model), nn.Linear(d_model, 1, bias=False)
        )
        self.match_boundary_critic = MatchBoundaryCritic(
            int(config["rank_critic_width"])
        )

    def actor_named_parameters(self):
        return (
            (name, parameter) for name, parameter in self.named_parameters()
            if not name.startswith("match_boundary_critic.")
        )

    def actor_parameters(self):
        return (parameter for _, parameter in self.actor_named_parameters())

    def critic_parameters(self):
        return self.match_boundary_critic.parameters()

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
                            action_offsets[:-1], action_offsets[1:], strict=True
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

    def forward_actor(
        self, token_factors, action_factors, actor_query_indices, action_offsets,
        token_numeric=None, *, lengths=None, action_lengths=None,
        backend="sdpa", rank_boundary_features=None, decision_seats=None,
        strategic_features=None, compute_entropy=True, **_,
    ) -> ActorOutput:
        if token_factors.shape[1] > self.context_tokens:
            raise ValueError(
                f"context overflow: {token_factors.shape[1]} > {self.context_tokens}"
            )
        if lengths is None:
            lengths = torch.full(
                (token_factors.shape[0],), token_factors.shape[1],
                dtype=torch.long, device=token_factors.device,
            )
        token_states = self.token_embedding(token_factors, token_numeric)
        token_states = self.token_tile_merge_norm(
            token_states
            + self.canonical_tile_embedding.tokens(token_factors).to(
                token_states.dtype
            )
        )
        history = self.backbone(token_states, lengths, backend)
        rows = torch.arange(history.shape[0], device=history.device)
        actor_state = history[rows, actor_query_indices]
        public_lengths = torch.minimum(lengths, actor_query_indices + 1)
        history_valid = self._valid(public_lengths, history.shape[1], history.device)
        if strategic_features is None:
            if rank_boundary_features is None or decision_seats is None:
                raise ValueError("actor requires boundary state and decision seat")
            strategic_features = self._strategic_features(
                rank_boundary_features, decision_seats
            )
        state, memory, memory_valid = self.action_memory.memory(
            actor_state,
            strategic_features,
            token_factors,
            public_lengths,
            history,
            history_valid,
            self.canonical_tile_embedding.all_tiles(),
        )
        padded, action_lengths = self._padded_actions(
            action_factors, action_lengths, action_offsets
        )
        action_valid = self._valid(
            action_lengths, padded.shape[1], padded.device
        )
        action_states = self.action_embedding(padded)
        canonical = (
            self.canonical_tile_embedding.action_tiles(padded)
            if self.share_all_action_tiles
            else self.canonical_tile_embedding.actions(padded)
        )
        action_states = self.action_tile_merge_norm(
            action_states + canonical.to(action_states.dtype)
        )
        action_states = self.action_memory(
            action_states, memory, memory_valid, action_valid
        )
        padded_logits = self.policy_head(action_states).squeeze(-1).float() \
            / self.policy_temperature
        logits = padded_logits[action_valid]
        layout = segment_layout(
            action_offsets, total=int(logits.shape[0]), device=logits.device
        )
        logp = segmented_log_softmax(logits, action_offsets, layout=layout)
        entropy = (
            segmented_entropy(logp, action_offsets, layout=layout)
            if compute_entropy else None
        )
        return ActorOutput(logits, logp, entropy, state, action_states)

    def forward_critic(
        self, decision_seats=None, rank_boundary_features=None, **_,
    ):
        return self.match_boundary_critic(
            rank_boundary_features, decision_seats
        )

    def forward(self, **inputs) -> ActorCriticOutput:
        actor = self.forward_actor(**inputs)
        critic = self.forward_critic(**inputs)
        if actor.entropy is None:
            raise ValueError("joint actor-critic forward requires entropy")
        return ActorCriticOutput(
            actor.logits,
            actor.log_probabilities,
            critic.rank_logits,
            critic.rank_probabilities,
            critic.rank_values,
            actor.entropy,
        )

    segmented_entropy = staticmethod(segmented_entropy)

    @torch.no_grad()
    def sample(self, output, action_offsets, *, generator=None, deterministic=False):
        return segmented_sample(
            output.log_probabilities,
            action_offsets,
            generator=generator,
            deterministic=deterministic,
        )
