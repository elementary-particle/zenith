"""Contextual public actor and disjoint dealer-canonical oracle critic."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from ..encoding.actions import (
    segment_layout,
    segmented_entropy,
    segmented_log_softmax,
    segmented_sample,
)
from .embeddings import FactorEmbedding
from .transformer import Decoder, Encoder


@dataclass(frozen=True)
class ActorOutput:
    logits: torch.Tensor
    log_probabilities: torch.Tensor
    entropy: torch.Tensor
    opponent_count_logits: torch.Tensor
    opponent_tenpai_logits: torch.Tensor


@dataclass(frozen=True)
class CriticOutput:
    score_values: torch.Tensor
    rank_logits: torch.Tensor
    rank_probabilities: torch.Tensor
    rank_values: torch.Tensor


@dataclass(frozen=True)
class ActorCriticOutput:
    logits: torch.Tensor
    log_probabilities: torch.Tensor
    score_values: torch.Tensor
    rank_logits: torch.Tensor
    rank_probabilities: torch.Tensor
    rank_values: torch.Tensor
    entropy: torch.Tensor
    opponent_count_logits: torch.Tensor
    opponent_tenpai_logits: torch.Tensor


class SwiGLUResidual(nn.Module):
    def __init__(self, d_model: int, ffn_dim: int):
        super().__init__()
        self.norm = nn.RMSNorm(d_model)
        self.up = nn.Linear(d_model, 2 * ffn_dim, bias=False)
        self.down = nn.Linear(ffn_dim, d_model, bias=False)

    def forward(self, value):
        gate, content = self.up(self.norm(value)).chunk(2, dim=-1)
        return value + self.down(F.silu(gate) * content)


class ActorCritic(nn.Module):
    """One checkpoint-breaking architecture with fully disjoint optimizers."""

    def __init__(self, config, *, token_cardinalities=(8, 32, 256, 8, 8, 16, 4,
                 16, 256, 8), action_cardinalities=(16, 256, 8, 8, 16, 4, 8,
                 256, 256, 256, 256, 256, 256, 256, 256)):
        super().__init__()
        d_model = int(config["d_model"])
        heads = int(config["query_heads"])
        self.context_tokens = int(config.get("context_tokens", 4096))
        self.token_embedding = FactorEmbedding(
            token_cardinalities, d_model, numeric_dim=8
        )
        self.action_embedding = FactorEmbedding(action_cardinalities, d_model)
        self.backbone = Decoder(
            layers=int(config["layers"]), d_model=d_model,
            query_heads=heads, kv_heads=int(config["kv_heads"]),
            head_dim=int(config["head_dim"]), ffn_dim=int(config["ffn_dim"]),
            context_tokens=self.context_tokens,
        )

        # Candidate contextualization is deliberately position-free, making
        # legal action ordering irrelevant while still allowing competition.
        self.action_public_norm = nn.RMSNorm(d_model)
        self.action_public_attention = nn.MultiheadAttention(
            d_model, heads, batch_first=True, bias=False
        )
        self.action_candidate_norm = nn.RMSNorm(d_model)
        self.action_candidate_attention = nn.MultiheadAttention(
            d_model, heads, batch_first=True, bias=False
        )
        self.action_ffn = SwiGLUResidual(d_model, int(config["ffn_dim"]))
        self.actor_query = nn.Linear(d_model, 128, bias=False)
        self.action_key = nn.Linear(d_model, 128, bias=False)
        self.action_score = nn.Sequential(
            nn.Linear(3 * d_model, d_model), nn.SiLU(), nn.Linear(d_model, 1)
        )

        self.opponent_queries = nn.Parameter(torch.empty(3, d_model))
        nn.init.normal_(self.opponent_queries, std=d_model ** -0.5)
        self.opponent_attention = nn.MultiheadAttention(
            d_model, heads, batch_first=True, bias=False
        )
        self.belief_tile_embedding = nn.Embedding(34, d_model)
        self.opponent_count = nn.Sequential(
            nn.Linear(3 * d_model, d_model), nn.SiLU(), nn.Linear(d_model, 5)
        )
        self.opponent_tenpai = nn.Sequential(
            nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, 1)
        )

        # The oracle owns every one of its embeddings and projections.  No
        # actor state is accepted by this path.
        self.oracle_embedding = FactorEmbedding(
            token_cardinalities, d_model, numeric_dim=8
        )
        self.oracle_backbone = Encoder(
            layers=int(config.get("critic_layers", 4)), d_model=d_model,
            query_heads=heads, kv_heads=int(config["kv_heads"]),
            head_dim=int(config["head_dim"]), ffn_dim=int(config["ffn_dim"]),
            context_tokens=self.context_tokens,
        )
        self.oracle_task_queries = nn.Parameter(torch.empty(2, 4, d_model))
        nn.init.normal_(self.oracle_task_queries, std=d_model ** -0.5)
        self.oracle_pool_attention = nn.MultiheadAttention(
            d_model, heads, batch_first=True, bias=False
        )
        self.oracle_action_embedding = FactorEmbedding(action_cardinalities, d_model)
        self.oracle_action_norm = nn.RMSNorm(d_model)
        self.score_value = nn.Sequential(
            nn.Linear(3 * d_model, d_model), nn.SiLU(), nn.Linear(d_model, 1)
        )
        self.rank_value = nn.Sequential(
            nn.Linear(3 * d_model, d_model), nn.SiLU(), nn.Linear(d_model, 4)
        )
        self.register_buffer(
            "rank_utilities",
            torch.tensor((1.0, 1.0 / 3.0, -1.0 / 3.0, -1.0)),
            persistent=False,
        )

    def actor_parameters(self):
        return (
            parameter for name, parameter in self.named_parameters()
            if not name.startswith(("oracle_", "score_value", "rank_value"))
        )

    def critic_parameters(self):
        return (
            parameter for name, parameter in self.named_parameters()
            if name.startswith(("oracle_", "score_value", "rank_value"))
        )

    @staticmethod
    def _valid(lengths, maximum, device):
        return torch.arange(maximum, device=device)[None] < lengths[:, None]

    @staticmethod
    def _padded_actions(action_factors, action_lengths, action_offsets):
        if action_factors.ndim == 3:
            if action_lengths is None:
                _, action_lengths, _ = segment_layout(
                    action_offsets,
                    total=sum(int(end) - int(start) for start, end in zip(
                        action_offsets[:-1], action_offsets[1:]
                    )),
                    device=action_factors.device,
                )
            return action_factors, action_lengths
        if action_factors.ndim != 2:
            raise ValueError("action factors must be padded [batch, actions, 15]")
        layout = segment_layout(
            action_offsets, total=int(action_factors.shape[0]),
            device=action_factors.device,
        )
        offsets, lengths, _ = layout
        maximum = int(lengths.max())
        padded = action_factors.new_zeros(lengths.numel(), maximum, 15)
        for row, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
            padded[row, :int(end - start)] = action_factors[start:end]
        return padded, lengths

    def forward_actor(
        self, token_factors, action_factors, actor_query_indices, action_offsets,
        token_numeric=None, *, lengths=None, action_lengths=None, backend="sdpa",
        **_,
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
        hidden = self.backbone(
            self.token_embedding(token_factors, token_numeric), lengths, backend
        )
        batch = hidden.shape[0]
        rows = torch.arange(batch, device=hidden.device)
        actor_state = hidden[rows, actor_query_indices]
        padded, action_lengths = self._padded_actions(
            action_factors, action_lengths, action_offsets
        )
        action_valid = self._valid(
            action_lengths, padded.shape[1], padded.device
        )
        # Cross-attention is restricted to the causal prefix ending at the
        # actor query even if a diagnostic caller appends additional tokens.
        public_valid = self._valid(
            torch.minimum(lengths, actor_query_indices + 1),
            hidden.shape[1], hidden.device,
        )
        actions = self.action_embedding(padded)
        cross, _ = self.action_public_attention(
            self.action_public_norm(actions), hidden, hidden,
            key_padding_mask=~public_valid, need_weights=False,
        )
        actions = actions + cross
        competition, _ = self.action_candidate_attention(
            self.action_candidate_norm(actions), actions, actions,
            key_padding_mask=~action_valid, need_weights=False,
        )
        actions = self.action_ffn(actions + competition)
        actions = torch.where(action_valid[..., None], actions, torch.zeros_like(actions))

        actor_bilinear = self.actor_query(actor_state)
        action_bilinear = self.action_key(actions)
        bilinear = (action_bilinear * actor_bilinear[:, None]).sum(-1) / math.sqrt(
            actor_bilinear.shape[-1]
        )
        query = actor_state[:, None].expand_as(actions)
        learned = self.action_score(torch.cat((query, actions, query * actions), -1)).squeeze(-1)
        padded_logits = (bilinear + learned).float()
        logits = padded_logits[action_valid]
        layout = segment_layout(
            action_offsets, total=int(logits.shape[0]), device=logits.device
        )
        logp = segmented_log_softmax(logits, action_offsets, layout=layout)
        entropy = segmented_entropy(logp, action_offsets, layout=layout)

        opponent_query = self.opponent_queries[None].expand(batch, -1, -1)
        opponent, _ = self.opponent_attention(
            opponent_query, hidden, hidden,
            key_padding_mask=~public_valid, need_weights=False,
        )
        tile = self.belief_tile_embedding.weight[None, None].expand(batch, 3, -1, -1)
        opponent_tile = opponent[:, :, None].expand(-1, -1, 34, -1)
        count_logits = self.opponent_count(torch.cat(
            (opponent_tile, tile, opponent_tile * tile), dim=-1
        )).float()
        tenpai_logits = self.opponent_tenpai(opponent).squeeze(-1).float()
        return ActorOutput(logits, logp, entropy, count_logits, tenpai_logits)

    def forward_critic(
        self, oracle_factors, oracle_lengths, decision_oracle_indices,
        decision_seats, action_factors, action_lengths, action_offsets,
        oracle_numeric=None, *, backend="sdpa", **_,
    ) -> CriticOutput:
        if oracle_factors.shape[1] == 0 or (oracle_lengths <= 0).any():
            raise ValueError("oracle critic requires a non-empty privileged snapshot")
        memory = self.oracle_backbone(
            self.oracle_embedding(oracle_factors, oracle_numeric),
            oracle_lengths, backend,
        )
        oracle_valid = self._valid(
            oracle_lengths, memory.shape[1], memory.device
        )
        unique = memory.shape[0]
        queries = self.oracle_task_queries.reshape(8, -1)[None].expand(unique, -1, -1)
        pooled, _ = self.oracle_pool_attention(
            queries, memory, memory,
            key_padding_mask=~oracle_valid, need_weights=False,
        )
        pooled = pooled.reshape(unique, 2, 4, -1)
        selected = pooled.index_select(0, decision_oracle_indices)
        rows = torch.arange(selected.shape[0], device=selected.device)
        score_state = selected[rows, 0, decision_seats]
        rank_state = selected[rows, 1, decision_seats]

        padded, action_lengths = self._padded_actions(
            action_factors, action_lengths, action_offsets
        )
        valid = self._valid(action_lengths, padded.shape[1], padded.device)
        legal = self.oracle_action_embedding(padded)
        legal = (legal * valid[..., None]).sum(1) / action_lengths.clamp_min(1)[:, None]
        legal = self.oracle_action_norm(legal)
        score_values = self.score_value(torch.cat(
            (score_state, legal, score_state * legal), -1
        )).squeeze(-1).float()
        rank_logits = self.rank_value(torch.cat(
            (rank_state, legal, rank_state * legal), -1
        )).float()
        rank_probabilities = rank_logits.softmax(-1)
        rank_values = (rank_probabilities * self.rank_utilities).sum(-1)
        return CriticOutput(
            score_values, rank_logits, rank_probabilities, rank_values.float()
        )

    def forward(self, **inputs) -> ActorCriticOutput:
        actor = self.forward_actor(**inputs)
        critic = self.forward_critic(**inputs)
        return ActorCriticOutput(
            actor.logits, actor.log_probabilities, critic.score_values,
            critic.rank_logits, critic.rank_probabilities, critic.rank_values,
            actor.entropy, actor.opponent_count_logits,
            actor.opponent_tenpai_logits,
        )

    segmented_entropy = staticmethod(segmented_entropy)

    @torch.no_grad()
    def sample(self, output, action_offsets, *, generator=None, deterministic=False):
        return segmented_sample(
            output.log_probabilities, action_offsets, generator=generator,
            deterministic=deterministic,
        )
