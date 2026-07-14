"""Decoder-only actor-critic with ragged candidate scoring."""

from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn

from ..encoding.actions import (
    segment_layout,
    segmented_entropy,
    segmented_log_softmax,
    segmented_sample,
)
from .embeddings import FactorEmbedding
from .transformer import Decoder


@dataclass(frozen=True)
class ActorCriticOutput:
    logits: torch.Tensor
    log_probabilities: torch.Tensor
    values: torch.Tensor
    entropy: torch.Tensor
    opponent_count_logits: torch.Tensor
    opponent_tenpai_logits: torch.Tensor


class ActorCritic(nn.Module):
    def __init__(self, config, *, token_cardinalities=(8, 32, 256, 8, 8, 16, 4, 16, 256, 4),
                 action_cardinalities=(16, 256, 8, 8, 16, 4, 8, 256, 256, 256, 256,
                                       256, 256, 256, 256)):
        super().__init__()
        d_model = int(config["d_model"])
        self.context_tokens = int(config.get("context_tokens", 4096))
        self.token_embedding = FactorEmbedding(token_cardinalities, d_model, numeric_dim=8)
        self.action_embedding = FactorEmbedding(action_cardinalities, d_model)
        self.backbone = Decoder(layers=int(config["layers"]), d_model=d_model,
            query_heads=int(config["query_heads"]), kv_heads=int(config["kv_heads"]),
            head_dim=int(config["head_dim"]), ffn_dim=int(config["ffn_dim"]),
            context_tokens=self.context_tokens)
        self.critic_embedding = FactorEmbedding(token_cardinalities, d_model, numeric_dim=8)
        self.critic_decoder = Decoder(layers=int(config.get("critic_layers", 2)), d_model=d_model,
            query_heads=int(config["query_heads"]), kv_heads=int(config["kv_heads"]),
            head_dim=int(config["head_dim"]), ffn_dim=int(config["ffn_dim"]),
            context_tokens=self.context_tokens)
        self.value_query = nn.Parameter(torch.empty(d_model))
        nn.init.normal_(self.value_query, std=d_model ** -0.5)
        self.actor_query = nn.Linear(d_model, 128, bias=False)
        self.action_key = nn.Linear(d_model, 128, bias=False)
        self.opponent_count = nn.Linear(d_model, 3 * 34 * 5)
        self.opponent_tenpai = nn.Linear(d_model, 3)
        self.value = nn.Linear(d_model, 1)

    def forward(self, token_factors, action_factors, actor_query_indices,
                action_offsets, critic_factors, critic_lengths, token_numeric=None,
                critic_numeric=None, *, lengths=None, backend="sdpa"):
        if token_factors.shape[1] > self.context_tokens:
            raise ValueError(f"context overflow: {token_factors.shape[1]} > {self.context_tokens}")
        hidden = self.backbone(self.token_embedding(token_factors, token_numeric), lengths, backend)
        batch = hidden.shape[0]
        rows = torch.arange(batch, device=hidden.device)
        actor_state = hidden[rows, actor_query_indices]
        actor = self.actor_query(actor_state)
        count_logits = self.opponent_count(actor_state).reshape(batch, 3, 34, 5).float()
        tenpai_logits = self.opponent_tenpai(actor_state).float()
        private = self.critic_embedding(critic_factors, critic_numeric)
        critic_token_count = private.shape[1] + 2
        if critic_token_count > self.context_tokens:
            raise ValueError(f"critic context overflow: {critic_token_count} > {self.context_tokens}")
        critic_input = private.new_zeros(batch, critic_token_count, hidden.shape[-1])
        critic_input[:, 0] = actor_state.detach()
        critic_input[:, 1:1 + private.shape[1]] = private
        critic_rows = torch.arange(batch, device=hidden.device)
        value_indices = critic_lengths + 1
        critic_input[critic_rows, value_indices] = self.value_query
        critic_sequence_lengths = critic_lengths + 2
        critic_hidden = self.critic_decoder(critic_input, critic_sequence_lengths, backend)
        values = self.value(critic_hidden[critic_rows, value_indices]).squeeze(-1).float()
        actions = self.action_key(self.action_embedding(action_factors))
        layout = segment_layout(
            action_offsets, total=int(actions.shape[0]), device=actions.device
        )
        _, _, segment_ids = layout
        queries = actor.index_select(0, segment_ids)
        logits = (actions * queries).sum(dim=-1).div(actor.shape[-1] ** 0.5).float()
        logp = segmented_log_softmax(logits, action_offsets, layout=layout)
        entropy = segmented_entropy(logp, action_offsets, layout=layout)
        return ActorCriticOutput(logits, logp, values, entropy, count_logits, tenpai_logits)

    segmented_entropy = staticmethod(segmented_entropy)

    @torch.no_grad()
    def sample(self, output, action_offsets, *, generator=None, deterministic=False):
        return segmented_sample(
            output.log_probabilities,
            action_offsets,
            generator=generator,
            deterministic=deterministic,
        )
