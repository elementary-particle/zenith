"""Verified suit-equivariant, object-referenced production policy."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from ..encoding.actions import (
    segment_layout,
    segmented_entropy,
    segmented_log_softmax,
)
from .actor_critic import (
    ConcealedShapeEncoder,
    PolicyScaffold,
    SwiGLUResidual,
)
from .verified_components import (
    CandidateBlock,
    SCORE_DELTA_ATOMS,
    StrategicMemory,
    VerifiedActorOutput,
    tile_player_relations,
)
from .embeddings import FactorEmbedding
from .tile import (
    TILE_PLANE_CHANNELS,
    current_public_tile_count_planes,
)

def _tile_indices(suit, rank):
    valid = suit.ge(1) & suit.le(4) & rank.ge(1) \
        & torch.where(suit.eq(4), rank.le(7), rank.le(9))
    indices = torch.where(
        suit.eq(4), 27 + rank - 1, (suit - 1) * 9 + rank - 1,
    )
    return indices.clamp(0, 33).long(), valid


class EquivariantTileContent(nn.Module):
    """Suit-shared numbered ranks plus explicit honor identities."""

    def __init__(self, d_model):
        super().__init__()
        self.rank = nn.Embedding(10, d_model, padding_idx=0)
        self.honor = nn.Embedding(8, d_model, padding_idx=0)
        self.red = nn.Embedding(3, d_model, padding_idx=0)
        self.norm = nn.RMSNorm(d_model)

    def tokens(self, factors):
        suit = factors[..., 4].long()
        rank = factors[..., 5].long()
        red = factors[..., 6].long()
        numbered = suit.ge(1) & suit.le(3) & rank.ge(1) & rank.le(9)
        honor = suit.eq(4) & rank.ge(1) & rank.le(7)
        valid = numbered | honor
        value = self.rank(torch.where(numbered, rank, 0))
        value = value + self.honor(torch.where(honor, rank, 0))
        value = value + self.red(torch.where(valid, red.clamp(0, 1) + 1, 0))
        return self.norm(value)

    def all_tiles(self):
        device = self.rank.weight.device
        numbered_rank = torch.arange(1, 10, device=device).repeat(3)
        ranks = torch.cat((numbered_rank, torch.zeros(7, device=device))).long()
        honors = torch.cat((
            torch.zeros(27, device=device),
            torch.arange(1, 8, device=device),
        )).long()
        return self.norm(self.rank(ranks) + self.honor(honors))


class EquivariantTileBlock(nn.Module):
    """Bidirectional tile reasoning with whole-suit-equivariant relations."""

    RELATIONS = 22

    def __init__(self, d_model, heads, ffn_dim):
        super().__init__()
        if d_model % heads:
            raise ValueError("tile workspace heads must divide width")
        self.heads = int(heads)
        self.head_dim = int(d_model) // int(heads)
        self.relation_bias_enabled = True
        self.norm = nn.RMSNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.output = nn.Linear(d_model, d_model, bias=False)
        self.relation_bias = nn.Embedding(self.RELATIONS, heads)
        self.ffn = SwiGLUResidual(d_model, ffn_dim)
        self.register_buffer(
            "relations", tile_player_relations()[:34, :34], persistent=False,
        )

    def forward(self, values):
        batch, tokens, width = values.shape
        if tokens != 34:
            raise ValueError("tile workspace requires exactly 34 tile nodes")
        qkv = self.qkv(self.norm(values)).view(
            batch, tokens, 3, self.heads, self.head_dim,
        )
        query, key, content = qkv.unbind(2)
        scores = torch.einsum("bthd,bshd->bhts", query, key).float()
        scores = scores / math.sqrt(self.head_dim)
        if self.relation_bias_enabled:
            bias = self.relation_bias(self.relations).permute(2, 0, 1)
            scores = scores + bias[None]
        weights = scores.softmax(-1).to(content.dtype)
        attended = torch.einsum("bhts,bshd->bthd", weights, content)
        attended = attended.reshape(batch, tokens, width)
        return self.ffn(values + self.output(attended))


class TileHistoryFusion(nn.Module):
    """Let tile nodes retrieve cached causal tokens through object relations."""

    RELATIONS = 23  # no reference plus the 22 tile-to-tile relations

    def __init__(self, d_model, heads, ffn_dim):
        super().__init__()
        if d_model % heads:
            raise ValueError("tile-history heads must divide width")
        self.heads = int(heads)
        self.head_dim = int(d_model) // int(heads)
        self.relation_bias_enabled = True
        self.tile_norm = nn.RMSNorm(d_model)
        self.history_norm = nn.RMSNorm(d_model)
        self.query = nn.Linear(d_model, d_model, bias=False)
        self.key_value = nn.Linear(d_model, 2 * d_model, bias=False)
        self.output = nn.Linear(d_model, d_model, bias=False)
        self.relation_bias = nn.Embedding(self.RELATIONS, heads)
        self.ffn = SwiGLUResidual(d_model, ffn_dim)
        self.register_buffer(
            "relations", tile_player_relations()[:34, :34], persistent=False,
        )

    def forward(self, tiles, history, history_valid, token_tile, token_has_tile):
        batch, tile_count, width = tiles.shape
        tokens = history.shape[1]
        query = self.query(self.tile_norm(tiles)).view(
            batch, tile_count, self.heads, self.head_dim,
        )
        key, value = self.key_value(self.history_norm(history)).view(
            batch, tokens, 2, self.heads, self.head_dim,
        ).unbind(2)
        scores = torch.einsum("bqhd,bkhd->bhqk", query, key).float()
        scores = scores / math.sqrt(self.head_dim)
        referenced = self.relations[:, token_tile.reshape(-1)].view(
            tile_count, batch, tokens,
        ).permute(1, 0, 2) + 1
        relation = torch.where(
            token_has_tile[:, None], referenced,
            torch.zeros_like(referenced),
        )
        if self.relation_bias_enabled:
            bias = self.relation_bias(relation).permute(0, 3, 1, 2)
            scores = scores + bias.float()
        scores = scores.masked_fill(
            ~history_valid[:, None, None], -torch.inf,
        )
        weights = scores.softmax(-1).to(value.dtype)
        attended = torch.einsum("bhqk,bkhd->bqhd", weights, value)
        attended = attended.reshape(batch, tile_count, width)
        return self.ffn(tiles + self.output(attended))


class ReferencedTileWorkspace(nn.Module):
    """Exact current tile state fused with causal public history."""

    def __init__(
        self, d_model, heads, ffn_dim, layers, *,
        concealed_shape_channels, concealed_shape_blocks,
    ):
        super().__init__()
        self.counts = nn.Linear(TILE_PLANE_CHANNELS, d_model, bias=False)
        self.shape = ConcealedShapeEncoder(
            d_model, concealed_shape_channels, concealed_shape_blocks,
        )
        self.initial_norm = nn.RMSNorm(d_model)
        self.history = nn.ModuleList(
            TileHistoryFusion(d_model, heads, ffn_dim)
            for _ in range(int(layers))
        )
        self.tiles = nn.ModuleList(
            EquivariantTileBlock(d_model, heads, ffn_dim)
            for _ in range(int(layers))
        )

    def forward(
        self, planes, tile_content, history, history_valid,
        token_tile, token_has_tile,
    ):
        counts = planes.flatten(2)[:, :, :34].transpose(1, 2) * 0.25
        values = self.initial_norm(
            self.counts(counts).to(tile_content.dtype)
            + tile_content[None]
            + self.shape(planes).to(tile_content.dtype)
        )
        for history_block, tile_block in zip(
            self.history, self.tiles, strict=True,
        ):
            values = history_block(
                values, history, history_valid, token_tile, token_has_tile,
            )
            values = tile_block(values)
        return values


class VerifiedPolicyCore(PolicyScaffold):
    """Causal tactical trunk plus equivariant tile/action object references."""
    MATCH_GLOBAL_PREFIX_TOKENS = 11
    strategic_enabled = True
    references_enabled = True

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        d_model = int(config["d_model"])
        heads = int(config["query_heads"])
        self.token_embedding = FactorEmbedding(
            (6, 32, 256, 8, 1, 16, 4, 16, 256, 8),
            d_model, numeric_dim=8,
        )
        self.action_embedding = FactorEmbedding(
            (16, 1, 8, 1, 16, 4, 8, 1, 1, 1, 1, 256, 256, 256, 256),
            d_model,
        )
        self.tile_content = EquivariantTileContent(d_model)
        self.workspace = ReferencedTileWorkspace(
            d_model, heads, int(config["ffn_dim"]),
            int(config.get("ground_board_layers", 2)),
            concealed_shape_channels=int(config["concealed_shape_channels"]),
            concealed_shape_blocks=int(config["concealed_shape_blocks"]),
        )
        self.reference_role = nn.Embedding(5, d_model)
        self.reference_red = nn.Embedding(2, d_model)
        self.reference_output = nn.Sequential(
            nn.RMSNorm(d_model), nn.Linear(d_model, d_model, bias=False),
            nn.SiLU(),
        )
        self.action_reference_norm = nn.RMSNorm(d_model)
        self.strategic_memory = StrategicMemory(d_model)
        self.candidate_blocks = nn.ModuleList(
            CandidateBlock(
                d_model, heads, int(config["action_memory_ffn_dim"]),
            )
            for _ in range(int(config["action_memory_layers"]))
        )
        self.policy = nn.Sequential(
            nn.RMSNorm(d_model), nn.Linear(d_model, 1, bias=False),
        )
        self.outcome = nn.Linear(d_model, 4)
        self.delta = nn.Linear(d_model, len(SCORE_DELTA_ATOMS))
        self.placement = nn.Sequential(
            nn.RMSNorm(2 * d_model),
            nn.Linear(2 * d_model, d_model, bias=False),
            nn.SiLU(), nn.Linear(d_model, 4),
        )
        self.register_buffer(
            "score_delta_atoms",
            torch.tensor(SCORE_DELTA_ATOMS, dtype=torch.float32),
            persistent=False,
        )
        # A public, action-independent decision baseline.  The residual is
        # zero-initialized, so an existing verified BC checkpoint begins at
        # the proven match-boundary value without changing its policy.
        self.decision_value = nn.Sequential(
            nn.RMSNorm(2 * d_model),
            nn.Linear(2 * d_model, d_model, bias=False),
            nn.SiLU(),
            nn.Linear(d_model, 1),
        )
        nn.init.zeros_(self.decision_value[-1].weight)
        nn.init.zeros_(self.decision_value[-1].bias)

    def actor_named_parameters(self):
        return (
            (name, parameter) for name, parameter in self.named_parameters()
            if not name.startswith((
                "match_boundary_critic.", "decision_value.",
            ))
        )

    def critic_parameters(self):
        yield from self.match_boundary_critic.parameters()
        yield from self.decision_value.parameters()

    def boundary_critic_parameters(self):
        return self.match_boundary_critic.parameters()

    def decision_critic_parameters(self):
        return self.decision_value.parameters()

    @staticmethod
    def _decision_features(output, action_lengths):
        actions = output.action_states.float()
        lengths = torch.as_tensor(
            action_lengths, device=actions.device,
        ).long()
        valid = torch.arange(
            actions.shape[1], device=actions.device,
        )[None] < lengths[:, None]
        probabilities = torch.zeros(
            valid.shape, dtype=torch.float32, device=actions.device,
        )
        probabilities[valid] = output.log_probabilities.float().exp()
        mean_action = (actions * probabilities[..., None]).sum(1)
        return torch.cat((output.actor_states.float(), mean_action), -1)

    def forward_decision_critic(self, **inputs):
        """Evaluate the detached public-state baseline for PPO training."""
        with torch.no_grad():
            output = self.forward_actor(
                **inputs,
                compute_entropy=False,
                compute_value=False,
                compute_auxiliary=False,
            )
            features = self.decision_features(
                output, inputs["action_lengths"],
            ).detach()
            boundary = self.forward_critic(**inputs).rank_values.float()
        return self.forward_decision_head(features, boundary)

    def decision_features(self, output, action_lengths):
        """Extract the detached public-state features used by the value head."""
        return self._decision_features(output, action_lengths)

    def forward_decision_head(self, features, boundary_values):
        """Fit the cheap decision residual without replaying the actor trunk."""
        return boundary_values.detach().float() + self.decision_value(
            features.detach(),
        ).squeeze(-1).float()

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Load verified BC weights while initializing the new value residual.

        The only accepted migration is the four zero-initialized
        ``decision_value`` tensors.  Any actor or boundary-critic mismatch
        remains a hard error.
        """
        result = super().load_state_dict(
            state_dict, strict=False, assign=assign,
        )
        allowed = {
            "decision_value.0.weight",
            "decision_value.1.weight",
            "decision_value.3.weight",
            "decision_value.3.bias",
        }
        missing = set(result.missing_keys)
        unexpected = set(result.unexpected_keys)
        if unexpected or missing - allowed or (strict and missing and missing != allowed):
            raise RuntimeError(
                "verified checkpoint tensor contract changed: "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )
        return result

    @staticmethod
    def _suit_blind_tokens(factors):
        result = factors.clone()
        result[..., 4] = 0
        return result

    @staticmethod
    def _suit_blind_actions(factors):
        result = factors.clone()
        result[..., 1] = 0
        result[..., 3] = 0
        result[..., 7:11] = 0
        return result

    def _referenced_actions(self, padded, workspace):
        batch, actions, _ = padded.shape
        primary = padded[..., 1].long()
        primary_valid = primary.ge(0) & primary.lt(34)
        semantic = padded[..., 7:11].long()
        semantic_type = semantic.div(4, rounding_mode="floor")
        semantic_valid = semantic.gt(0) & semantic_type.lt(34)
        indices = torch.cat((
            primary.clamp(0, 33)[..., None],
            semantic_type.clamp(0, 33),
        ), -1)
        valid = torch.cat((primary_valid[..., None], semantic_valid), -1)
        batch_indices = torch.arange(
            batch, device=padded.device,
        )[:, None, None].expand(-1, actions, 5)
        referenced = workspace[batch_indices, indices]
        roles = self.reference_role(
            torch.arange(5, device=padded.device),
        )[None, None]
        primary_red = padded[..., 5].long().clamp(0, 1)[..., None]
        semantic_red = semantic.remainder(4).eq(0).long()
        red = torch.cat((primary_red, semantic_red), -1)
        referenced = referenced + roles + self.reference_red(red)
        referenced = referenced * valid[..., None]
        count = valid.sum(-1, keepdim=True).clamp_min(1)
        return self.reference_output(referenced.sum(-2) / count.sqrt())

    def _strategic_tokens(
        self, strategic_features, full_factors, full_numeric,
    ):
        del full_factors, full_numeric
        strategic = self.strategic_memory(strategic_features)
        return strategic if self.strategic_enabled \
            else torch.zeros_like(strategic)

    def forward_actor(
        self, token_factors, action_factors, actor_query_indices, action_offsets,
        token_numeric=None, *, lengths=None, action_lengths=None,
        backend="sdpa", rank_boundary_features=None, decision_seats=None,
        strategic_features=None, compute_entropy=True, compute_value=True,
        compute_auxiliary=True, **_,
    ):
        if token_factors.shape[1] > self.context_tokens:
            raise ValueError(
                f"context overflow: {token_factors.shape[1]} > "
                f"{self.context_tokens}"
            )
        if strategic_features is None:
            if rank_boundary_features is None or decision_seats is None:
                raise ValueError("referenced actor requires match state and seat")
            strategic_features = self._strategic_features(
                rank_boundary_features, decision_seats,
            )
        prefix = self.MATCH_GLOBAL_PREFIX_TOKENS
        if token_factors.shape[1] <= prefix:
            raise ValueError("encoded context has no tactical tokens")
        full_factors = token_factors
        full_numeric = token_numeric
        full_lengths = lengths
        token_factors = token_factors[:, prefix:]
        token_numeric = None if token_numeric is None \
            else token_numeric[:, prefix:]
        lengths = None if lengths is None else lengths - prefix
        actor_query_indices = actor_query_indices - prefix
        if lengths is None:
            lengths = torch.full(
                (token_factors.shape[0],), token_factors.shape[1],
                dtype=torch.long, device=token_factors.device,
            )
        if full_lengths is None:
            full_lengths = lengths + prefix
        suit_blind = self._suit_blind_tokens(token_factors)
        history = self.token_embedding(suit_blind, token_numeric)
        history = self.token_tile_merge_norm(
            history + self.tile_content.tokens(token_factors).to(history.dtype)
        )
        history = self.backbone(history, lengths, backend)
        rows = torch.arange(history.shape[0], device=history.device)
        actor_state = history[rows, actor_query_indices]
        history_valid = self._valid(lengths, history.shape[1], history.device)
        token_tile, token_has_tile = _tile_indices(
            token_factors[..., 4], token_factors[..., 5],
        )
        planes = current_public_tile_count_planes(full_factors, full_lengths)
        workspace = self.workspace(
            planes, self.tile_content.all_tiles().to(history.dtype),
            history, history_valid, token_tile, token_has_tile,
        )
        strategic = self._strategic_tokens(
            strategic_features, full_factors, full_numeric,
        )
        memory = torch.cat((actor_state[:, None], workspace, history, strategic), 1)
        memory_valid = torch.cat((
            torch.ones(
                history.shape[0], 1 + 34, dtype=torch.bool,
                device=history.device,
            ),
            history_valid,
            torch.ones(
                strategic.shape[:2], dtype=torch.bool, device=history.device,
            ),
        ), 1)
        padded, action_lengths = self._padded_actions(
            action_factors, action_lengths, action_offsets,
        )
        action_valid = self._valid(
            action_lengths, padded.shape[1], padded.device,
        )
        actions = self.action_embedding(self._suit_blind_actions(padded))
        references = self._referenced_actions(padded, workspace).to(actions.dtype)
        if not self.references_enabled:
            references = torch.zeros_like(references)
        actions = self.action_reference_norm(actions + references)
        for block in self.candidate_blocks:
            actions = block(actions, action_valid, memory, memory_valid)
        padded_logits = self.policy(actions).squeeze(-1).float() \
            / self.policy_temperature
        logits = padded_logits[action_valid]
        layout = segment_layout(
            action_offsets, total=int(logits.shape[0]), device=logits.device,
        )
        logp = segmented_log_softmax(logits, action_offsets, layout=layout)
        entropy = segmented_entropy(logp, action_offsets, layout=layout) \
            if compute_entropy else None
        outcome_logits = score_delta_logits = placement_logits = None
        if compute_auxiliary:
            strategic_summary = strategic.mean(1)[:, None].expand(
                -1, actions.shape[1], -1,
            )
            outcome_logits = self.outcome(actions).float()[action_valid]
            score_delta_logits = self.delta(actions).float()[action_valid]
            placement_logits = self.placement(
                torch.cat((actions, strategic_summary), -1)
            ).float()[action_valid]
        state_values = None
        if compute_value:
            provisional = VerifiedActorOutput(
                logits, logp, entropy, actor_state, actions,
                outcome_logits, score_delta_logits, placement_logits,
                None,
            )
            state_features = self._decision_features(
                provisional, action_lengths,
            ).detach()
            boundary_values = self.forward_critic(
                rank_boundary_features=rank_boundary_features,
                decision_seats=decision_seats,
            ).rank_values.detach()
            state_values = boundary_values + self.decision_value(
                state_features,
            ).squeeze(-1).float()
        return VerifiedActorOutput(
            logits, logp, entropy, actor_state, actions,
            outcome_logits, score_delta_logits, placement_logits,
            state_values,
        )

    def auxiliary_loss(self, output, selected, examples):
        if any(value is None for value in (
            output.hand_outcome_logits,
            output.score_delta_logits,
            output.placement_logits,
        )):
            raise ValueError("auxiliary actor outputs were not requested")
        outcome_targets = torch.tensor(
            [row.critic.hand_outcome_target for row in examples],
            dtype=torch.long, device=selected.device,
        )
        score_deltas = torch.tensor(
            [row.critic.hand_score_delta for row in examples],
            dtype=torch.float32, device=selected.device,
        )
        placement_targets = torch.tensor(
            [row.critic.terminal_placement for row in examples],
            dtype=torch.long, device=selected.device,
        )
        delta_targets = (
            score_deltas[:, None] - self.score_delta_atoms[None]
        ).abs().argmin(-1)
        return {
            "outcome": F.cross_entropy(
                output.hand_outcome_logits.index_select(0, selected),
                outcome_targets,
            ),
            "score_delta": F.cross_entropy(
                output.score_delta_logits.index_select(0, selected),
                delta_targets,
            ),
            "placement": F.cross_entropy(
                output.placement_logits.index_select(0, selected),
                placement_targets,
            ),
        }
