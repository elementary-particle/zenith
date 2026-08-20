"""Dealer-relative structured boundary critic for the verified actor."""

from __future__ import annotations

from itertools import permutations

import torch
from torch import nn

from .actor_critic import MatchBoundaryCriticOutput
from .verified import VerifiedPolicyCore


VERIFIED_BC_ARCHITECTURE = (
    "ground-up-referenced-workspace-structured-boundary-rank-v1"
)
PRODUCTION_ARCHITECTURE = "verified-public-state-value-ppo-v1"
ORDERS = tuple(permutations(range(4)))


class PlayerMixBlock(nn.Module):
    """Four-player interaction through a cache-friendly pooled residual."""

    def __init__(self, width):
        super().__init__()
        width = int(width)
        self.update = nn.Sequential(
            nn.RMSNorm(2 * width),
            nn.Linear(2 * width, 2 * width, bias=False),
            nn.SiLU(),
            nn.Linear(2 * width, width, bias=False),
        )
        self.output_norm = nn.RMSNorm(width)

    def forward(self, players):
        pooled = players.mean(1, keepdim=True).expand_as(players)
        return self.output_norm(
            players + self.update(torch.cat((players, pooled), -1))
        )


class StructuredOrderCritic(nn.Module):
    """Dealer-relative player objects with a shared joint-order scorer."""

    def __init__(self, width=64, layers=2):
        super().__init__()
        width = int(width)
        self.score = nn.Sequential(
            nn.Linear(1, width, bias=False),
            nn.SiLU(),
        )
        # Features 8:28 are round, counters, and remaining match structure.
        # Absolute dealer identity at 4:8 is deliberately absent.
        self.match = nn.Sequential(
            nn.Linear(20, width, bias=False),
            nn.SiLU(),
            nn.RMSNorm(width),
        )
        self.role = nn.Embedding(4, width)
        self.input_norm = nn.RMSNorm(width)
        self.blocks = nn.ModuleList(
            PlayerMixBlock(width) for _ in range(int(layers))
        )
        self.rank = nn.Embedding(4, width)
        self.order_scorer = nn.Sequential(
            nn.RMSNorm(4 * width),
            nn.Linear(4 * width, width, bias=False),
            nn.SiLU(),
            nn.RMSNorm(width),
            nn.Linear(width, 1),
        )
        self.register_buffer(
            "orders", torch.tensor(ORDERS, dtype=torch.long), persistent=False,
        )

    def forward(self, features):
        values = features.float()
        batch = int(values.shape[0])
        roles = self.role(torch.arange(4, device=values.device))[None]
        global_state = self.match(values[:, 8:])[:, None]
        players = self.input_norm(
            self.score(values[:, :4, None]) + roles + global_state
        )
        for block in self.blocks:
            players = block(players)
        batch_indices = torch.arange(batch, device=values.device)[:, None, None]
        ordered = players[
            batch_indices,
            self.orders[None].expand(batch, -1, -1),
        ]
        ordered = ordered + self.rank(
            torch.arange(4, device=values.device)
        )[None, None]
        return self.order_scorer(ordered.flatten(-2)).squeeze(-1).float()


class StructuredMatchBoundaryCritic(nn.Module):
    """Normal boundary-critic interface backed by structured order scoring."""

    def __init__(self, width=64, layers=2):
        super().__init__()
        self.order_model = StructuredOrderCritic(width, layers)
        order_to_marginals = torch.zeros(24, 4, 4)
        for order_index, order in enumerate(ORDERS):
            for rank, seat in enumerate(order):
                order_to_marginals[order_index, seat, rank] = 1.0
        self.register_buffer(
            "order_to_marginals", order_to_marginals, persistent=False,
        )
        self.register_buffer(
            "rank_utilities",
            torch.tensor((1.0, 1 / 3, -1 / 3, -1.0)),
            persistent=False,
        )

    def forward(self, rank_boundary_features, decision_seats):
        if rank_boundary_features is None or decision_seats is None:
            raise ValueError("rank boundary critic requires state and seat")
        order_logits = self.order_model(rank_boundary_features.float()).float()
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


class VerifiedActorCritic(
    VerifiedPolicyCore,
):
    """Bit-identical verified actor plus structured boundary critic."""

    architecture_id = PRODUCTION_ARCHITECTURE

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.match_boundary_critic = StructuredMatchBoundaryCritic(
            int(config["boundary_critic_width"]),
            int(config.get("structured_boundary_layers", 2)),
        )
