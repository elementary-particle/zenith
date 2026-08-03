"""Independent privileged action-Q critic for the current-kyoku experiment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from zenith_ppo.encoding.critic import encode_rank_boundary
from zenith_ppo.model.embeddings import FactorEmbedding


PRIVILEGED_TILE_CHANNELS = 15
PRIVILEGED_STATE_FEATURES = 81


@dataclass(frozen=True, slots=True)
class PrivilegedSnapshot:
    tile_features: np.ndarray
    wall_types: np.ndarray
    wall_status: np.ndarray
    state_features: np.ndarray


def _relative_seats(dealer: int) -> tuple[int, ...]:
    return tuple((int(dealer) + offset) % 4 for offset in range(4))


def _one_hot(value: int, size: int) -> tuple[float, ...]:
    return tuple(float(index == int(value)) for index in range(int(size)))


def _flag_bits(value: int, width: int = 8) -> tuple[float, ...]:
    return tuple(float(bool(int(value) & (1 << bit))) for bit in range(width))


def _u8_array(value) -> np.ndarray:
    if isinstance(value, (bytes, bytearray, memoryview)):
        return np.frombuffer(value, dtype=np.uint8)
    return np.asarray(value, dtype=np.uint8)


def _frame_mapping(state) -> dict:
    return {
        "scores": tuple(state.scores),
        "round_wind": int(state.round_wind),
        "hand_number": int(state.hand_number),
        "dealer": int(state.dealer),
        "honba": int(state.honba),
        "riichi_deposits": int(state.riichi_deposits),
    }


def encode_privileged_snapshot(state) -> PrivilegedSnapshot:
    """Encode hidden and public state in dealer-relative seat coordinates."""
    hidden = state.hidden
    if hidden is None:
        raise ValueError("privileged Q critic requires hidden native state")
    dealer = int(state.dealer)
    seats = _relative_seats(dealer)
    tiles = np.zeros((34, PRIVILEGED_TILE_CHANNELS), dtype=np.float16)
    concealed = tuple(tuple(row) for row in hidden.concealed_counts)
    for relative, absolute in enumerate(seats):
        tiles[:, relative] = np.asarray(
            concealed[absolute], dtype=np.float16
        ) / 4.0
    for river in state.rivers:
        relative = (int(river.seat) - dealer) % 4
        tiles[int(river.tile) // 4, 4 + relative] += 0.25
    for meld in state.melds:
        relative = (int(meld.seat) - dealer) % 4
        for physical in meld.tiles:
            tiles[int(physical) // 4, 8 + relative] += 0.25
    tiles[:, 12] = _u8_array(hidden.live_wall_counts).astype(np.float16) / 4.0
    for physical in state.dora_indicators:
        tiles[int(physical) // 4, 13] += 0.25
    for physical in hidden.ura_indicators:
        tiles[int(physical) // 4, 14] += 0.25

    wall = _u8_array(hidden.wall).astype(np.int64)
    wall_types = np.ascontiguousarray(wall // 4, dtype=np.uint8)
    live_start, live_end, rinshan, dora_count = map(int, hidden.wall_indices)
    status = np.zeros(136, dtype=np.uint8)
    status[max(0, live_start):min(136, live_end)] = 1
    status[122:136] = 2
    for indicator in range(min(5, dora_count)):
        status[130 - 2 * indicator] = 3
        status[131 - 2 * indicator] = 3
    if 0 <= rinshan < 136:
        status[rinshan:136] = np.maximum(status[rinshan:136], 2)

    flags = tuple(state.seat_flags)
    eligible = int(state.eligible_mask)
    raw_current = int(hidden.current_seat)
    current = (raw_current - dealer) % 4 if 0 <= raw_current < 4 else -1
    features = np.asarray([
        *encode_rank_boundary(_frame_mapping(state)).tolist(),
        *(bit for seat in seats for bit in _flag_bits(flags[seat])),
        *_one_hot(min(int(state.phase), 7), 8),
        *_one_hot(current, 4),
        *(float(bool(eligible & (1 << seat))) for seat in seats),
        live_start / 136.0,
        live_end / 136.0,
        rinshan / 136.0,
        dora_count / 5.0,
        float(state.live_wall_remaining) / 70.0,
    ], dtype=np.float16)
    if features.shape != (PRIVILEGED_STATE_FEATURES,):
        raise AssertionError("privileged state feature contract changed")
    return PrivilegedSnapshot(
        np.ascontiguousarray(tiles),
        wall_types,
        np.ascontiguousarray(status),
        np.ascontiguousarray(features),
    )


class ResidualMLP(nn.Module):
    def __init__(self, width: int, expansion: int):
        super().__init__()
        self.norm = nn.RMSNorm(width)
        self.up = nn.Linear(width, 2 * expansion, bias=False)
        self.down = nn.Linear(expansion, width, bias=False)

    def forward(self, value):
        gate, content = self.up(self.norm(value)).chunk(2, dim=-1)
        return value + self.down(torch.nn.functional.silu(gate) * content)


class PrivilegedActionQ(nn.Module):
    """Privileged Q network, optionally conditioned on detached actor states."""

    def __init__(
        self,
        width: int = 96,
        heads: int = 4,
        layers: int = 2,
        *,
        actor_hidden_width: int | None = None,
    ):
        super().__init__()
        action_cardinalities = (
            16, 256, 8, 8, 16, 4, 8,
            256, 256, 256, 256, 256, 256, 256, 256,
        )
        self.tile_identity = nn.Embedding(34, width)
        self.tile_features = nn.Linear(PRIVILEGED_TILE_CHANNELS, width)
        self.wall_tile = nn.Embedding(34, width)
        self.wall_status = nn.Embedding(4, width)
        self.wall_position = nn.Embedding(136, width)
        wall_layer = nn.TransformerEncoderLayer(
            width,
            heads,
            dim_feedforward=3 * width,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
            bias=False,
        )
        self.wall_encoder = nn.TransformerEncoder(
            wall_layer, num_layers=layers, enable_nested_tensor=False
        )
        self.state = nn.Sequential(
            nn.Linear(PRIVILEGED_STATE_FEATURES, width, bias=False),
            nn.SiLU(),
            nn.RMSNorm(width),
        )
        self.decision_seat = nn.Embedding(4, width)
        self.action_embedding = FactorEmbedding(action_cardinalities, width)
        self.actor_hidden = (
            nn.Sequential(
                nn.RMSNorm(2 * actor_hidden_width),
                nn.Linear(2 * actor_hidden_width, width, bias=False),
                nn.SiLU(),
                nn.RMSNorm(width),
            )
            if actor_hidden_width is not None else None
        )
        self.action_norm = nn.RMSNorm(width)
        self.memory_attention = nn.MultiheadAttention(
            width, heads, batch_first=True, bias=False
        )
        self.blocks = nn.ModuleList(
            ResidualMLP(width, 3 * width) for _ in range(layers)
        )
        self.head = nn.Sequential(nn.RMSNorm(width), nn.Linear(width, 1))

    def forward(
        self,
        tile_features,
        wall_types,
        wall_status,
        state_features,
        decision_seats,
        action_factors,
        action_lengths,
        actor_states=None,
        actor_action_states=None,
    ):
        batch, maximum = action_factors.shape[:2]
        device = action_factors.device
        tile_ids = torch.arange(34, device=device)
        tiles = self.tile_identity(tile_ids)[None] + self.tile_features(
            tile_features.float()
        )
        positions = torch.arange(136, device=device)
        wall = (
            self.wall_tile(wall_types.long())
            + self.wall_status(wall_status.long())
            + self.wall_position(positions)[None]
        )
        wall = self.wall_encoder(wall)
        state = (
            self.state(state_features.float())
            + self.decision_seat(decision_seats.long())
        )[:, None]
        memory = torch.cat((state, tiles, wall), dim=1)
        actions = self.action_embedding(action_factors)
        if self.actor_hidden is None:
            if actor_states is not None or actor_action_states is not None:
                raise ValueError("critic was not configured for actor hidden states")
        else:
            if actor_states is None or actor_action_states is None:
                raise ValueError("critic requires actor and actor-action states")
            if actor_action_states.shape[:2] != actions.shape[:2]:
                raise ValueError("actor action-state shape does not match legal actions")
            broadcast_state = actor_states[:, None].expand(
                -1, actor_action_states.shape[1], -1
            )
            actor_hidden = torch.cat(
                (broadcast_state, actor_action_states), dim=-1
            ).detach()
            actions = actions + self.actor_hidden(actor_hidden.float()).to(
                actions.dtype
            )
        cross, _ = self.memory_attention(
            self.action_norm(actions), memory, memory, need_weights=False
        )
        actions = actions + cross + state
        for block in self.blocks:
            actions = block(actions)
        valid = torch.arange(maximum, device=device)[None] < action_lengths[:, None]
        return self.head(actions).squeeze(-1).float().tanh().masked_fill(~valid, 0.0)


def privileged_batch(samples, snapshots, *, device: str):
    rows = tuple(samples)
    encoded = [row.encoded for row in rows]
    hidden = [snapshots[row.binding] for row in rows]
    lengths = np.asarray([len(row.action_factors) for row in encoded], dtype=np.int64)
    maximum = int(lengths.max())
    actions = np.zeros((len(rows), maximum, 15), dtype=np.int32)
    for index, row in enumerate(encoded):
        actions[index, :lengths[index]] = row.action_factors
    return {
        "tile_features": torch.as_tensor(
            np.stack([row.tile_features for row in hidden]),
            dtype=torch.float32,
            device=device,
        ),
        "wall_types": torch.as_tensor(
            np.stack([row.wall_types for row in hidden]),
            dtype=torch.long,
            device=device,
        ),
        "wall_status": torch.as_tensor(
            np.stack([row.wall_status for row in hidden]),
            dtype=torch.long,
            device=device,
        ),
        "state_features": torch.as_tensor(
            np.stack([row.state_features for row in hidden]),
            dtype=torch.float32,
            device=device,
        ),
        "decision_seats": torch.as_tensor(
            [row.decision_seat for row in encoded],
            dtype=torch.long,
            device=device,
        ),
        "action_factors": torch.as_tensor(
            actions, dtype=torch.int32, device=device
        ),
        "action_lengths": torch.as_tensor(
            lengths, dtype=torch.long, device=device
        ),
    }
