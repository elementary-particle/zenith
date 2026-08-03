"""Shared public tile-count projection for policy-side structured encoders."""

from __future__ import annotations

import torch

from ..encoding.schema import TokenKind
from ..encoding.state import SNAPSHOT_RIVER_CALLED


TILE_PLANE_CHANNELS = 10
TILE_GRID_SIZE = 36


def tile_count_planes(token_factors, lengths):
    """Project sparse public tile tokens onto source-aware 4-by-9 planes."""
    batch, tokens, _ = token_factors.shape
    positions = torch.arange(tokens, device=token_factors.device)[None]
    valid = positions < lengths[:, None]
    kind = token_factors[..., 1]
    field = token_factors[..., 2]
    seat = token_factors[..., 3]
    suit = token_factors[..., 4]
    rank = token_factors[..., 5]
    tile_valid = valid & suit.ge(1) & suit.le(4) & rank.ge(1) & rank.le(9)

    concealed = kind.eq(4) & field.eq(1)
    dora = kind.eq(4) & field.eq(3)
    discard = kind.eq(1) & field.eq(4) & seat.ge(1) & seat.le(4)
    exposed = kind.eq(1) & field.ge(5) & field.le(10) \
        & seat.ge(1) & seat.le(4)
    represented = tile_valid & (concealed | dora | discard | exposed)
    source = torch.where(
        concealed,
        torch.zeros_like(field),
        torch.where(
            dora,
            torch.ones_like(field),
            torch.where(discard, seat + 1, seat + 5),
        ),
    )
    tile = (suit - 1) * 9 + rank - 1
    scatter_index = source.clamp(0, TILE_PLANE_CHANNELS - 1) * TILE_GRID_SIZE \
        + tile.clamp(0, TILE_GRID_SIZE - 1)
    multiplicity = torch.where(
        concealed, token_factors[..., 7].clamp_min(1), torch.ones_like(field)
    ).to(torch.float32)
    values = multiplicity * represented.to(multiplicity.dtype)
    planes = torch.zeros(
        batch,
        TILE_PLANE_CHANNELS * TILE_GRID_SIZE,
        dtype=values.dtype,
        device=values.device,
    )
    planes.scatter_add_(1, scatter_index, values)
    return planes.view(batch, TILE_PLANE_CHANNELS, 4, 9)


def current_public_tile_count_planes(token_factors, lengths):
    """Project exact current hand, dora, rivers, and melds by relative seat."""
    batch, tokens, _ = token_factors.shape
    positions = torch.arange(tokens, device=token_factors.device)[None]
    valid = positions < lengths[:, None]
    kind = token_factors[..., 1]
    field = token_factors[..., 2]
    seat = token_factors[..., 3]
    suit = token_factors[..., 4]
    rank = token_factors[..., 5]
    flags = token_factors[..., 8]
    tile_valid = valid & suit.ge(1) & suit.le(4) & rank.ge(1) & rank.le(9)

    concealed = kind.eq(int(TokenKind.TILE_COUNT)) & field.eq(1)
    dora = kind.eq(int(TokenKind.TILE_COUNT)) & field.eq(3)
    river = kind.eq(int(TokenKind.RIVER)) \
        & flags.bitwise_and(SNAPSHOT_RIVER_CALLED).eq(0) \
        & seat.ge(1) & seat.le(4)
    meld = kind.eq(int(TokenKind.MELD)) & seat.ge(1) & seat.le(4)
    represented = tile_valid & (concealed | dora | river | meld)
    source = torch.where(
        concealed,
        torch.zeros_like(field),
        torch.where(
            dora,
            torch.ones_like(field),
            torch.where(river, seat + 1, seat + 5),
        ),
    )
    tile = (suit - 1) * 9 + rank - 1
    scatter_index = source.clamp(0, TILE_PLANE_CHANNELS - 1).long() \
        * TILE_GRID_SIZE + tile.clamp(0, TILE_GRID_SIZE - 1).long()
    multiplicity = torch.where(
        concealed, token_factors[..., 7].clamp_min(1), torch.ones_like(field)
    ).to(torch.float32)
    values = multiplicity * represented.to(multiplicity.dtype)
    planes = torch.zeros(
        batch,
        TILE_PLANE_CHANNELS * TILE_GRID_SIZE,
        dtype=values.dtype,
        device=values.device,
    )
    planes.scatter_add_(1, scatter_index, values)
    return planes.view(batch, TILE_PLANE_CHANNELS, 4, 9)
