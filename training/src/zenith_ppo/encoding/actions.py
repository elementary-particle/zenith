"""Semantic action grouping and factorization."""

from __future__ import annotations

from dataclasses import dataclass
from .schema import physical_tile_factors, relative_seat


@dataclass(frozen=True, slots=True)
class EncodedActions:
    factors: tuple[tuple[int, ...], ...]
    representatives: tuple[int, ...]
    members: tuple[tuple[int, ...], ...]


def _factor_key(kind, primary, source_seat, tiles, aux, flags, observer):
    semantic_tiles = tuple(_semantic_tile(tile) for tile in tiles)
    padded = semantic_tiles[:4] + (255,) * (4 - len(semantic_tiles[:4]))
    suit, rank, red = physical_tile_factors(tiles[0]) if tiles else (0, 0, 0)
    if not 0 <= aux <= 0xFFFF or not 0 <= flags <= 0xFFFF:
        raise ValueError("action aux and flags must be unsigned 16-bit values")
    return (kind, primary, relative_seat(observer, source_seat),
            suit, rank, red, len(tiles), *padded,
            aux & 0xFF, aux >> 8, flags & 0xFF, flags >> 8)


def _key(row, observer):
    tiles = tuple(int(tile) for tile in row.get("tiles", ()) if int(tile) != 255)
    return _factor_key(int(row["kind"]), int(row.get("primary_tile_type", 255)),
        int(row.get("source_seat", 255)), tiles, int(row.get("aux", 0)),
        int(row.get("flags", 0)), observer)


def _semantic_tile(physical: int) -> int:
    tile_type, copy = divmod(int(physical), 4)
    red = tile_type in (4, 13, 22) and copy == 0
    return tile_type * 4 + (0 if red else 1)


def encode_actions(rows, *, observer: int) -> EncodedActions:
    if not rows:
        raise ValueError("a decision must have at least one valid action")
    groups = {}
    for native, row in enumerate(rows):
        groups.setdefault(_key(row, observer), []).append(native)
    ordered = sorted(groups.items(), key=lambda item: (item[1][0], item[0]))
    return EncodedActions(tuple(key for key, _ in ordered),
        tuple(members[0] for _, members in ordered), tuple(tuple(members) for _, members in ordered))


def encode_native_actions(rows, *, observer: int) -> EncodedActions:
    """Factor native action objects without materializing intermediate mappings."""
    if not rows:
        raise ValueError("a decision must have at least one valid action")
    groups = {}
    for native, row in enumerate(rows):
        tiles = tuple(int(tile) for tile in row.tiles if int(tile) != 255)
        primary = 255 if row.primary_tile_type is None else int(row.primary_tile_type)
        source = 255 if row.source_seat is None else int(row.source_seat)
        key = _factor_key(int(row.kind), primary, source, tiles,
                          int(row.aux), int(row.flags), observer)
        groups.setdefault(key, []).append(native)
    ordered = sorted(groups.items(), key=lambda item: (item[1][0], item[0]))
    return EncodedActions(tuple(key for key, _ in ordered),
        tuple(members[0] for _, members in ordered),
        tuple(tuple(members) for _, members in ordered))


def segmented_log_softmax(logits, offsets, *, layout=None):
    import torch

    _, lengths, segment_ids = layout or segment_layout(
        offsets, total=int(logits.shape[0]), device=logits.device
    )
    values = logits.float()
    segments = int(lengths.shape[0])
    maxima = torch.full(
        (segments,), -torch.inf, dtype=values.dtype, device=values.device
    ).scatter_reduce_(0, segment_ids, values, reduce="amax", include_self=True)
    shifted = values - maxima.index_select(0, segment_ids)
    sums = torch.zeros(
        segments, dtype=values.dtype, device=values.device
    ).scatter_add_(0, segment_ids, shifted.exp())
    return shifted - sums.index_select(0, segment_ids).log()


def segmented_entropy(log_probabilities, offsets, *, layout=None):
    import torch

    _, lengths, segment_ids = layout or segment_layout(
        offsets,
        total=int(log_probabilities.shape[0]),
        device=log_probabilities.device,
    )
    terms = -(log_probabilities.exp() * log_probabilities)
    return torch.zeros(
        int(lengths.shape[0]), dtype=terms.dtype, device=terms.device
    ).scatter_add_(0, segment_ids, terms)


def segmented_sample(
    log_probabilities, offsets, *, generator=None, deterministic=False, layout=None
):
    import torch

    _, lengths, segment_ids = layout or segment_layout(
        offsets,
        total=int(log_probabilities.shape[0]),
        device=log_probabilities.device,
    )
    scores = log_probabilities
    if not deterministic:
        uniform = torch.rand(
            scores.shape,
            dtype=scores.dtype,
            device=scores.device,
            generator=generator,
        )
        epsilon = torch.finfo(uniform.dtype).eps
        uniform = uniform.clamp(min=epsilon, max=1.0 - epsilon)
        scores = scores - torch.log(-torch.log(uniform))
    segments = int(lengths.shape[0])
    maxima = torch.full(
        (segments,), -torch.inf, dtype=scores.dtype, device=scores.device
    ).scatter_reduce_(0, segment_ids, scores, reduce="amax", include_self=True)
    indices = torch.arange(scores.numel(), device=scores.device, dtype=torch.long)
    candidates = torch.where(
        scores == maxima.index_select(0, segment_ids),
        indices,
        torch.full_like(indices, scores.numel()),
    )
    return torch.full(
        (segments,), scores.numel(), dtype=torch.long, device=scores.device
    ).scatter_reduce_(0, segment_ids, candidates, reduce="amin", include_self=True)


def segment_layout(offsets, *, total, device):
    import torch

    if isinstance(offsets, torch.Tensor):
        values = offsets.to(device=device, dtype=torch.long)
        if values.ndim != 1 or values.numel() < 2:
            raise ValueError("action offsets must be a one-dimensional boundary array")
    else:
        raw = tuple(int(value) for value in offsets)
        if len(raw) < 2 or raw[0] != 0 or raw[-1] != total:
            raise ValueError("action offsets do not span the candidate array")
        if any(end <= start for start, end in zip(raw[:-1], raw[1:])):
            raise ValueError("empty action segment")
        values = torch.tensor(raw, dtype=torch.long, device=device)
    lengths = values[1:] - values[:-1]
    segment_ids = torch.repeat_interleave(
        torch.arange(lengths.numel(), dtype=torch.long, device=device),
        lengths,
        output_size=total,
    )
    return values, lengths, segment_ids
