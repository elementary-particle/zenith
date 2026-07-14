"""Token schema 5: ordinary actor observations and private critic factors."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from functools import lru_cache
import json
import math


class Segment(IntEnum):
    EVENT = 1
    ACTOR_STATE = 2
    ACTOR_QUERY = 3
    CRITIC_PRIVATE = 4


class TokenKind(IntEnum):
    EVENT = 1
    SCORE = 2
    COUNTER = 3
    TILE_COUNT = 4
    MELD = 5
    RIVER = 6
    MASKED = 7
    QUERY = 8


MASKED_ID = 1
ABSENT_ID = 2
TOKEN_SCHEMA_VERSION = 5
MAX_FACTORS = 15


@dataclass(frozen=True, slots=True)
class Token:
    segment: int
    kind: int
    field: int = 0
    seat: int = 0
    tile_suit: int = 0
    tile_rank: int = 0
    tile_red: int = 0
    count: int = 0
    flag: int = 0
    visibility: int = 0
    numeric_field: int = 0
    numeric_value: float = 0.0

    def categorical(self) -> tuple[int, ...]:
        return (self.segment, self.kind, self.field, self.seat, self.tile_suit,
                self.tile_rank, self.tile_red, self.count, self.flag, self.visibility)


def numeric_features(token: Token) -> tuple[float, ...]:
    """Factor one semantic number into the shared Fourier representation."""
    return numeric_value_features(token.numeric_field, token.numeric_value)


@lru_cache(maxsize=4096)
def numeric_value_features(numeric_field: int, numeric_value: float) -> tuple[float, ...]:
    """Factor a semantic number without first materializing a :class:`Token`."""
    if numeric_field == 0:
        return (0.0,) * 8
    if numeric_field not in (1, 2):
        raise ValueError(f"unknown numeric field: {numeric_field}")
    value = float(numeric_value)
    periods = (100.0, 1_000.0, 10_000.0, 100_000.0) if numeric_field == 1 \
        else (2.0, 8.0, 32.0, 128.0)
    return tuple(
        feature
        for period in periods
        for feature in (
            math.sin(2.0 * math.pi * value / period),
            math.cos(2.0 * math.pi * value / period),
        )
    )


def relative_seat(observer: int, seat: int) -> int:
    return 0 if seat == 255 else (seat - observer) % 4 + 1


def tile_type_factors(tile_type: int, *, red: int = 0) -> tuple[int, int, int]:
    """Return the canonical suit/rank/red factors shared by every tile use."""
    tile_type = int(tile_type)
    red = int(red)
    if not 0 <= tile_type < 34:
        raise ValueError(f"tile type out of range: {tile_type}")
    if red not in (0, 1):
        raise ValueError(f"tile red status out of range: {red}")
    suit = tile_type // 9 + 1 if tile_type < 27 else 4
    rank = tile_type % 9 + 1 if tile_type < 27 else tile_type - 26
    return suit, rank, red


def physical_tile_factors(tile: int) -> tuple[int, int, int]:
    if tile == 255:
        return 0, 0, 0
    if not 0 <= tile < 136:
        raise ValueError(f"physical tile out of range: {tile}")
    tile_type, copy = divmod(tile, 4)
    red = int(tile_type in (4, 13, 22) and copy == 0)
    return tile_type_factors(tile_type, red=red)


def schema_json() -> str:
    return json.dumps({"version": TOKEN_SCHEMA_VERSION, "max_factors": MAX_FACTORS,
        "segments": {item.name: item.value for item in Segment},
        "token_kinds": {item.name: item.value for item in TokenKind}}, sort_keys=True)
