"""Independent compatibility axes for artifacts and native env boundaries."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json


class CompatibilityError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class CompatibilitySet:
    state_schema: int = 2
    event_schema: int = 2
    hand_analysis_schema: int = 2
    snapshot_schema: int = 1
    rules_profile: int = 2
    rng_profile: int = 1
    token_schema: int = 5
    action_schema: int = 2
    model_schema: int = 5
    reward_schema: int = 1
    curriculum_schema: int = 2
    sampler_schema: int = 1
    rating_schema: int = 1
    metric_schema: int = 3
    checkpoint_schema: int = 1

    @property
    def digest(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        return sha256(payload).hexdigest()

    def require(self, other: "CompatibilitySet", *, context: str = "artifact") -> None:
        mismatches = [
            f"{name}: expected {value!r}, got {getattr(other, name)!r}"
            for name, value in asdict(self).items()
            if getattr(other, name) != value
        ]
        if mismatches:
            raise CompatibilityError(f"incompatible {context}: " + "; ".join(mismatches))

    @classmethod
    def from_mapping(cls, value: dict[str, int]) -> "CompatibilitySet":
        unknown = set(value) - set(cls.__dataclass_fields__)
        if unknown:
            raise CompatibilityError(f"unknown compatibility axes: {sorted(unknown)}")
        return cls(**value)
