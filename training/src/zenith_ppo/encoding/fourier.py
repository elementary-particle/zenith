"""Schema-bound Fourier number embeddings computed in FP32."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class FourierDomain:
    minimum: float
    maximum: float
    periods: tuple[float, ...]
    unit: str

    def validate(self, value: float) -> float:
        value = float(value)
        if not math.isfinite(value) or not self.minimum <= value <= self.maximum:
            raise ValueError(f"{value!r} outside {self.unit} domain [{self.minimum},{self.maximum}]")
        return value


DOMAINS = {
    "score": FourierDomain(-100_000, 200_000, (100.0, 1_000.0, 10_000.0, 100_000.0), "points"),
    "count": FourierDomain(0, 255, (2.0, 8.0, 32.0, 128.0), "count"),
    "progress": FourierDomain(0, 1, (0.125, 0.25, 0.5, 1.0), "ratio"),
}


def features(values, domain: FourierDomain):
    import torch
    values = torch.as_tensor(values, dtype=torch.float32)
    if not torch.isfinite(values).all() or (values < domain.minimum).any() or (values > domain.maximum).any():
        raise ValueError(f"numeric values outside {domain.unit} domain")
    periods = torch.tensor(domain.periods, dtype=torch.float32, device=values.device)
    angles = values.unsqueeze(-1) * (2.0 * math.pi / periods)
    return torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1)
