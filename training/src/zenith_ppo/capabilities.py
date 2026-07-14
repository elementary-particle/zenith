"""Explicit execution profiles; requested CUDA never silently falls back."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import platform
import sys
import sysconfig


@dataclass(frozen=True, slots=True)
class Capabilities:
    profile: str
    device: str
    python: str
    platform: str
    torch: str
    cuda: str | None
    cudnn: int | None
    device_name: str | None
    bf16: bool
    deterministic: bool
    precision: str
    attention: str
    compilation: bool

    def as_dict(self):
        return asdict(self)


def detect(profile: str) -> Capabilities:
    if profile not in {"cpu-smoke", "cuda-strict", "cuda-production"}:
        raise ValueError(f"unknown execution profile {profile!r}")
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required by every execution profile") from exc
    wants_cuda = profile.startswith("cuda-")
    available = torch.cuda.is_available()
    if wants_cuda and not available:
        raise RuntimeError(f"{profile} requires CUDA; CPU fallback is prohibited")
    device = "cuda" if wants_cuda else "cpu"
    strict = profile != "cuda-production"
    bf16 = bool(wants_cuda and torch.cuda.is_bf16_supported())
    python_headers = Path(sysconfig.get_paths()["include"], "Python.h").is_file()
    return Capabilities(
        profile, device, sys.version.split()[0], platform.platform(), torch.__version__,
        torch.version.cuda if wants_cuda else None,
        torch.backends.cudnn.version() if wants_cuda else None,
        torch.cuda.get_device_name() if wants_cuda else None,
        bf16, strict, "bf16" if bf16 and not strict else "fp32",
        "sdpa" if wants_cuda and not strict else "eager", not strict and python_headers,
    )


def configure(profile: str) -> Capabilities:
    value = detect(profile)
    import torch
    torch.use_deterministic_algorithms(value.deterministic)
    if value.device == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = not value.deterministic
        torch.backends.cudnn.allow_tf32 = not value.deterministic
    return value
