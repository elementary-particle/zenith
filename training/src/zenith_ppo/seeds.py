"""Named deterministic RNG ownership with serializable state."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import base64
import pickle
import random
from typing import Any

STREAMS = (
    "env", "action", "visibility", "opponent", "minibatch",
    "actor_minibatch", "critic_minibatch", "evaluation", "histogram",
)


def derive_seed(root: int, name: str, version: int = 1) -> int:
    digest = sha256(f"zenith-seed-v{version}\0{root}\0{name}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


@dataclass
class SeedStreams:
    root: int
    version: int = 1

    def __post_init__(self) -> None:
        if self.root < 0:
            raise ValueError("seed root must be non-negative")
        self.python = {name: random.Random(derive_seed(self.root, name, self.version)) for name in STREAMS}
        self._numpy: dict[str, Any] = {}
        self._torch: dict[tuple[str, str], Any] = {}

    def python_rng(self, name: str) -> random.Random:
        return self.python[self._validate(name)]

    def numpy_rng(self, name: str):
        name = self._validate(name)
        if name not in self._numpy:
            import numpy as np
            self._numpy[name] = np.random.Generator(np.random.PCG64(derive_seed(self.root, name, self.version)))
        return self._numpy[name]

    def torch_generator(self, name: str, device: str = "cpu"):
        name = self._validate(name)
        key = (name, str(device))
        if key not in self._torch:
            import torch
            generator = torch.Generator(device=device)
            generator.manual_seed(derive_seed(self.root, name, self.version))
            self._torch[key] = generator
        return self._torch[key]

    def keyed_uniform(self, name: str, *parts: object) -> float:
        name = self._validate(name)
        digest = sha256((f"{derive_seed(self.root, name, self.version)}\0" +
                         "\0".join(map(str, parts))).encode()).digest()
        return int.from_bytes(digest[:8], "little") / 2**64

    def state_dict(self) -> dict[str, Any]:
        def encode(value):
            return base64.b64encode(
                pickle.dumps(value, protocol=5)
            ).decode()

        return {
            "root": self.root,
            "version": self.version,
            "python": {name: encode(rng.getstate()) for name, rng in self.python.items()},
            "numpy": {name: encode(rng.bit_generator.state) for name, rng in self._numpy.items()},
            "torch": {f"{name}\0{device}": base64.b64encode(gen.get_state().cpu().numpy().tobytes()).decode()
                      for (name, device), gen in self._torch.items()},
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state["root"] != self.root or state["version"] != self.version:
            raise ValueError("seed root/version mismatch")
        def decode(value):
            return pickle.loads(base64.b64decode(value))

        for name, value in state.get("python", {}).items():
            self.python_rng(name).setstate(decode(value))
        for name, value in state.get("numpy", {}).items():
            self.numpy_rng(name).bit_generator.state = decode(value)
        if state.get("torch"):
            import numpy as np
            import torch
            for key, value in state["torch"].items():
                name, device = key.split("\0", 1)
                data = np.frombuffer(base64.b64decode(value), dtype=np.uint8).copy()
                self.torch_generator(name, device).set_state(torch.from_numpy(data))

    @staticmethod
    def _validate(name: str) -> str:
        if name not in STREAMS:
            raise KeyError(f"unknown RNG stream {name!r}")
        return name
