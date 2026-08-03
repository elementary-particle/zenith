"""Parameter-space EMA policy used as an adaptive PPO magnet."""

from __future__ import annotations

import math

import torch
from torch import nn


_STATE_VERSION = 1


class _ActorForward(nn.Module):
    """Expose only the actor path for stateless functional evaluation."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, **inputs):
        return self.model.forward_actor(**inputs)


class EMAMagnet:
    """Detached EMA of actor parameters with match-invariant decay.

    The actor stays part of the production model.  Keeping only detached
    parameter tensors avoids duplicating the privileged critics and lets
    ``functional_call`` evaluate the exact same actor implementation.
    """

    def __init__(self, model, *, half_life_matches: float):
        self.half_life_matches = float(half_life_matches)
        if not math.isfinite(self.half_life_matches) \
                or self.half_life_matches <= 0:
            raise ValueError("EMA magnet half-life must be finite and positive")
        self._actor = _ActorForward(model)
        self._parameters: dict[str, torch.Tensor] = {}
        self.updates = 0
        self.completed_matches = 0
        self.last_tau = 0.0
        self.reset(model)

    @staticmethod
    def _current(model):
        return dict(model.actor_named_parameters())

    @torch.no_grad()
    def reset(self, model) -> None:
        current = self._current(model)
        if not current:
            raise ValueError("EMA magnet requires actor parameters")
        self._parameters = {
            name: parameter.detach().clone()
            for name, parameter in current.items()
        }
        self.updates = 0
        self.completed_matches = 0
        self.last_tau = 0.0

    def forward(self, model_inputs):
        replacements = {
            f"model.{name}": parameter
            for name, parameter in self._parameters.items()
        }
        return torch.func.functional_call(
            self._actor,
            replacements,
            args=(),
            kwargs=model_inputs,
            strict=False,
        )

    @torch.no_grad()
    def copy_to(self, model) -> None:
        """Copy the detached EMA actor into a rollout-policy replica."""
        target = self._current(model)
        if target.keys() != self._parameters.keys():
            raise ValueError("EMA target actor parameter names changed")
        for name, parameter in target.items():
            value = self._parameters[name]
            if parameter.shape != value.shape:
                raise ValueError(f"EMA target shape changed for {name}")
            parameter.copy_(value)

    def tau_for_matches(self, matches: int) -> float:
        matches = int(matches)
        if matches <= 0:
            raise ValueError("EMA magnet update requires completed matches")
        # Numerically stable form of 1 - 2 ** (-matches / half_life).
        return -math.expm1(
            -math.log(2.0) * matches / self.half_life_matches
        )

    @torch.no_grad()
    def update(self, model, *, matches: int) -> float:
        current = self._current(model)
        if current.keys() != self._parameters.keys():
            raise ValueError("EMA magnet actor parameter names changed")
        tau = self.tau_for_matches(matches)
        for name, target in self._parameters.items():
            source = current[name].detach()
            if source.shape != target.shape:
                raise ValueError(f"EMA magnet shape changed for {name}")
            target.lerp_(source, tau)
        self.updates += 1
        self.completed_matches += int(matches)
        self.last_tau = tau
        return tau

    @torch.no_grad()
    def metrics(self, model) -> dict[str, float]:
        squared_difference = squared_reference = 0.0
        elements = 0
        for name, current in self._current(model).items():
            reference = self._parameters[name]
            difference = current.detach().float() - reference.float()
            squared_difference += float(difference.square().sum())
            squared_reference += float(reference.float().square().sum())
            elements += reference.numel()
        rms = math.sqrt(squared_difference / max(1, elements))
        reference_rms = math.sqrt(squared_reference / max(1, elements))
        return {
            "ema_tau": self.last_tau,
            "parameter_rms_distance": rms,
            "relative_parameter_rms_distance": rms / max(reference_rms, 1e-12),
        }

    def state_dict(self) -> dict:
        return {
            "version": _STATE_VERSION,
            "half_life_matches": self.half_life_matches,
            "updates": self.updates,
            "completed_matches": self.completed_matches,
            "last_tau": self.last_tau,
            "actor_parameters": {
                name: parameter.detach().clone()
                for name, parameter in self._parameters.items()
            },
        }

    @torch.no_grad()
    def load_state_dict(self, state, model) -> None:
        state = dict(state or {})
        if int(state.get("version", -1)) != _STATE_VERSION:
            raise ValueError("unsupported EMA magnet state version")
        saved_half_life = float(state.get("half_life_matches", float("nan")))
        if not math.isclose(
            saved_half_life, self.half_life_matches, rel_tol=0.0, abs_tol=1e-9
        ):
            raise ValueError("EMA magnet checkpoint half-life differs from config")
        current = self._current(model)
        saved = dict(state.get("actor_parameters") or {})
        if saved.keys() != current.keys():
            raise ValueError("EMA magnet checkpoint actor parameters differ")
        restored = {}
        for name, parameter in current.items():
            value = saved[name]
            if not torch.is_tensor(value) or value.shape != parameter.shape:
                raise ValueError(f"invalid EMA magnet parameter {name}")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"non-finite EMA magnet parameter {name}")
            restored[name] = value.detach().to(
                device=parameter.device, dtype=parameter.dtype
            ).clone()
        updates = int(state.get("updates", -1))
        completed_matches = int(state.get("completed_matches", -1))
        last_tau = float(state.get("last_tau", float("nan")))
        if updates < 0 or completed_matches < 0 \
                or not math.isfinite(last_tau) or not 0 <= last_tau <= 1:
            raise ValueError("invalid EMA magnet checkpoint counters")
        self._parameters = restored
        self.updates = updates
        self.completed_matches = completed_matches
        self.last_tau = last_tau
