"""Construction of the single verified production actor-critic."""

from __future__ import annotations

from .structured_boundary import (
    PRODUCTION_ARCHITECTURE,
    VERIFIED_BC_ARCHITECTURE,
    VerifiedActorCritic,
)


def actor_critic_architecture_supported(architecture):
    return str(architecture) == PRODUCTION_ARCHITECTURE


def configured_architecture(config):
    return str(config.get(
        "architecture", PRODUCTION_ARCHITECTURE,
    ))


def build_actor_critic(config, *, architecture=None):
    selected = configured_architecture(config) \
        if architecture is None else str(architecture)
    if selected not in {PRODUCTION_ARCHITECTURE, VERIFIED_BC_ARCHITECTURE}:
        raise ValueError(
            f"unsupported actor/critic architecture {selected!r}; "
            f"production requires {PRODUCTION_ARCHITECTURE!r}"
        )
    return VerifiedActorCritic(config)


def checkpoint_architecture(model):
    architecture = str(getattr(model, "architecture_id", ""))
    if architecture != PRODUCTION_ARCHITECTURE:
        raise ValueError("model is not the verified production architecture")
    return architecture
