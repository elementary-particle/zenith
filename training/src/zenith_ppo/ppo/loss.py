"""Finite-gated clipped PPO objective."""

from __future__ import annotations

from dataclasses import dataclass
import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class PPOLoss:
    total: torch.Tensor
    policy: torch.Tensor
    value: torch.Tensor
    entropy_loss: torch.Tensor
    approximate_kl: torch.Tensor
    clip_fraction: torch.Tensor


def ppo_loss(new_logp, old_logp, advantages, new_values, returns, entropy, *,
             ratio_clip=0.2, value_coefficient=0.5, entropy_coefficient=0.01):
    tensors = (new_logp, old_logp, advantages, new_values, returns, entropy)
    if any(not torch.isfinite(value).all() for value in tensors):
        raise FloatingPointError("non-finite PPO input")
    new_logp, old_logp, advantages = new_logp.float(), old_logp.float(), advantages.float()
    ratio = torch.exp(new_logp - old_logp)
    unclipped = ratio * advantages
    clipped = ratio.clamp(1 - ratio_clip, 1 + ratio_clip) * advantages
    policy = -torch.minimum(unclipped, clipped).mean()
    value = F.huber_loss(new_values.float(), returns.float(), reduction="mean")
    entropy_loss = -float(entropy_coefficient) * entropy.float().mean()
    total = policy + float(value_coefficient) * value + entropy_loss
    approximate_kl = ((ratio - 1) - (new_logp - old_logp)).mean()
    clip_fraction = ((ratio - 1).abs() > ratio_clip).float().mean()
    if not torch.isfinite(total): raise FloatingPointError("non-finite PPO loss")
    return PPOLoss(total, policy, value, entropy_loss, approximate_kl, clip_fraction)

