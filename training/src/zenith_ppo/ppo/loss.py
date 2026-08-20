"""Finite-gated PPO policy objective."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ActorLoss:
    total: torch.Tensor
    policy: torch.Tensor
    entropy_loss: torch.Tensor
    entropy_efficiency: torch.Tensor
    entropy_rows: int
    kl_loss: torch.Tensor
    magnet_loss: torch.Tensor
    magnet_kl: torch.Tensor
    approximate_kl: torch.Tensor
    clip_fraction: torch.Tensor


def legal_action_entropy(
    log_probabilities, action_offsets, action_lengths,
):
    """Shannon entropy over every legal action in each ragged policy row.

    Raw entropy is the standard PPO regularizer. Entropy efficiency is
    normalized by the row's maximum entropy solely as a diagnostic.
    Singleton rows have zero entropy and are marked inapplicable.
    """
    offsets = action_offsets.to(
        device=log_probabilities.device, dtype=torch.long,
    )
    lengths = action_lengths.to(
        device=log_probabilities.device, dtype=torch.long,
    )
    rows = lengths.numel()
    if offsets.ndim != 1 or offsets.numel() != rows + 1 \
            or int(offsets[0]) != 0 \
            or int(offsets[-1]) != log_probabilities.numel():
        raise ValueError("legal-action entropy offsets are inconsistent")
    if bool(lengths.le(0).any()) \
            or not torch.equal(offsets[1:] - offsets[:-1], lengths):
        raise ValueError("legal-action entropy lengths are inconsistent")
    if log_probabilities.ndim != 1 \
            or not torch.isfinite(log_probabilities).all():
        raise FloatingPointError("non-finite legal-action log-probability")
    segment_ids = torch.repeat_interleave(
        torch.arange(rows, device=log_probabilities.device),
        lengths,
        output_size=int(log_probabilities.numel()),
    )
    logp = log_probabilities.float()
    entropy = logp.new_zeros(rows).scatter_add(
        0, segment_ids, -(logp.exp() * logp),
    )
    applicable = lengths > 1
    efficiency = torch.where(
        applicable,
        entropy / lengths.clamp_min(2).float().log(),
        torch.zeros_like(entropy),
    ).clamp(0.0, 1.0)
    return entropy.clamp_min(0.0), efficiency, applicable


def segmented_forward_kl(reference_logp, current_logp, action_offsets):
    """Compute KL(reference || current) for each ragged legal-action row."""
    reference = reference_logp.float()
    current = current_logp.float()
    offsets = action_offsets.to(device=current.device, dtype=torch.long)
    if reference.shape != current.shape or reference.ndim != 1:
        raise ValueError("magnet and current log-probabilities must match")
    if offsets.ndim != 1 or offsets.numel() < 2 \
            or int(offsets[0]) != 0 or int(offsets[-1]) != current.numel():
        raise ValueError("magnet KL action offsets are inconsistent")
    if not torch.isfinite(reference).all() or not torch.isfinite(current).all():
        raise FloatingPointError("non-finite magnet policy log-probability")
    lengths = offsets[1:] - offsets[:-1]
    if bool(lengths.le(0).any()):
        raise ValueError("magnet KL requires a legal action in every row")
    rows = torch.repeat_interleave(
        torch.arange(lengths.numel(), device=current.device),
        lengths,
        output_size=current.numel(),
    )
    terms = reference.exp() * (reference - current)
    result = current.new_zeros(lengths.numel()).scatter_add_(0, rows, terms)
    # Roundoff can produce tiny negative values for identical distributions.
    return result.clamp_min(0.0)


def actor_loss(new_logp, old_logp, advantages, entropy, entropy_efficiency,
               entropy_applicable, *,
               ratio_clip=0.2, entropy_coefficient=0.01, kl_coefficient=0.0,
               magnet_kl=None, magnet_coefficient=0.0):
    tensors = (new_logp, old_logp, advantages, entropy, entropy_efficiency)
    if any(not torch.isfinite(value).all() for value in tensors):
        raise FloatingPointError("non-finite PPO actor input")
    new_logp, old_logp, advantages = (
        new_logp.float(), old_logp.float(), advantages.float()
    )
    log_ratio = new_logp - old_logp
    ratio = torch.exp(log_ratio)
    unclipped = ratio * advantages
    clipped = ratio.clamp(1 - ratio_clip, 1 + ratio_clip) * advantages
    policy = -torch.minimum(unclipped, clipped).mean()
    approximate_kl = (torch.expm1(log_ratio) - log_ratio).mean()
    applicable = entropy_applicable.to(
        device=entropy.device, dtype=torch.bool,
    )
    if applicable.shape != entropy.shape \
            or entropy_efficiency.shape != entropy.shape:
        raise ValueError("entropy inputs must have matching policy rows")
    entropy_rows = int(applicable.sum())
    # Apply the coefficient to standard Shannon entropy over the complete
    # legal-action distribution. Singleton decisions contribute exact zero.
    entropy_mean = entropy.float().mean() if entropy.numel() \
        else entropy.float().sum()
    entropy_efficiency = entropy_efficiency.float().mean() \
        if entropy_efficiency.numel() else entropy_efficiency.float().sum()
    entropy_loss = -float(entropy_coefficient) * entropy_mean
    kl_loss = float(kl_coefficient) * approximate_kl
    if magnet_kl is None:
        magnet_kl = policy.new_zeros(())
    magnet_kl = magnet_kl.float()
    if magnet_kl.ndim or not torch.isfinite(magnet_kl):
        raise FloatingPointError("non-finite scalar EMA magnet KL")
    magnet_loss = float(magnet_coefficient) * magnet_kl
    total = policy + entropy_loss + kl_loss + magnet_loss
    clip_fraction = ((ratio - 1).abs() > ratio_clip).float().mean()
    if not torch.isfinite(total):
        raise FloatingPointError("non-finite PPO actor loss")
    return ActorLoss(
        total, policy, entropy_loss, entropy_efficiency, entropy_rows,
        kl_loss, magnet_loss, magnet_kl, approximate_kl, clip_fraction,
    )
