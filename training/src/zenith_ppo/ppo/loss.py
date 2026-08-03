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


def conditional_family_entropy(
    log_probabilities, action_offsets, action_factors, action_lengths,
):
    """Normalized within-family entropy without changing family probability.

    Chi, pon, and open-kan candidates form one call family. Other action kinds
    each form their own family. Singleton families are ignored, so the
    objective explores tiles/call variants while exerting no direct pressure
    on call-versus-pass or riichi-versus-dama mass.
    """
    offsets = action_offsets.to(
        device=log_probabilities.device, dtype=torch.long,
    )
    lengths = action_lengths.to(
        device=log_probabilities.device, dtype=torch.long,
    )
    factors = action_factors.to(device=log_probabilities.device)
    if factors.ndim != 3 or factors.shape[0] != lengths.numel():
        raise ValueError("conditional entropy requires padded action factors")
    rows, maximum = factors.shape[:2]
    if offsets.numel() != rows + 1 or int(offsets[-1]) != log_probabilities.numel():
        raise ValueError("conditional entropy action offsets are inconsistent")
    segment_ids = torch.repeat_interleave(
        torch.arange(rows, device=log_probabilities.device),
        lengths,
        output_size=int(log_probabilities.numel()),
    )
    local = torch.arange(
        log_probabilities.numel(), device=log_probabilities.device,
    ) - offsets.index_select(0, segment_ids)
    logp = log_probabilities.float()
    kinds = factors[segment_ids, local, 0].long()
    families = torch.where(
        kinds.ge(3) & kinds.le(5), torch.full_like(kinds, 3), kinds,
    )
    group_ids = segment_ids * 11 + families
    group_width = rows * 11
    # Compute each family normalizer in log space.  Directly exponentiating
    # the global policy log-probability underflows when an entire family has
    # negligible policy mass, even though its *conditional* distribution is
    # still well-defined and relevant to this objective.
    group_max = logp.new_full((group_width,), -torch.inf).scatter_reduce(
        0, group_ids, logp, reduce="amax", include_self=True,
    )
    selected_max = group_max.index_select(0, group_ids)
    shifted = (logp - selected_max).exp()
    group_normalizer = logp.new_zeros(group_width).scatter_add(
        0, group_ids, shifted,
    )
    group_log_mass = group_max + group_normalizer.clamp_min(1e-30).log()
    group_counts = torch.zeros(
        group_width, dtype=torch.long, device=logp.device,
    ).scatter_add(0, group_ids, torch.ones_like(group_ids))
    conditional_logp = logp - group_log_mass.index_select(0, group_ids)
    conditional = conditional_logp.exp()
    group_entropy = logp.new_zeros(group_width).scatter_add(
        0, group_ids, -(conditional * conditional_logp),
    )
    group_applicable = group_counts > 1
    group_efficiency = torch.where(
        group_applicable,
        group_entropy / group_counts.clamp_min(2).float().log(),
        torch.zeros_like(group_entropy),
    ).clamp(0.0, 1.0)
    group_rows = torch.arange(
        rows, device=logp.device,
    ).repeat_interleave(11)
    efficiency = logp.new_zeros(rows).scatter_add(
        0, group_rows, group_efficiency,
    )
    family_count = torch.zeros(
        rows, dtype=torch.long, device=logp.device,
    ).scatter_add(0, group_rows, group_applicable.long())
    applicable = family_count > 0
    efficiency = torch.where(
        applicable, efficiency / family_count.clamp_min(1),
        torch.zeros_like(efficiency),
    )
    return efficiency.clamp(0.0, 1.0), applicable


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


def actor_loss(new_logp, old_logp, advantages, entropy, entropy_normalizers, *,
               ratio_clip=0.2, entropy_coefficient=0.01, kl_coefficient=0.0,
               magnet_kl=None, magnet_coefficient=0.0):
    tensors = (new_logp, old_logp, advantages, entropy, entropy_normalizers)
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
    normalizers = entropy_normalizers.float()
    applicable = normalizers > 0
    entropy_rows = int(applicable.sum())
    if entropy_rows:
        entropy_efficiency = (
            entropy.float()[applicable] / normalizers[applicable]
        ).mean()
    else:
        entropy_efficiency = entropy.float().sum() * 0.0
    entropy_loss = -float(entropy_coefficient) * entropy_efficiency
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
