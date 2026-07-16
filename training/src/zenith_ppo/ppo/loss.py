"""Independent finite-gated actor PPO and oracle critic objectives."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class ActorLoss:
    total: torch.Tensor
    policy: torch.Tensor
    entropy_loss: torch.Tensor
    approximate_kl: torch.Tensor
    clip_fraction: torch.Tensor


@dataclass(frozen=True)
class CriticLoss:
    total: torch.Tensor
    score_mse: torch.Tensor
    score_mae: torch.Tensor
    score_rmse: torch.Tensor
    score_explained_variance: torch.Tensor
    score_clip_fraction: torch.Tensor
    rank_cross_entropy: torch.Tensor
    rank_accuracy: torch.Tensor
    rank_brier: torch.Tensor
    rank_utility_explained_variance: torch.Tensor


def _explained_variance(predictions, targets):
    predictions, targets = predictions.float(), targets.float()
    variance = torch.var(targets, unbiased=False)
    return torch.where(
        variance > 0,
        1.0 - torch.var(targets - predictions, unbiased=False) / variance,
        torch.zeros_like(variance),
    )


def actor_loss(new_logp, old_logp, advantages, entropy, *, ratio_clip=0.2,
               entropy_coefficient=0.01):
    tensors = (new_logp, old_logp, advantages, entropy)
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
    entropy_loss = -float(entropy_coefficient) * entropy.float().mean()
    total = policy + entropy_loss
    approximate_kl = (torch.expm1(log_ratio) - log_ratio).mean()
    clip_fraction = ((ratio - 1).abs() > ratio_clip).float().mean()
    if not torch.isfinite(total):
        raise FloatingPointError("non-finite PPO actor loss")
    return ActorLoss(total, policy, entropy_loss, approximate_kl, clip_fraction)


def critic_loss(score_values, old_score_values, score_returns,
                rank_logits, rank_targets, rank_returns, *,
                score_value_scale=10.0, value_clip=0.0,
                value_coefficient=0.5):
    tensors = (
        score_values, old_score_values, score_returns,
        rank_logits, rank_returns,
    )
    if any(not torch.isfinite(value).all() for value in tensors):
        raise FloatingPointError("non-finite oracle critic input")
    scale = float(score_value_scale)
    clip = float(value_clip)
    if scale <= 0 or clip < 0:
        raise ValueError("score scale must be positive and value clip non-negative")
    score_values = score_values.float()
    old_score_values = old_score_values.float()
    score_returns = score_returns.float()
    if clip > 0:
        clipped_values = old_score_values + (score_values - old_score_values).clamp(
            -clip, clip
        )
        raw_unclipped = (score_values / scale - score_returns / scale).square()
        raw_clipped = (clipped_values / scale - score_returns / scale).square()
        score_mse = torch.maximum(raw_unclipped, raw_clipped).mean()
        score_clip_fraction = (
            (score_values - old_score_values).abs() > clip
        ).float().mean()
    else:
        score_mse = F.mse_loss(score_values / scale, score_returns / scale)
        score_clip_fraction = score_values.new_zeros(())
    residual = score_values - score_returns
    score_mae = residual.abs().mean()
    score_rmse = residual.square().mean().sqrt()
    score_ev = _explained_variance(score_values, score_returns)

    rank_targets = rank_targets.long()
    if rank_targets.ndim != 1 or ((rank_targets < 0) | (rank_targets > 3)).any():
        raise ValueError("rank targets must be terminal placement classes in [0,3]")
    rank_cross_entropy = F.cross_entropy(rank_logits.float(), rank_targets)
    probabilities = rank_logits.float().softmax(-1)
    one_hot = F.one_hot(rank_targets, 4).float()
    rank_accuracy = (probabilities.argmax(-1) == rank_targets).float().mean()
    rank_brier = (probabilities - one_hot).square().sum(-1).mean()
    utilities = rank_logits.new_tensor((1.0, 1 / 3, -1 / 3, -1.0))
    expected = (probabilities * utilities).sum(-1)
    rank_ev = _explained_variance(expected, rank_returns)
    # Both tasks receive equal weight from update one.  Curriculum rank weights
    # affect policy advantages only, never oracle supervision.
    total = float(value_coefficient) * 0.5 * (score_mse + rank_cross_entropy)
    if not torch.isfinite(total):
        raise FloatingPointError("non-finite oracle critic loss")
    return CriticLoss(
        total, score_mse, score_mae, score_rmse, score_ev,
        score_clip_fraction, rank_cross_entropy, rank_accuracy, rank_brier,
        rank_ev,
    )


# Public compatibility name now denotes the policy-only PPO objective.
ppo_loss = actor_loss
