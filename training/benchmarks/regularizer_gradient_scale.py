#!/usr/bin/env python3
"""Measure PPO policy, entropy, and EMA-magnet gradient scales.

The audit replays fresh stochastic self-play from a checkpoint without taking
an optimizer step.  It uses the streaming trainer's logical-batch reductions:
advantages are centered and standardized once across the whole rollout, and
both full legal-action Shannon entropy (zero on singleton rows) and magnet KL
are averaged over all eligible policy rows.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

import riichi
from zenith_ppo.capabilities import configure
from zenith_ppo.checkpoint import resolve_latest, restore
from zenith_ppo.config import load
from zenith_ppo.model.factory import build_actor_critic
from zenith_ppo.ppo.loss import (
    legal_action_entropy,
    segmented_forward_kl,
)
from zenith_ppo.ppo.trainer import PPOTrainer
from zenith_ppo.rollout.native import NativeInferenceRunner


def _select(value, eligible):
    return value if eligible is None else value[eligible]


def _zeros(parameters):
    return [torch.zeros_like(parameter) for parameter in parameters]


def _add_gradients(target, gradients):
    with torch.no_grad():
        for accumulator, gradient in zip(target, gradients, strict=True):
            if gradient is not None:
                accumulator.add_(gradient)


def _dot(left, right):
    return sum(
        float((a.double() * b.double()).sum())
        for a, b in zip(left, right, strict=True)
    )


def _norm(vector):
    return math.sqrt(max(0.0, _dot(vector, vector)))


def _cosine(left, right):
    denominator = _norm(left) * _norm(right)
    return _dot(left, right) / denominator if denominator else float("nan")


def _scaled_norm(vector, coefficient):
    return abs(float(coefficient)) * _norm(vector)


def _adam_denominators(trainer, parameters):
    """Return saved-Adam RMS denominators, before incorporating this audit."""
    result = []
    beta2 = float(trainer.config["adam_beta2"])
    epsilon = float(trainer.config["adam_epsilon"])
    for parameter in parameters:
        state = trainer.actor_optimizer.state[parameter]
        square = state.get("exp_avg_sq")
        step = state.get("step", 0)
        step = float(step.item() if torch.is_tensor(step) else step)
        if square is None or step <= 0:
            result.append(torch.full_like(parameter, epsilon))
            continue
        correction = 1.0 - beta2 ** step
        result.append((square.float() / correction).sqrt().add_(epsilon))
    return result


def _precondition(vector, denominators):
    return [
        gradient.float() / denominator
        for gradient, denominator in zip(vector, denominators, strict=True)
    ]


def _component_report(vector, policy, coefficient, *, preconditioned=None,
                      policy_preconditioned=None):
    base_norm = _norm(vector)
    weighted_norm = _scaled_norm(vector, coefficient)
    policy_norm = _norm(policy)
    report = {
        "coefficient": float(coefficient),
        "base_gradient_norm": base_norm,
        "weighted_gradient_norm": weighted_norm,
        "weighted_to_policy_gradient_norm": (
            weighted_norm / policy_norm if policy_norm else float("nan")
        ),
        "cosine_with_policy_gradient": _cosine(vector, policy),
        "coefficient_for_policy_gradient_fraction": {
            str(fraction): (
                fraction * policy_norm / base_norm
                if base_norm else float("nan")
            )
            for fraction in (0.1, 0.25, 0.5, 1.0)
        },
    }
    if preconditioned is not None:
        base_preconditioned_norm = _norm(preconditioned)
        policy_preconditioned_norm = _norm(policy_preconditioned)
        report.update({
            "base_adam_preconditioned_norm": base_preconditioned_norm,
            "weighted_adam_preconditioned_norm": (
                abs(float(coefficient)) * base_preconditioned_norm
            ),
            "weighted_to_policy_adam_preconditioned_norm": (
                abs(float(coefficient)) * base_preconditioned_norm
                / policy_preconditioned_norm
                if policy_preconditioned_norm else float("nan")
            ),
            "adam_preconditioned_cosine_with_policy": _cosine(
                preconditioned, policy_preconditioned,
            ),
            "coefficient_for_policy_adam_preconditioned_fraction": {
                str(fraction): (
                    fraction * policy_preconditioned_norm
                    / base_preconditioned_norm
                    if base_preconditioned_norm else float("nan")
                )
                for fraction in (0.1, 0.25, 0.5, 1.0)
            },
        })
    return report


def run(args):
    started = perf_counter()
    configure("cpu-smoke" if args.device == "cpu" else "cuda-production")
    device = torch.device(args.device)
    config = load(args.config)
    model_config = dict(config.values["model"])
    model_config["context_tokens"] = config.values["encoding"]["context_tokens"]
    checkpoint = Path(args.checkpoint)
    checkpoint = resolve_latest(checkpoint) if (checkpoint / "latest").is_file() \
        else checkpoint
    restored = restore(checkpoint)
    model = build_actor_critic(model_config).to(device)
    model.load_state_dict(restored["model"])
    model.eval()

    ppo = dict(config.values["ppo"])
    trainer = PPOTrainer(
        model, ppo, device_type=device.type, use_bf16=False,
    )
    # Frozen random-initialization checkpoints intentionally have no optimizer
    # state. Their audit still has a well-defined raw gradient scale; the
    # preconditioned report falls back to Adam's configured epsilon.
    if restored["optimizer"]:
        trainer.load_optimizer_state_dict(restored["optimizer"])
    model.eval()

    engine = riichi.RolloutEngine(
        args.matches,
        master_seed=args.seed,
        num_threads=args.threads,
        context_tokens=model_config["context_tokens"],
        token_budget=args.token_budget,
    )
    match_ids = engine.reset_chunk(args.matches)
    engine.register_lineups(
        match_ids,
        [(0, 0, 0, 0)] * args.matches,
        [15] * args.matches,
    )
    generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    runner = NativeInferenceRunner(
        {0: model}, device=device, backend=args.backend,
        use_bf16=False, generator=generator,
        gae_lambda=float(ppo["gae_lambda"]), compile_cuda=False,
    )
    rollout_started = perf_counter()
    chunk = runner.run_chunk(engine)
    runner.prepare_training_chunk(chunk)
    rollout_seconds = perf_counter() - rollout_started
    batches = list(chunk.actor_minibatches(
        0, args.token_budget,
        max_padding_fraction=args.max_padding_fraction,
        backend=args.backend,
    ))

    raw = []
    rows = 0
    for batch in batches:
        eligible = batch.get("ppo_eligible")
        advantages = np.asarray(batch["raw_advantages"], dtype=np.float64)
        if eligible is not None:
            advantages = advantages[np.asarray(eligible, dtype=bool)]
        raw.append(advantages)
        rows += advantages.size
    raw = np.concatenate(raw)
    advantage_mean = float(raw.mean())
    advantage_deviation = float(raw.std())
    if advantage_deviation < 1e-8:
        raise RuntimeError("audit rollout has degenerate advantages")

    parameters = trainer.actor_parameter_list
    gradients = {
        "policy": _zeros(parameters),
        "entropy": _zeros(parameters),
        "magnet": _zeros(parameters),
    }
    losses = {
        "policy": 0.0,
        "entropy": 0.0,
        "magnet": 0.0,
    }
    entropy_rows = 0
    maximum_replay_log_ratio = 0.0
    gradient_started = perf_counter()
    for raw_batch in batches:
        batch = trainer._materialize_batch(raw_batch)
        output = model.forward_actor(**batch["model_inputs"])
        with torch.no_grad():
            magnet_output = trainer.ema_magnet.forward(batch["model_inputs"])
        eligible = batch.get("ppo_eligible")
        selected = _select(batch["selected"], eligible)
        old_logp = _select(batch["old_logp"], eligible).float()
        advantages = _select(batch["raw_advantages"], eligible).float()
        selected_logp = output.log_probabilities.index_select(0, selected).float()
        log_ratio = selected_logp - old_logp
        maximum_replay_log_ratio = max(
            maximum_replay_log_ratio, float(log_ratio.detach().abs().max()),
        )
        ratio = log_ratio.exp()
        policy = -(
            ratio * (advantages - advantage_mean)
        ).sum() / (rows * (advantage_deviation + 1e-8))

        entropy, _efficiency, applicable = legal_action_entropy(
            output.log_probabilities,
            batch["model_inputs"]["action_offsets"],
            batch["model_inputs"]["action_lengths"],
        )
        entropy = _select(entropy, eligible)
        applicable = _select(applicable, eligible)
        entropy_rows += int(applicable.sum())
        entropy_objective = -entropy.sum() / rows

        magnet_rows = segmented_forward_kl(
            magnet_output.log_probabilities,
            output.log_probabilities,
            batch["model_inputs"]["action_offsets"],
        )
        magnet = _select(magnet_rows, eligible).sum() / rows

        components = {
            "policy": policy,
            "entropy": entropy_objective,
            "magnet": magnet,
        }
        for index, (name, objective) in enumerate(components.items()):
            component_gradients = torch.autograd.grad(
                objective, parameters,
                retain_graph=index < len(components) - 1,
                allow_unused=True,
            )
            _add_gradients(gradients[name], component_gradients)
            losses[name] += float(objective.detach())

    gradient_seconds = perf_counter() - gradient_started
    denominators = _adam_denominators(trainer, parameters)
    preconditioned = {
        name: _precondition(vector, denominators)
        for name, vector in gradients.items()
    }
    policy_norm = _norm(gradients["policy"])
    policy_preconditioned_norm = _norm(preconditioned["policy"])
    entropy_coefficient = float(ppo["entropy_coefficient"])
    magnet_coefficient = float(ppo["magnet_kl_coefficient"])
    entropy_report = _component_report(
        gradients["entropy"], gradients["policy"], entropy_coefficient,
        preconditioned=preconditioned["entropy"],
        policy_preconditioned=preconditioned["policy"],
    )
    magnet_report = _component_report(
        gradients["magnet"], gradients["policy"], magnet_coefficient,
        preconditioned=preconditioned["magnet"],
        policy_preconditioned=preconditioned["policy"],
    )
    result = {
        "schema": "zenith-regularizer-gradient-scale-v1",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_id": restored["manifest"]["checkpoint_id"],
        "configuration": {
            "matches": args.matches,
            "seed": args.seed,
            "device": str(device),
            "backend": args.backend,
            "token_budget": args.token_budget,
            "entropy_coefficient": entropy_coefficient,
            "magnet_coefficient": magnet_coefficient,
            "magnet_half_life_matches": float(ppo["magnet_half_life_matches"]),
        },
        "sample": {
            "actor_rows": rows,
            "entropy_applicable_rows": entropy_rows,
            "actor_minibatches": len(batches),
            "advantage_mean": advantage_mean,
            "advantage_deviation": advantage_deviation,
            "maximum_replay_log_ratio": maximum_replay_log_ratio,
        },
        "losses": {
            "policy": losses["policy"],
            "entropy_loss_unweighted": losses["entropy"],
            "entropy_weighted": entropy_coefficient * losses["entropy"],
            "magnet_kl_unweighted": losses["magnet"],
            "magnet_weighted": magnet_coefficient * losses["magnet"],
        },
        "gradients": {
            "policy": {
                "gradient_norm": policy_norm,
                "adam_preconditioned_norm": policy_preconditioned_norm,
            },
            "entropy": entropy_report,
            "magnet": magnet_report,
            "entropy_magnet_cosine": _cosine(
                gradients["entropy"], gradients["magnet"],
            ),
            "entropy_magnet_adam_preconditioned_cosine": _cosine(
                preconditioned["entropy"], preconditioned["magnet"],
            ),
        },
        "timing_seconds": {
            "rollout": rollout_seconds,
            "gradient_audit": gradient_seconds,
            "total": perf_counter() - started,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("training/configs/default.toml"),
    )
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path("runs/ppo-bc2024-nova/checkpoints"),
    )
    # Four full self-play games already yield roughly 2k policy rows for the
    # stable regularizer estimates.  Increase this on GPU when estimating the
    # much noisier policy-gradient vector itself.
    parser.add_argument("--matches", type=int, default=4)
    parser.add_argument("--seed", type=int, default=81423)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--backend", default="eager")
    parser.add_argument("--token-budget", type=int, default=65_536)
    parser.add_argument("--max-padding-fraction", type=float, default=0.1)
    parser.add_argument(
        "--output", type=Path,
        default=Path("runs/diagnostics/nova-regularizer-gradient-scale.json"),
    )
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
