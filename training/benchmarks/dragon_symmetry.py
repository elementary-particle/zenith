#!/usr/bin/env python3
"""Audit within-state dragon exchangeability and cyclic policy symmetry.

The three dragons form a directed dora cycle, so the rotations
``white -> green -> red -> white`` and its inverse are rule symmetries.  The
Primary probes compare two legal dragon discards with matching observable
semantic signatures in the same information state.  A secondary probe
relabels every dragon occurrence and legal action, maps the resulting density
back by candidate index, and reports generalized Jensen-Shannon divergence.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import random

import numpy as np
import torch

from zenith_ppo.bc.data import ArchiveCorpus
from zenith_ppo.checkpoint import resolve_latest, restore
from zenith_ppo.config import load
from zenith_ppo.encoding.packing import model_batch
from zenith_ppo.model.factory import build_actor_critic


_FIRST_DRAGON = 31
_DRAGON_SUIT = 4
_FIRST_DRAGON_RANK = 5


def _rotate_range(values, mask, *, first: int, step: int):
    values[mask] = first + (values[mask] - first + step) % 3


def rotate_dragons(encoded, step: int):
    """Return an encoded information state under one cyclic dragon rotation."""
    step = int(step) % 3
    if not step:
        return encoded
    tokens = encoded.token_factors.copy()
    token_dragons = (
        (tokens[:, 4] == _DRAGON_SUIT)
        & (tokens[:, 5] >= _FIRST_DRAGON_RANK)
        & (tokens[:, 5] < _FIRST_DRAGON_RANK + 3)
    )
    _rotate_range(
        tokens[:, 5], token_dragons, first=_FIRST_DRAGON_RANK, step=step,
    )
    # Current concealed-count rows are canonically emitted in tile order.
    # Re-sort that order-insensitive snapshot block so the transformed input
    # is exactly what encoding the transformed game state would produce.
    concealed = np.flatnonzero(
        (tokens[:, 1] == 4) & (tokens[:, 2] == 1) & (tokens[:, 3] == 1)
    )
    if concealed.size:
        rows = tokens[concealed]
        order = np.lexsort((rows[:, 6], rows[:, 5], rows[:, 4]))
        tokens[concealed] = rows[order]

    actions = encoded.action_factors.copy()
    primary = actions[:, 1]
    primary_dragons = (
        (primary >= _FIRST_DRAGON) & (primary < _FIRST_DRAGON + 3)
    )
    _rotate_range(
        primary, primary_dragons, first=_FIRST_DRAGON, step=step,
    )
    action_dragons = (
        (actions[:, 3] == _DRAGON_SUIT)
        & (actions[:, 4] >= _FIRST_DRAGON_RANK)
        & (actions[:, 4] < _FIRST_DRAGON_RANK + 3)
    )
    _rotate_range(
        actions[:, 4], action_dragons,
        first=_FIRST_DRAGON_RANK, step=step,
    )
    semantic = actions[:, 7:11]
    semantic_types = semantic // 4
    semantic_dragons = (
        (semantic > 0)
        & (semantic_types >= _FIRST_DRAGON)
        & (semantic_types < _FIRST_DRAGON + 3)
    )
    semantic[semantic_dragons] = (
        (_FIRST_DRAGON
         + (semantic_types[semantic_dragons] - _FIRST_DRAGON + step) % 3)
        * 4
        + semantic[semantic_dragons] % 4
    )
    return replace(encoded, token_factors=tokens, action_factors=actions)


def _cyclic_stabilizer_actions(encoded):
    """Return three discard indices only when the state is C3-invariant."""
    discards = {
        int(row[1]): index
        for index, row in enumerate(encoded.action_factors)
        if int(row[0]) == 1
        and _FIRST_DRAGON <= int(row[1]) < _FIRST_DRAGON + 3
    }
    dragons = tuple(range(_FIRST_DRAGON, _FIRST_DRAGON + 3))
    if tuple(sorted(discards)) != dragons:
        return ()
    expected_actions = sorted(map(tuple, encoded.action_factors.tolist()))
    for step in (1, 2):
        transformed = rotate_dragons(encoded, step)
        # This is the strict stabilizer condition: after canonical encoding,
        # the complete information state is byte-identical and the legal
        # action set is identical up to candidate order.
        if not np.array_equal(
            transformed.token_factors, encoded.token_factors,
        ) or sorted(map(tuple, transformed.action_factors.tolist())) \
                != expected_actions:
            return ()
    return tuple(discards[dragon] for dragon in dragons)


def _collect_states(
    archive, *, games, states, stabilizer_states, seed, replay_threads,
):
    rng = random.Random(seed)
    candidates = []
    stabilizers = []
    rejected = 0
    with ArchiveCorpus([archive]) as corpus:
        members = rng.sample(corpus.members, min(int(games), len(corpus.members)))
        chunk = max(1, int(replay_threads) * 2)
        for start in range(0, len(members), chunk):
            rows = corpus.examples_many(
                [(member, None) for member in members[start:start + chunk]],
                num_threads=replay_threads,
            )
            for offset, examples in enumerate(rows):
                game = start + offset
                if isinstance(examples, Exception):
                    rejected += 1
                    continue
                for example in examples:
                    encoded = example.encoded
                    if len(encoded.action_factors) <= 1:
                        continue
                    candidates.append((game, encoded))
                    actions = _cyclic_stabilizer_actions(encoded)
                    if actions:
                        stabilizers.append((game, encoded, actions))
    rng.shuffle(candidates)
    rng.shuffle(stabilizers)
    return (
        candidates[:int(states)], stabilizers[:int(stabilizer_states)],
        len(members), rejected, len(candidates), len(stabilizers),
    )


def _load_model(config_path, checkpoint):
    config = load(config_path)
    model_config = dict(config.values["model"])
    model_config["context_tokens"] = config.values["encoding"]["context_tokens"]
    checkpoint = resolve_latest(checkpoint) if (checkpoint / "latest").is_file() \
        else checkpoint
    restored = restore(checkpoint)
    model = build_actor_critic(model_config)
    model.load_state_dict(restored["model"])
    return model.eval(), checkpoint, restored["manifest"]["checkpoint_id"]


def _infer(model, rows, *, batch_size, device):
    model.to(device)
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(rows), int(batch_size)):
            batch = rows[start:start + int(batch_size)]
            inputs = model_batch(
                batch, device=device, backend="eager", pin_memory=False,
            )
            output = model.forward_actor(
                **inputs, compute_entropy=False, compute_value=False,
            )
            offsets = inputs["action_offsets"].tolist()
            logp = output.log_probabilities.float().cpu().numpy()
            outputs.extend(
                logp[left:right].copy()
                for left, right in zip(offsets[:-1], offsets[1:], strict=True)
            )
    return outputs


def _state_metrics(log_densities):
    logp = np.stack(log_densities)
    probabilities = np.exp(logp)
    mixture = probabilities.mean(axis=0)
    log_mixture = np.log(np.maximum(mixture, np.finfo(np.float64).tiny))
    generalized_js = float(np.mean(np.sum(
        probabilities * (logp - log_mixture[None]), axis=1,
    )))
    tv = []
    maximum_delta = 0.0
    for left in range(3):
        for right in range(left + 1, 3):
            difference = np.abs(probabilities[left] - probabilities[right])
            tv.append(0.5 * float(difference.sum()))
            maximum_delta = max(maximum_delta, float(difference.max()))
    return {
        "generalized_js_nats": generalized_js,
        "maximum_pairwise_total_variation": max(tv),
        "maximum_action_probability_delta": maximum_delta,
        "argmax_disagreement": float(len(set(map(np.argmax, probabilities))) > 1),
        "reference_top_action_probability": float(probabilities[0].max()),
    }


def _stabilizer_metrics(log_density, actions):
    selected = np.asarray([log_density[index] for index in actions], dtype=np.float64)
    maximum = selected.max()
    conditional = np.exp(selected - maximum)
    conditional /= conditional.sum()
    full = np.exp(selected)
    greedy = int(np.argmax(log_density))
    return {
        "maximum_logit_gap": float(selected.max() - selected.min()),
        "maximum_conditional_probability_bias": float(
            np.abs(conditional - 1.0 / 3.0).max()
        ),
        "dragon_action_probability_mass": float(full.sum()),
        "dragon_contains_greedy_action": float(greedy in actions),
        "white_conditional_probability": float(conditional[0]),
        "green_conditional_probability": float(conditional[1]),
        "red_conditional_probability": float(conditional[2]),
    }


def _interval(values, games, *, seed, draws=2000):
    grouped = {}
    for value, game in zip(values, games, strict=True):
        grouped.setdefault(int(game), []).append(float(value))
    groups = [np.asarray(grouped[key]) for key in sorted(grouped)]
    if len(groups) < 2:
        return [math.nan, math.nan]
    rng = np.random.default_rng(seed)
    samples = np.empty(int(draws), dtype=np.float64)
    for index in range(int(draws)):
        selected = rng.integers(0, len(groups), size=len(groups))
        samples[index] = np.concatenate([groups[item] for item in selected]).mean()
    return np.quantile(samples, (0.025, 0.975)).tolist()


def _summary(metrics, games, *, seed):
    result = {"states": len(metrics), "games": len(set(map(int, games)))}
    for offset, key in enumerate(metrics[0] if metrics else ()):
        values = np.asarray([row[key] for row in metrics], dtype=np.float64)
        result[key] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "p95": float(np.quantile(values, 0.95)),
            "maximum": float(values.max()),
            "game_clustered_mean_95pct_ci": _interval(
                values, games, seed=seed + offset,
            ),
        }
    return result


def _stabilizer_summary(metrics, games, *, seed):
    result = {"states": len(metrics), "games": len(set(map(int, games)))}
    if not metrics:
        return result
    for offset, key in enumerate((
        "maximum_logit_gap",
        "maximum_conditional_probability_bias",
        "dragon_action_probability_mass",
        "dragon_contains_greedy_action",
        "white_conditional_probability",
        "green_conditional_probability",
        "red_conditional_probability",
    )):
        values = np.asarray([row[key] for row in metrics], dtype=np.float64)
        result[key] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "p95": float(np.quantile(values, 0.95)),
            "maximum": float(values.max()),
            "game_clustered_mean_95pct_ci": _interval(
                values, games, seed=seed + offset,
            ),
        }
    return result


def _impact_summary(metrics, games, *, seed):
    output = {
        "states": len(metrics),
        "argmax_disagreement_states": int(sum(
            row["argmax_disagreement"] for row in metrics
        )),
    }
    for threshold in (0.05, 0.10, 0.25):
        selected = [
            index for index, row in enumerate(metrics)
            if row["maximum_action_probability_delta"] >= threshold
        ]
        output[f"probability_delta_at_least_{threshold:.2f}"] = {
            "states": len(selected),
            "fraction": len(selected) / max(1, len(metrics)),
        }
    selected = [
        index for index, row in enumerate(metrics) if row["argmax_disagreement"]
    ]
    if selected:
        subset = [metrics[index] for index in selected]
        subset_games = [games[index] for index in selected]
        output["argmax_disagreement_subset"] = {
            key: _summary(subset, subset_games, seed=seed + offset)[key]
            for offset, key in enumerate((
                "maximum_pairwise_total_variation",
                "maximum_action_probability_delta",
                "reference_top_action_probability",
            ))
        }
    return output


def _filtered_stabilizer_summary(metrics, games, *, seed):
    output = {"all": _stabilizer_summary(metrics, games, seed=seed)}
    filters = {
        "probability_mass_at_least_0.25": lambda row: (
            row["dragon_action_probability_mass"] >= 0.25
        ),
        "probability_mass_at_least_0.50": lambda row: (
            row["dragon_action_probability_mass"] >= 0.50
        ),
        "contains_greedy_action": lambda row: bool(
            row["dragon_contains_greedy_action"]
        ),
    }
    for offset, (name, predicate) in enumerate(filters.items(), 1):
        selected = [index for index, row in enumerate(metrics) if predicate(row)]
        output[name] = _stabilizer_summary(
            [metrics[index] for index in selected],
            [games[index] for index in selected],
            seed=seed + offset * 10,
        ) if selected else {"states": 0, "games": 0}
    return output


def _paired_change(current, reference, games, *, keys, seed):
    if len(current) != len(reference) or len(current) != len(games):
        raise ValueError("paired symmetry metrics do not align")
    result = {"observations": len(current), "games": len(set(map(int, games)))}
    if not current:
        return result
    for offset, key in enumerate(keys):
        values = np.asarray([
            row[key] - baseline[key]
            for row, baseline in zip(current, reference, strict=True)
        ], dtype=np.float64)
        result[f"delta_{key}"] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "game_clustered_mean_95pct_ci": _interval(
                values, games, seed=seed + offset,
            ),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("training/configs/default.toml"))
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path("runs/ppo-bc2024-nova/checkpoints"),
    )
    parser.add_argument("--reference-checkpoint", type=Path)
    parser.add_argument(
        "--archive", type=Path,
        default=Path("datasets/tenhou-to-mjai/v2.0.0/2025.zip"),
    )
    parser.add_argument("--games", type=int, default=32)
    parser.add_argument("--states", type=int, default=1024)
    parser.add_argument("--stabilizer-states", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--replay-threads", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    torch.set_num_threads(max(1, int(args.replay_threads)))
    (
        selected, stabilizers, sampled_games, rejected,
        eligible, eligible_stabilizers,
    ) = _collect_states(
        args.archive, games=args.games, states=args.states, seed=args.seed,
        stabilizer_states=args.stabilizer_states,
        replay_threads=args.replay_threads,
    )
    if not selected:
        raise RuntimeError("held-out sample contained no multi-action decisions")
    model, checkpoint, checkpoint_id = _load_model(args.config, args.checkpoint)
    encoded = [row[1] for row in selected]
    rotations = [
        _infer(
            model, [rotate_dragons(row, step) for row in encoded],
            batch_size=args.batch_size, device=args.device,
        )
        for step in range(3)
    ]
    metrics = [
        _state_metrics([rotations[step][index] for step in range(3)])
        for index in range(len(encoded))
    ]
    games = [row[0] for row in selected]
    stabilizer_encoded = [row[1] for row in stabilizers]
    stabilizer_logp = _infer(
        model, stabilizer_encoded, batch_size=args.batch_size, device=args.device,
    ) if stabilizers else []
    stabilizer_metrics = [
        _stabilizer_metrics(logp, row[2])
        for logp, row in zip(stabilizer_logp, stabilizers, strict=True)
    ]
    stabilizer_games = [row[0] for row in stabilizers]
    report = {
        "probe": "exact-cyclic-dragon-density-symmetry-v2",
        "symmetry": "white->green->red->white",
        "checkpoint_id": checkpoint_id,
        "checkpoint_path": str(checkpoint),
        "archive": str(args.archive),
        "seed": args.seed,
        "sampled_games": sampled_games,
        "rejected_games": rejected,
        "eligible_multi_action_states": eligible,
        "selected_multi_action_states": len(selected),
        "eligible_cyclic_stabilizer_states": eligible_stabilizers,
        "selected_cyclic_stabilizer_states": len(stabilizers),
        "cyclic_density_symmetry": {
            "all": _summary(metrics, games, seed=args.seed),
            "impact": _impact_summary(metrics, games, seed=args.seed + 100),
        },
        "within_state_cyclic_stabilizer": _filtered_stabilizer_summary(
            stabilizer_metrics, stabilizer_games, seed=args.seed + 200,
        ),
    }
    if args.reference_checkpoint:
        reference_model, reference_checkpoint, reference_id = _load_model(
            args.config, args.reference_checkpoint,
        )
        reference_rotations = [
            _infer(
                reference_model, [rotate_dragons(row, step) for row in encoded],
                batch_size=args.batch_size, device=args.device,
            )
            for step in range(3)
        ]
        reference_metrics = [
            _state_metrics([
                reference_rotations[step][index] for step in range(3)
            ])
            for index in range(len(encoded))
        ]
        reference_stabilizer_logp = _infer(
            reference_model, stabilizer_encoded,
            batch_size=args.batch_size, device=args.device,
        ) if stabilizers else []
        reference_stabilizer_metrics = [
            _stabilizer_metrics(logp, row[2])
            for logp, row in zip(
                reference_stabilizer_logp, stabilizers, strict=True,
            )
        ]
        report["reference"] = {
            "checkpoint_id": reference_id,
            "checkpoint_path": str(reference_checkpoint),
            "within_state_cyclic_stabilizer": _filtered_stabilizer_summary(
                reference_stabilizer_metrics, stabilizer_games,
                seed=args.seed + 300,
            ),
            "cyclic_density_symmetry": {
                "all": _summary(
                    reference_metrics, games, seed=args.seed + 400,
                ),
                "impact": _impact_summary(
                    reference_metrics, games, seed=args.seed + 500,
                ),
            },
        }
        report["paired_change_from_reference"] = {
            "within_state_cyclic_stabilizer": _paired_change(
                stabilizer_metrics, reference_stabilizer_metrics,
                stabilizer_games,
                keys=(
                    "maximum_logit_gap",
                    "maximum_conditional_probability_bias",
                ),
                seed=args.seed + 600,
            ),
            "cyclic_density_symmetry": _paired_change(
                metrics, reference_metrics, games,
                keys=(
                    "generalized_js_nats",
                    "maximum_pairwise_total_variation",
                ),
                seed=args.seed + 700,
            ),
        }
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
