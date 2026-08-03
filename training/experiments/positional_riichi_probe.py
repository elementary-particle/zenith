#!/usr/bin/env python3
"""Matched counterfactual probe of discard order around opponent riichi.

For a held-out decision, this probe finds the closest non-declarer discards on
opposite sides of an opponent's riichi declaration.  The later tile is safe
against that opponent because they passed it while in riichi; the earlier tile
is not certified safe.  Swapping those tile identities in both event history
and the exact river snapshot preserves the tile multiset, hand, legal actions,
scores, and all non-tile features.  A policy using the declaration boundary
should reverse its relative preference after the swap.

The declarer's entire own river is genbutsu regardless of timing and is
deliberately excluded.  "Not certified safe" is not a hidden-wait danger label.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass, replace
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
from zenith_ppo.encoding.schema import TokenKind
from zenith_ppo.model.actor_critic import ActorCritic


EVENT_DAHAI = 4
RIVER_RIICHI = 1


@dataclass(frozen=True)
class Probe:
    game: int
    boundary: str
    original: object
    swapped: object
    safe_action: int
    pre_action: int


def _tile_type(row) -> int:
    suit, rank = int(row[4]), int(row[5])
    return (suit - 1) * 9 + rank - 1 if suit < 4 else 26 + rank


def _ordinary_discards(encoded) -> dict[int, int]:
    return {
        int(row[1]): index
        for index, row in enumerate(encoded.action_factors)
        if int(row[0]) == 1 and 0 <= int(row[1]) < 34
    }


def _swap_tile_factors(factors, pairs):
    swapped = factors.copy()
    for left, right in pairs:
        temporary = swapped[left, 4:7].copy()
        swapped[left, 4:7] = swapped[right, 4:7]
        swapped[right, 4:7] = temporary
    return swapped


def _candidate_probe(example, *, game: int, seen: set):
    encoded = example.encoded
    factors = encoded.token_factors
    discards = _ordinary_discards(encoded)
    rivers = [
        (index, row) for index, row in enumerate(factors)
        if int(row[1]) == int(TokenKind.RIVER)
    ]
    observer = int(encoded.binding.seat)
    riichi_seats = {
        int(row[3]) for row in factors
        if int(row[1]) == int(TokenKind.COUNTER)
        and int(row[2]) == 9
        and int(row[3]) in (2, 3, 4)
        and int(row[8]) & 0b11
    }
    if len(riichi_seats) != 1:
        return
    declarer = next(iter(riichi_seats))
    events = [
        (index, row) for index, row in enumerate(factors)
        if int(row[1]) == int(TokenKind.EVENT)
    ]
    reaches = [
        position for position, (_, row) in enumerate(events)
        if int(row[2]) == 11 and int(row[3]) == declarer
    ]
    if len(reaches) != 1:
        return
    reach = reaches[0]
    before = [
        (index, row) for index, row in events[:reach]
        if int(row[2]) == EVENT_DAHAI and int(row[3]) != declarer
    ]
    after = [
        (index, row) for index, row in events[reach + 1:]
        if int(row[2]) == EVENT_DAHAI and int(row[3]) != declarer
    ]
    if not before or not after:
        return
    history_pre, pre_row = before[-1]
    history_safe, safe_row = after[0]
    pre_seat, safe_seat = int(pre_row[3]), int(safe_row[3])
    pre_tile, safe_tile = _tile_type(pre_row), _tile_type(safe_row)
    if pre_tile == safe_tile or pre_tile not in discards or safe_tile not in discards:
        return
    declarer_tiles = {
        _tile_type(row) for _, row in rivers if int(row[3]) == declarer
    }
    post_tiles = {
        _tile_type(row) for _, row in after
    }
    if pre_tile in declarer_tiles or safe_tile in declarer_tiles:
        return
    if pre_tile in post_tiles or sum(_tile_type(row) == safe_tile for _, row in after) != 1:
        return
    pre_rivers = [
        index for index, row in rivers
        if int(row[3]) == pre_seat and _tile_type(row) == pre_tile
    ]
    safe_rivers = [
        index for index, row in rivers
        if int(row[3]) == safe_seat and _tile_type(row) == safe_tile
    ]
    if len(pre_rivers) != 1 or len(safe_rivers) != 1:
        return
    observer = int(encoded.binding.seat)
    opponent = (observer + declarer - 1) % 4
    reach_identity = tuple(
        (int(row[2]), int(row[3]), _tile_type(row))
        for _, row in events[:reach + 1]
        if int(row[2]) in (EVENT_DAHAI, 11)
    )
    key = (game, observer, opponent, reach_identity)
    if key in seen:
        return
    seen.add(key)
    swapped = _swap_tile_factors(factors, (
        (pre_rivers[0], safe_rivers[0]),
        (history_pre, history_safe),
    ))
    yield Probe(
        game,
        "other_players_across_riichi",
        encoded,
        replace(encoded, token_factors=swapped),
        discards[safe_tile],
        discards[pre_tile],
    )


def _collect_probes(archive: Path, *, games: int, seed: int, replay_threads: int):
    corpus = ArchiveCorpus([archive])
    rng = random.Random(seed)
    members = rng.sample(corpus.members, min(int(games), len(corpus.members)))
    probes = []
    seen = set()
    rejected = 0
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
                probes.extend(_candidate_probe(
                    example, game=game, seen=seen,
                ))
    return probes, len(members), rejected


def _load_model(config_path: Path, checkpoint: Path):
    config = load(config_path)
    model_config = dict(config.values["model"])
    model_config["context_tokens"] = config.values["encoding"]["context_tokens"]
    model = ActorCritic(model_config)
    checkpoint = resolve_latest(checkpoint) if (checkpoint / "latest").is_file() \
        else checkpoint
    model.load_state_dict(restore(checkpoint)["model"])
    model.eval()
    return model, checkpoint


def _infer(model, probes, *, batch_size: int, disable_rope: bool):
    if disable_rope:
        model.backbone.rope_cos.fill_(1)
        model.backbone.rope_sin.zero_()
    rows = [row for probe in probes for row in (probe.original, probe.swapped)]
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            inputs = model_batch(rows[start:start + batch_size], device="cpu")
            output = model.forward_actor(**inputs)
            offsets = inputs["action_offsets"].tolist()
            outputs.extend(
                output.logits[offsets[index]:offsets[index + 1]].cpu().numpy()
                for index in range(len(offsets) - 1)
            )
    margins = []
    for index, probe in enumerate(probes):
        original, swapped = outputs[2 * index:2 * index + 2]
        margins.append((
            float(original[probe.safe_action] - original[probe.pre_action]),
            float(swapped[probe.safe_action] - swapped[probe.pre_action]),
        ))
    return np.asarray(margins, dtype=np.float64)


def _sigmoid(value):
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))


def _game_equal_ci(values, games, *, seed: int, samples: int = 5000):
    grouped = defaultdict(list)
    for value, game in zip(values, games, strict=True):
        grouped[int(game)].append(float(value))
    groups = [np.asarray(grouped[game], dtype=np.float64) for game in sorted(grouped)]
    if not groups:
        return [math.nan, math.nan]
    sums = np.asarray([group.sum() for group in groups])
    counts = np.asarray([len(group) for group in groups])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(groups), size=(samples, len(groups)))
    statistics = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return np.quantile(statistics, (0.025, 0.975)).tolist()


def _summary(margins, probes, *, seed: int):
    original, swapped = margins[:, 0], margins[:, 1]
    effect = original - swapped
    games = np.asarray([probe.game for probe in probes])
    metrics = {
        "probes": len(probes),
        "games_with_probes": int(len(set(games.tolist()))),
        "original_safe_preference_rate": float(np.mean(original > 0)),
        "swapped_safe_preference_rate": float(np.mean(swapped < 0)),
        "both_counterfactuals_correct_rate": float(
            np.mean((original > 0) & (swapped < 0))
        ),
        "correct_temporal_direction_rate": float(np.mean(effect > 0)),
        "mean_original_safe_margin_logits": float(np.mean(original)),
        "mean_swapped_same_identity_margin_logits": float(np.mean(swapped)),
        "mean_temporal_effect_logits": float(np.mean(effect)),
        "median_temporal_effect_logits": float(np.median(effect)),
        "mean_symmetrized_safety_margin_logits": float(np.mean(effect) / 2),
        "mean_original_pairwise_safe_probability": float(np.mean(_sigmoid(original))),
        "mean_swapped_pairwise_safe_probability": float(np.mean(_sigmoid(-swapped))),
    }
    metrics["game_clustered_95pct_ci"] = {
        "correct_temporal_direction_rate": _game_equal_ci(
            (effect > 0).astype(float), games, seed=seed,
        ),
        "mean_temporal_effect_logits": _game_equal_ci(
            effect, games, seed=seed + 1,
        ),
    }
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("training/configs/default.toml"))
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path("runs/behavior-cloning-rank-v/checkpoints"),
    )
    parser.add_argument(
        "--archive", type=Path,
        default=Path("datasets/tenhou-to-mjai/v2.0.0/2025.zip"),
    )
    parser.add_argument("--games", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20250803)
    parser.add_argument("--replay-threads", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    torch.set_num_threads(max(1, args.replay_threads))
    probes, sampled, rejected = _collect_probes(
        args.archive, games=args.games, seed=args.seed,
        replay_threads=args.replay_threads,
    )
    if not probes:
        raise RuntimeError("no qualifying riichi-boundary probes found")
    model, checkpoint = _load_model(args.config, args.checkpoint)
    normal = _infer(model, probes, batch_size=args.batch_size, disable_rope=False)
    # Reload so the ablation cannot leak into the regular evaluation.
    model, _ = _load_model(args.config, checkpoint)
    no_rope = _infer(model, probes, batch_size=args.batch_size, disable_rope=True)

    report = {
        "schema_version": 1,
        "checkpoint": str(checkpoint.resolve()),
        "archive": str(args.archive.resolve()),
        "sampled_games": sampled,
        "rejected_games": rejected,
        "seed": args.seed,
        "interpretation": {
            "safe": "closest non-declarer discard after riichi; the declarer passed it while riichi-locked",
            "comparison": "closest non-declarer discard before riichi, absent from the declarer's river and all post-riichi discards; not a hidden-wait danger label",
            "counterfactual": "swap only the two tile identities in event history and river snapshot, preserving the public tile multiset and all other inputs",
        },
        "normal_rope": {},
        "rope_disabled": {},
    }
    boundary = "other_players_across_riichi"
    report["normal_rope"][boundary] = _summary(
        normal, probes, seed=args.seed,
    )
    report["rope_disabled"][boundary] = _summary(
        no_rope, probes, seed=args.seed,
    )

    rendered = json.dumps(report, sort_keys=True, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
