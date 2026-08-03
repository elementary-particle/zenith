#!/usr/bin/env python3
"""Matched probes for strategic information extraction by the BC actor.

The suite tests three computations that are available to, but not explicitly
provided by, the architecture:

* dora: infer the successor of an indicator and preserve that action tile;
* suji: bind a numbered discard identity to the correct riichi player's river;
* kabe: combine four visible copies with adjacent terminal-tile safety;
* placement strategy: change riichi-versus-dama preference with late-game score.

Each tile probe swaps identities in a matched public state so hand shape, legal
actions, and current visible tile counts stay fixed.  Score probes edit both
the public score tokens and the actor's direct strategic-feature input.
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
from zenith_ppo.encoding.critic import POINTS_PER_REWARD
from zenith_ppo.encoding.packing import model_batch
from zenith_ppo.encoding.schema import TokenKind, numeric_value_features
from zenith_ppo.model.actor_critic import ActorCritic


EVENT_DAHAI = 4
RIVER_RIICHI = 1
SNAPSHOT_SEAT_FLAGS_FIELD = 9


@dataclass(frozen=True)
class PairProbe:
    game: int
    kind: str
    original: object
    swapped: object
    favorable_action: int
    other_action: int


@dataclass(frozen=True)
class ScoreProbe:
    game: int
    kind: str
    leader: object
    trailer: object
    riichi_action: int
    dama_action: int


def _tile_type(row) -> int:
    suit, rank = int(row[4]), int(row[5])
    return (suit - 1) * 9 + rank - 1 if suit < 4 else 26 + rank


def _ordinary_discards(encoded) -> dict[int, int]:
    return {
        int(row[1]): index
        for index, row in enumerate(encoded.action_factors)
        if int(row[0]) == 1 and 0 <= int(row[1]) < 34
    }


def _riichi_seats(factors) -> set[int]:
    return {
        int(row[3]) for row in factors
        if int(row[1]) == int(TokenKind.COUNTER)
        and int(row[2]) == SNAPSHOT_SEAT_FLAGS_FIELD
        and int(row[3]) in (2, 3, 4)
        and int(row[8]) & 0b11
    }


def _swap_tiles(factors, pairs):
    swapped = factors.copy()
    for left, right in pairs:
        temporary = swapped[left, 4:7].copy()
        swapped[left, 4:7] = swapped[right, 4:7]
        swapped[right, 4:7] = temporary
    return swapped


def _matching_history(factors, *, seat: int, tile: int):
    return [
        index for index, row in enumerate(factors)
        if int(row[1]) == int(TokenKind.EVENT)
        and int(row[2]) == EVENT_DAHAI
        and int(row[3]) == seat
        and _tile_type(row) == tile
    ]


def _dora_successor(indicator: int) -> int:
    indicator = int(indicator)
    if indicator < 27:
        suit = indicator // 9
        return suit * 9 + (indicator % 9 + 1) % 9
    if indicator < 31:
        return 27 + (indicator - 27 + 1) % 4
    return 31 + (indicator - 31 + 1) % 3


def _dora_probe(example, *, game: int, seen: set):
    encoded = example.encoded
    factors = encoded.token_factors
    if _riichi_seats(factors):
        return None
    indicators = [
        (index, row) for index, row in enumerate(factors)
        if int(row[1]) == int(TokenKind.TILE_COUNT) and int(row[2]) == 3
    ]
    if len(indicators) != 1:
        return None
    indicator_index, indicator_row = indicators[0]
    indicator = _tile_type(indicator_row)
    dora = _dora_successor(indicator)
    discards = _ordinary_discards(encoded)
    if dora not in discards:
        return None
    observer = int(encoded.binding.seat)
    hand_key = (
        game, observer, bytes(encoded.rank_boundary_features), indicator,
    )
    if hand_key in seen:
        return None
    rivers = [
        (index, row) for index, row in enumerate(factors)
        if int(row[1]) == int(TokenKind.RIVER)
    ]
    for river_index, river_row in rivers:
        replacement_indicator = _tile_type(river_row)
        other = _dora_successor(replacement_indicator)
        if replacement_indicator == indicator or other == dora or other not in discards:
            continue
        seat = int(river_row[3])
        if sum(
            _tile_type(row) == replacement_indicator and int(row[3]) == seat
            for _, row in rivers
        ) != 1:
            continue
        history = _matching_history(
            factors, seat=seat, tile=replacement_indicator,
        )
        if len(history) != 1:
            continue
        seen.add(hand_key)
        swapped = _swap_tiles(factors, ((indicator_index, river_index),))
        # Keep the historical discard consistent with the edited river.  The
        # exact current public tile multiset is preserved: one indicator and
        # one river tile exchange identities.
        swapped[history[0], 4:7] = swapped[river_index, 4:7]
        return PairProbe(
            game, "dora_indicator", encoded,
            replace(encoded, token_factors=swapped),
            discards[other], discards[dora],
        )
    return None


def _terminal_suji(generator: int) -> set[int]:
    generator = int(generator)
    if generator >= 27 or generator % 9 not in (3, 4, 5):
        return set()
    return {generator - 3, generator + 3}


def _kabe_protected_terminal(blocker: int) -> int | None:
    blocker = int(blocker)
    if blocker >= 27:
        return None
    suit, rank = divmod(blocker, 9)
    if rank in (1, 2):
        return suit * 9
    if rank in (6, 7):
        return suit * 9 + 8
    return None


def _current_visible_counts(factors):
    counts = np.zeros(34, dtype=np.int16)
    concealed = np.zeros(34, dtype=np.int16)
    for row in factors:
        suit, rank = int(row[4]), int(row[5])
        if suit < 1 or rank < 1:
            continue
        tile = _tile_type(row)
        kind, field, flags = int(row[1]), int(row[2]), int(row[8])
        if kind == int(TokenKind.TILE_COUNT) and field == 1:
            value = max(1, int(row[7]))
            counts[tile] += value
            concealed[tile] += value
        elif kind == int(TokenKind.TILE_COUNT) and field == 3:
            counts[tile] += 1
        elif kind == int(TokenKind.RIVER) and not flags & 2:
            counts[tile] += 1
        elif kind == int(TokenKind.MELD):
            counts[tile] += 1
    return counts, concealed


def _kabe_probe(example, *, game: int, seen: set):
    encoded = example.encoded
    factors = encoded.token_factors
    riichi_seats = _riichi_seats(factors)
    if len(riichi_seats) != 1:
        return None
    target = next(iter(riichi_seats))
    target_tiles = {
        _tile_type(row) for row in factors
        if int(row[1]) == int(TokenKind.RIVER) and int(row[3]) == target
    }
    target_suji = set().union(*(_terminal_suji(tile) for tile in target_tiles))
    events = [
        row for row in factors if int(row[1]) == int(TokenKind.EVENT)
    ]
    reaches = [
        index for index, row in enumerate(events)
        if int(row[2]) == 11 and int(row[3]) == target
    ]
    if len(reaches) != 1:
        return None
    passed_tiles = {
        _tile_type(row) for row in events[reaches[0] + 1:]
        if int(row[2]) == EVENT_DAHAI and int(row[3]) != target
    }
    certified_safe = target_tiles | target_suji | passed_tiles
    discards = _ordinary_discards(encoded)
    counts, concealed = _current_visible_counts(factors)
    full = [
        tile for tile in range(27)
        if counts[tile] == 4 and concealed[tile] == 0
        and _kabe_protected_terminal(tile) in discards
    ]
    absent = [
        tile for tile in range(27)
        if counts[tile] == 0 and concealed[tile] == 0
        and _kabe_protected_terminal(tile) in discards
    ]
    for original_blocker in full:
        favorable = _kabe_protected_terminal(original_blocker)
        for swapped_blocker in absent:
            other = _kabe_protected_terminal(swapped_blocker)
            if favorable is None or other is None or favorable == other:
                continue
            if favorable in certified_safe or other in certified_safe:
                continue
            observer = int(encoded.binding.seat)
            key = (
                game, observer, bytes(encoded.rank_boundary_features),
                tuple(int(value) for value in concealed),
            )
            if key in seen:
                return None
            seen.add(key)
            swapped = factors.copy()
            original_rows = [
                index for index, row in enumerate(factors)
                if int(row[4]) > 0 and _tile_type(row) == original_blocker
            ]
            swapped_rows = [
                index for index, row in enumerate(factors)
                if int(row[4]) > 0 and _tile_type(row) == swapped_blocker
            ]
            original_identity = factors[original_rows[0], 4:7].copy()
            suit = swapped_blocker // 9 + 1
            rank = swapped_blocker % 9 + 1
            swapped_identity = np.asarray((suit, rank, 0), dtype=swapped.dtype)
            swapped[original_rows, 4:7] = swapped_identity
            if swapped_rows:
                swapped[swapped_rows, 4:7] = original_identity
            return PairProbe(
                game, "kabe_visible_four", encoded,
                replace(encoded, token_factors=swapped),
                discards[favorable], discards[other],
            )
    return None


def _suji_probe(example, *, game: int, seen: set):
    encoded = example.encoded
    factors = encoded.token_factors
    riichi_seats = _riichi_seats(factors)
    if len(riichi_seats) != 1:
        return None
    seat = next(iter(riichi_seats))
    target_river = [
        (index, row) for index, row in enumerate(factors)
        if int(row[1]) == int(TokenKind.RIVER) and int(row[3]) == seat
    ]
    if not any(int(row[8]) & RIVER_RIICHI for _, row in target_river):
        return None
    other_rivers = [
        (index, row) for index, row in enumerate(factors)
        if int(row[1]) == int(TokenKind.RIVER)
        and int(row[3]) in (1, 2, 3, 4)
        and int(row[3]) != seat
    ]
    discards = _ordinary_discards(encoded)
    target_tiles = {_tile_type(row) for _, row in target_river}
    target_suji = set().union(*(_terminal_suji(tile) for tile in target_tiles))
    for target_index, target_row in target_river:
        target_generator = _tile_type(target_row)
        if not _terminal_suji(target_generator):
            continue
        target_history = _matching_history(
            factors, seat=seat, tile=target_generator,
        )
        if len(target_history) != 1 or sum(
            _tile_type(row) == target_generator for _, row in target_river
        ) != 1:
            continue
        for other_index, other_row in other_rivers:
            other_seat = int(other_row[3])
            other_generator = _tile_type(other_row)
            if target_generator == other_generator or not _terminal_suji(other_generator):
                continue
            other_history = _matching_history(
                factors, seat=other_seat, tile=other_generator,
            )
            if len(other_history) != 1 or sum(
                _tile_type(row) == other_generator and int(row[3]) == other_seat
                for _, row in other_rivers
            ) != 1:
                continue
            swapped_target_tiles = target_tiles - {target_generator} | {other_generator}
            swapped_target_suji = set().union(*(
                _terminal_suji(tile) for tile in swapped_target_tiles
            ))
            original_options = sorted(
                _terminal_suji(target_generator) & discards.keys()
            )
            swapped_options = sorted(
                _terminal_suji(other_generator) & discards.keys()
            )
            for favorable in original_options:
                for other in swapped_options:
                    if favorable == other or favorable in target_tiles or other in target_tiles:
                        continue
                    if favorable not in target_suji or other in target_suji:
                        continue
                    if favorable in swapped_target_suji or other not in swapped_target_suji:
                        continue
                    observer = int(encoded.binding.seat)
                    opponent = (observer + seat - 1) % 4
                    river_identity = tuple(
                        (_tile_type(row), int(row[8])) for _, row in target_river
                    )
                    key = (
                        game, observer, opponent,
                        bytes(encoded.rank_boundary_features), river_identity,
                    )
                    if key in seen:
                        return None
                    seen.add(key)
                    swapped = _swap_tiles(factors, (
                        (target_index, other_index),
                        (target_history[0], other_history[0]),
                    ))
                    return PairProbe(
                        game, "suji_seat_attribution", encoded,
                        replace(encoded, token_factors=swapped),
                        discards[favorable], discards[other],
                    )
    return None


def _with_actor_relative_scores(encoded, scores):
    factors = encoded.token_factors
    numeric = encoded.token_numeric.copy()
    for index, row in enumerate(factors):
        if int(row[1]) == int(TokenKind.SCORE) and int(row[2]) == 1:
            relative = int(row[3]) - 1
            numeric[index] = numeric_value_features(1, scores[relative])
    boundary = encoded.rank_boundary_features.copy()
    actor_from_dealer = int(encoded.decision_seat)
    for dealer_offset in range(4):
        actor_relative = (dealer_offset - actor_from_dealer) % 4
        boundary[dealer_offset] = scores[actor_relative] / POINTS_PER_REWARD
    return replace(
        encoded, token_numeric=numeric, rank_boundary_features=boundary,
    )


def _score_probe(example, *, game: int):
    encoded = example.encoded
    discards = {
        int(row[1]): index for index, row in enumerate(encoded.action_factors)
        if int(row[0]) == 1
    }
    riichi = {
        int(row[1]): index for index, row in enumerate(encoded.action_factors)
        if int(row[0]) == 2
    }
    shared = sorted(discards.keys() & riichi.keys())
    if not shared:
        return None
    tile = shared[0]
    round_index = int(np.argmax(encoded.rank_boundary_features[8:24]))
    if round_index == 7:
        phase = "south4"
    elif round_index >= 8:
        phase = "overtime"
    else:
        phase = "other_rounds"
    return ScoreProbe(
        game,
        f"riichi_score_{phase}",
        _with_actor_relative_scores(encoded, (50_000, 16_000, 17_000, 17_000)),
        _with_actor_relative_scores(encoded, (10_000, 35_000, 28_000, 27_000)),
        riichi[tile], discards[tile],
    )


def _collect(archive: Path, *, games: int, seed: int, replay_threads: int):
    corpus = ArchiveCorpus([archive])
    members = random.Random(seed).sample(
        corpus.members, min(int(games), len(corpus.members)),
    )
    pair_probes = []
    score_probes = []
    seen_dora, seen_kabe, seen_suji = set(), set(), set()
    rejected = 0
    chunk = max(1, int(replay_threads) * 2)
    for start in range(0, len(members), chunk):
        batches = corpus.examples_many(
            [(member, None) for member in members[start:start + chunk]],
            num_threads=replay_threads,
        )
        for offset, examples in enumerate(batches):
            game = start + offset
            if isinstance(examples, Exception):
                rejected += 1
                continue
            for example in examples:
                dora = _dora_probe(example, game=game, seen=seen_dora)
                if dora is not None:
                    pair_probes.append(dora)
                kabe = _kabe_probe(example, game=game, seen=seen_kabe)
                if kabe is not None:
                    pair_probes.append(kabe)
                suji = _suji_probe(example, game=game, seen=seen_suji)
                if suji is not None:
                    pair_probes.append(suji)
                score = _score_probe(example, game=game)
                if score is not None:
                    score_probes.append(score)
    return pair_probes, score_probes, len(members), rejected


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


def _infer_rows(model, rows, *, batch_size: int):
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
    return outputs


def _pair_margins(model, probes, *, batch_size: int):
    rows = [row for probe in probes for row in (probe.original, probe.swapped)]
    outputs = _infer_rows(model, rows, batch_size=batch_size)
    return np.asarray([
        (
            outputs[2 * index][probe.favorable_action]
            - outputs[2 * index][probe.other_action],
            outputs[2 * index + 1][probe.favorable_action]
            - outputs[2 * index + 1][probe.other_action],
        )
        for index, probe in enumerate(probes)
    ], dtype=np.float64)


def _score_margins(model, probes, *, batch_size: int):
    rows = [row for probe in probes for row in (probe.leader, probe.trailer)]
    outputs = _infer_rows(model, rows, batch_size=batch_size)
    return np.asarray([
        (
            outputs[2 * index][probe.riichi_action]
            - outputs[2 * index][probe.dama_action],
            outputs[2 * index + 1][probe.riichi_action]
            - outputs[2 * index + 1][probe.dama_action],
        )
        for index, probe in enumerate(probes)
    ], dtype=np.float64)


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


def _summary(margins, probes, *, score_probe: bool, seed: int):
    first, second = margins[:, 0], margins[:, 1]
    effect = second - first if score_probe else first - second
    games = np.asarray([probe.game for probe in probes])
    result = {
        "probes": len(probes),
        "games_with_probes": len(set(games.tolist())),
        "correct_direction_rate": float(np.mean(effect > 0)),
        "mean_effect_logits": float(np.mean(effect)),
        "median_effect_logits": float(np.median(effect)),
        "game_clustered_95pct_ci": {
            "correct_direction_rate": _game_equal_ci(
                (effect > 0).astype(float), games, seed=seed,
            ),
            "mean_effect_logits": _game_equal_ci(
                effect, games, seed=seed + 1,
            ),
        },
    }
    if score_probe:
        result.update({
            "leader_riichi_minus_dama_logits": float(np.mean(first)),
            "trailer_riichi_minus_dama_logits": float(np.mean(second)),
        })
    else:
        result.update({
            "original_favorable_margin_logits": float(np.mean(first)),
            "swapped_same_identity_margin_logits": float(np.mean(second)),
            "both_counterfactuals_correct_rate": float(
                np.mean((first > 0) & (second < 0))
            ),
        })
    return result


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
    parser.add_argument("--games", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20250804)
    parser.add_argument("--replay-threads", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    torch.set_num_threads(max(1, args.replay_threads))
    pair_probes, score_probes, sampled, rejected = _collect(
        args.archive, games=args.games, seed=args.seed,
        replay_threads=args.replay_threads,
    )
    model, checkpoint = _load_model(args.config, args.checkpoint)
    pair_margins = _pair_margins(model, pair_probes, batch_size=args.batch_size)
    score_margins = _score_margins(model, score_probes, batch_size=args.batch_size)
    report = {
        "schema_version": 1,
        "checkpoint": str(checkpoint.resolve()),
        "archive": str(args.archive.resolve()),
        "sampled_games": sampled,
        "rejected_games": rejected,
        "seed": args.seed,
        "probes": {},
    }
    for kind in sorted({probe.kind for probe in pair_probes}):
        selected = [index for index, probe in enumerate(pair_probes) if probe.kind == kind]
        report["probes"][kind] = _summary(
            pair_margins[selected], [pair_probes[index] for index in selected],
            score_probe=False, seed=args.seed,
        )
    for kind in sorted({probe.kind for probe in score_probes}):
        selected = [index for index, probe in enumerate(score_probes) if probe.kind == kind]
        report["probes"][kind] = _summary(
            score_margins[selected], [score_probes[index] for index in selected],
            score_probe=True, seed=args.seed,
        )
    rendered = json.dumps(report, sort_keys=True, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
