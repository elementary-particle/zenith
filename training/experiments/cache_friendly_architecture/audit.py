"""Matched BC audit for an incrementally cacheable policy layout.

The production encoder puts mutable match-state tokens before the immutable
within-kyoku event stream, which prevents reuse of decoder K/V state.  The
production action head also lets every candidate revisit every history token
in each action-memory block.  This audit independently tests:

* an event-first causal layout, leaving a stable prefix within each kyoku;
* a fixed action memory containing hierarchical query summaries and 34 tiles.

All variants use the same games, initialization seed, optimizer, and update
order.  This is deliberately an experiment rather than a production config
switch: a checkpoint is only useful with the matching layout at inference.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import random
from statistics import NormalDist
from time import perf_counter

import numpy as np

from zenith_ppo.bc.data import ArchiveCorpus
from zenith_ppo.capabilities import configure
from zenith_ppo.cli.train_bc import _run_epoch
from zenith_ppo.config import load
from zenith_ppo.encoding.packing import model_batch, pack
from zenith_ppo.encoding.schema import (
    Segment,
    TokenKind,
    numeric_value_features,
)
from zenith_ppo.model.actor_critic import ActorCritic, ActionMemoryCore
from zenith_ppo.seeds import derive_seed


VARIANTS = (
    "baseline",
    "event_prefix",
    "stable_boundary",
    "summary_memory",
    "event_prefix_summary_memory",
)


def event_prefix_encoded(encoded):
    """Move immutable event tokens before every mutable current-state token."""
    factors = np.asarray(encoded.token_factors)
    # EVENT shares its segment value with KYOKU_STATE; token kind is the only
    # discriminator that does not accidentally move the mutable snapshot.
    event = factors[:, 1] == int(TokenKind.EVENT)
    order = np.concatenate((np.flatnonzero(event), np.flatnonzero(~event)))
    old_query = int(encoded.actor_query_index)
    query_matches = np.flatnonzero(order == old_query)
    if query_matches.size != 1:
        raise AssertionError("actor query was lost while changing token layout")
    reordered_factors = np.ascontiguousarray(factors[order])
    reordered_numeric = np.ascontiguousarray(
        np.asarray(encoded.token_numeric)[order]
    )
    if int(reordered_factors[-1, 0]) != int(Segment.ACTOR_QUERY):
        raise AssertionError("event-first layout must retain a final actor query")
    return replace(
        encoded,
        token_factors=reordered_factors,
        token_numeric=reordered_numeric,
        actor_query_index=int(query_matches[0]),
    )


def stable_boundary_encoded(encoded):
    """Use start-of-kyoku numerics in the permanently cacheable match prefix."""
    factors = np.asarray(encoded.token_factors)
    numeric = np.asarray(encoded.token_numeric).copy()
    boundary = np.asarray(encoded.rank_boundary_features, dtype=np.float32)
    round_index = int(np.argmax(boundary[8:24]))
    for index, row in enumerate(factors):
        if int(row[0]) != int(Segment.MATCH_STATE):
            continue
        kind, field, relative = map(int, row[1:4])
        if kind == int(TokenKind.SCORE):
            dealer_offset = (
                int(encoded.decision_seat) + relative - 1
            ) % 4
            value = float(boundary[dealer_offset]) * 24_000.0
            numeric[index] = numeric_value_features(1, value)
        elif kind == int(TokenKind.COUNTER) and field in {1, 2, 3, 4}:
            value = {
                1: round_index // 4,
                2: round_index % 4,
                3: float(boundary[24]) * 5.0,
                4: float(boundary[25]) * 5.0,
            }[field]
            numeric[index] = numeric_value_features(2, value)
    return replace(encoded, token_numeric=np.ascontiguousarray(numeric))


def transform_examples(examples, *, event_prefix, stable_boundary=False):
    result = []
    for example in examples:
        encoded = example.encoded
        if stable_boundary:
            encoded = stable_boundary_encoded(encoded)
        if event_prefix:
            encoded = event_prefix_encoded(encoded)
        result.append(replace(example, encoded=encoded))
    return tuple(result)


class SummaryActionMemoryCore(ActionMemoryCore):
    """Use fixed-size semantic summaries instead of revisiting raw history."""

    def memory(
        self, actor_state, strategic_features, token_factors, public_lengths,
        history, history_valid, tile_identity,
    ):
        from zenith_ppo.model.tile import current_public_tile_count_planes
        import torch

        planes = current_public_tile_count_planes(token_factors, public_lengths)
        counts = planes.flatten(2)[:, :, :34].transpose(1, 2) * 0.25
        tiles = self.tile_counts(counts).to(actor_state.dtype)
        tiles = tiles + tile_identity[None].to(actor_state.dtype)
        tiles = tiles + self.concealed_shape(planes).to(actor_state.dtype)
        tiles = self.tile_input_norm(tiles)
        strategy = self.strategy(strategic_features.float()).to(actor_state.dtype)
        state = self.state(torch.cat((actor_state, strategy), -1))

        positions = torch.arange(
            token_factors.shape[1], device=token_factors.device
        )[None].expand(token_factors.shape[0], -1)
        query = token_factors[..., 1].eq(int(TokenKind.QUERY)) & history_valid
        sentinel = torch.full_like(positions, token_factors.shape[1])
        query_indices = torch.where(query, positions, sentinel).sort(-1).values[:, :3]
        if bool(query_indices.ge(token_factors.shape[1]).any()):
            raise AssertionError("fixed action memory requires three summary queries")
        query_states = history.gather(
            1, query_indices[..., None].expand(-1, -1, history.shape[-1])
        )
        # The final query is already represented by `state`; retain the match
        # and kyoku summaries as two additional, semantically stable slots.
        memory = torch.cat((
            state[:, None], strategy[:, None], tiles, query_states[:, :2]
        ), 1)
        memory_valid = torch.ones(
            memory.shape[:2], dtype=torch.bool, device=memory.device
        )
        return state, memory, memory_valid


class SummaryMemoryActorCritic(ActorCritic):
    """Parameter-identical ActorCritic with a fixed-size action-memory path."""

    def __init__(self, config):
        super().__init__(config)
        # The subclass has exactly the same parameters and state-dict keys.
        # Changing the class in place preserves identical random initialization.
        self.action_memory.__class__ = SummaryActionMemoryCore


def event_prefix_tensors(
    token_factors, token_numeric, lengths, actor_query_indices,
):
    """Torch equivalent of :func:`event_prefix_encoded` for live inference."""
    import torch

    factors = torch.zeros_like(token_factors)
    numeric = None if token_numeric is None else torch.zeros_like(token_numeric)
    queries = torch.empty_like(actor_query_indices)
    for row in range(token_factors.shape[0]):
        length = int(lengths[row])
        source = token_factors[row, :length]
        event = source[:, 1].eq(int(TokenKind.EVENT))
        positions = torch.arange(length, device=source.device)
        order = torch.cat((positions[event], positions[~event]))
        factors[row, :length] = source.index_select(0, order)
        if numeric is not None:
            numeric[row, :length] = token_numeric[row, :length].index_select(
                0, order
            )
        matched = torch.nonzero(
            order.eq(actor_query_indices[row]), as_tuple=False
        ).flatten()
        if matched.numel() != 1:
            raise AssertionError("actor query was lost during live token reorder")
        queries[row] = matched[0]
    return factors, numeric, queries


class EventPrefixActorCritic(ActorCritic):
    """Inference wrapper for checkpoints trained with event-first tokens."""

    def forward_actor(
        self, token_factors, action_factors, actor_query_indices, action_offsets,
        token_numeric=None, *, lengths=None, **kwargs,
    ):
        if lengths is None:
            import torch
            lengths = torch.full(
                (token_factors.shape[0],), token_factors.shape[1],
                dtype=torch.long, device=token_factors.device,
            )
        factors, numeric, queries = event_prefix_tensors(
            token_factors, token_numeric, lengths, actor_query_indices
        )
        return super().forward_actor(
            factors, action_factors, queries, action_offsets,
            token_numeric=numeric, lengths=lengths, **kwargs,
        )


def stable_boundary_tensors(
    token_factors, token_numeric, rank_boundary_features, decision_seats,
):
    """Torch equivalent of :func:`stable_boundary_encoded`."""
    import torch

    numeric = token_numeric.clone()
    score_periods = torch.tensor(
        (100.0, 1_000.0, 10_000.0, 100_000.0),
        dtype=torch.float32, device=numeric.device,
    )
    count_periods = torch.tensor(
        (2.0, 8.0, 32.0, 128.0),
        dtype=torch.float32, device=numeric.device,
    )

    def features(value, periods):
        angle = 2.0 * torch.pi * value.double()[..., None] / periods.double()
        return torch.stack((angle.sin(), angle.cos()), -1).flatten(-2).to(
            numeric.dtype
        )

    for batch in range(token_factors.shape[0]):
        rows = token_factors[batch]
        match = rows[:, 0].eq(int(Segment.MATCH_STATE))
        score = match & rows[:, 1].eq(int(TokenKind.SCORE))
        relative = rows[score, 3].long()
        offsets = (decision_seats[batch].long() + relative - 1) % 4
        values = rank_boundary_features[batch, :4].index_select(0, offsets) \
            * 24_000.0
        numeric[batch, score] = features(values, score_periods)

        counter = match & rows[:, 1].eq(int(TokenKind.COUNTER))
        round_index = rank_boundary_features[batch, 8:24].argmax().float()
        counter_values = {
            1: torch.div(round_index, 4, rounding_mode="floor"),
            2: round_index.remainder(4),
            3: rank_boundary_features[batch, 24] * 5.0,
            4: rank_boundary_features[batch, 25] * 5.0,
        }
        for field, value in counter_values.items():
            selected = counter & rows[:, 2].eq(field)
            if bool(selected.any()):
                numeric[batch, selected] = features(value[None], count_periods)
    return numeric


class StableBoundaryActorCritic(ActorCritic):
    """Inference wrapper with a permanently stable match/event prefix."""

    def forward_actor(
        self, token_factors, action_factors, actor_query_indices, action_offsets,
        token_numeric=None, *, rank_boundary_features=None,
        decision_seats=None, **kwargs,
    ):
        if token_numeric is None or rank_boundary_features is None \
                or decision_seats is None:
            raise ValueError("stable boundary inference requires boundary inputs")
        numeric = stable_boundary_tensors(
            token_factors, token_numeric, rank_boundary_features, decision_seats
        )
        return super().forward_actor(
            token_factors, action_factors, actor_query_indices, action_offsets,
            token_numeric=numeric,
            rank_boundary_features=rank_boundary_features,
            decision_seats=decision_seats,
            **kwargs,
        )


def _variant_flags(name):
    if name not in VARIANTS:
        raise ValueError(f"unknown architecture variant {name!r}")
    return {
        "event_prefix": name.startswith("event_prefix"),
        "stable_boundary": name == "stable_boundary",
        "summary_memory": name.endswith("summary_memory"),
    }


def _model(values, variant, device, seed):
    import torch

    torch.manual_seed(seed)
    if torch.device(device).type == "cuda":
        torch.cuda.manual_seed_all(seed)
    config = dict(values["model"])
    config["context_tokens"] = values["encoding"]["context_tokens"]
    cls = SummaryMemoryActorCritic \
        if _variant_flags(variant)["summary_memory"] else ActorCritic
    return cls(config).to(device)


def _optimizer(model, bc, *, fused):
    import torch

    return torch.optim.AdamW(
        (
            {"params": list(model.actor_parameters()),
             "lr": float(bc["learning_rate"])},
            {"params": list(model.match_boundary_critic.parameters()),
             "lr": float(bc["boundary_rank_learning_rate"])},
        ),
        betas=(float(bc["adam_beta1"]), float(bc["adam_beta2"])),
        eps=float(bc["adam_epsilon"]),
        weight_decay=float(bc["weight_decay"]),
        fused=bool(fused),
    )


def _descriptors(manifest):
    return tuple((Path(archive), name, int(limit)) for archive, name, limit in manifest)


def _row_nll(model, examples, *, config, device, backend, use_bf16):
    import torch

    result = []
    lengths = [len(row.encoded.token_factors) for row in examples]
    packed = pack(lengths, int(config["token_budget"]), max_padding_fraction=0.10)
    model.eval()
    with torch.inference_mode():
        for indices in packed.batches:
            rows = tuple(examples[index] for index in indices)
            inputs = model_batch(
                [row.encoded for row in rows], device=device, backend=backend
            )
            targets = torch.tensor(
                [row.target for row in rows], dtype=torch.long, device=device
            )
            selected = inputs["action_offsets"][:-1] + targets
            with torch.autocast(
                device_type=torch.device(device).type,
                dtype=torch.bfloat16,
                enabled=use_bf16,
            ):
                actor = model.forward_actor(**inputs)
            nll = -actor.log_probabilities.index_select(0, selected)
            result.extend(zip(
                (row.family for row in rows),
                map(float, nll.float().cpu()),
                strict=True,
            ))
    return tuple(result)


def _paired_validation(
    models, corpus, manifest, *, bc, device, backend, use_bf16,
    replay_threads,
):
    """Evaluate paired NLL differences with games as independent clusters."""
    names = tuple(models)
    baseline = names[0]
    game_rows = []
    total_tokens = total_actions = total_rows = 0
    family_totals = {
        name: {} for name in names
    }
    timings = {name: 0.0 for name in names}
    for archive, member, limit in _descriptors(manifest):
        values = corpus.examples_many(
            (((archive, member), limit),), num_threads=replay_threads
        )[0]
        if isinstance(values, Exception):
            continue
        base_examples = tuple(values)
        total_rows += len(base_examples)
        total_tokens += sum(len(row.encoded.token_factors) for row in base_examples)
        total_actions += sum(len(row.encoded.action_factors) for row in base_examples)
        per_name = {}
        for name, model in models.items():
            flags = _variant_flags(name)
            examples = transform_examples(
                base_examples,
                event_prefix=flags["event_prefix"],
                stable_boundary=flags["stable_boundary"],
            )
            if torch_device_is_cuda(device):
                import torch
                torch.cuda.synchronize()
            started = perf_counter()
            rows = _row_nll(
                model, examples, config=bc, device=device, backend=backend,
                use_bf16=use_bf16,
            )
            if torch_device_is_cuda(device):
                import torch
                torch.cuda.synchronize()
            timings[name] += perf_counter() - started
            per_name[name] = rows
            for family, nll in rows:
                target = family_totals[name].setdefault(family, [0.0, 0])
                target[0] += nll
                target[1] += 1
        count = len(per_name[baseline])
        if any(len(per_name[name]) != count for name in names):
            raise AssertionError("paired validation row count differs by variant")
        game = {"rows": count, "differences": {}}
        baseline_nll = np.asarray([value for _, value in per_name[baseline]])
        for name in names[1:]:
            other = np.asarray([value for _, value in per_name[name]])
            game["differences"][name] = float((other - baseline_nll).sum())
        game_rows.append(game)

    def cluster_interval(name):
        # Ratio-estimator influence values, clustered by game.  The validation
        # manifest has thousands of games, so a normal cluster interval is
        # stable and avoids pretending decisions from one game are independent.
        rows = np.asarray([game["rows"] for game in game_rows], dtype=np.float64)
        sums = np.asarray([
            game["differences"][name] for game in game_rows
        ], dtype=np.float64)
        estimate = float(sums.sum() / rows.sum())
        influence = sums - estimate * rows
        games = len(game_rows)
        standard_error = math.sqrt(
            games / max(games - 1, 1) * float(np.square(influence).sum())
        ) / float(rows.sum())
        radius = NormalDist().inv_cdf(0.975) * standard_error
        return {
            "mean_nll_difference_vs_baseline": estimate,
            "cluster_standard_error": standard_error,
            "ci95": [estimate - radius, estimate + radius],
            "games": games,
            "rows": int(rows.sum()),
        }

    return {
        "paired": {name: cluster_interval(name) for name in names[1:]},
        "family_nll": {
            name: {
                family: total / count
                for family, (total, count) in families.items()
            }
            for name, families in family_totals.items()
        },
        "corpus": {
            "games": len(game_rows),
            "rows": total_rows,
            "mean_tokens": total_tokens / max(total_rows, 1),
            "mean_actions": total_actions / max(total_rows, 1),
        },
        "inference": {
            name: {
                "seconds": seconds,
                "rows_per_second": total_rows / max(seconds, 1e-9),
            }
            for name, seconds in timings.items()
        },
    }


def torch_device_is_cuda(device):
    return str(device).startswith("cuda")


def run(args):
    import torch
    from zenith_ppo.checkpoint import resolve_latest, restore

    resolved = load(args.config)
    values = resolved.values
    bc = dict(values["behavior_cloning"])
    bc["train_decisions"] = int(args.train_decisions)
    bc["validation_decisions"] = int(args.validation_decisions)
    profile = configure(values["run"]["profile"])
    device = "cuda" if profile.device == "cuda" else "cpu"
    use_bf16 = profile.precision == "bf16"
    seed = int(values["run"]["seed"])
    variants = tuple(args.variants)
    if variants[0] != "baseline":
        raise ValueError("the first variant must be baseline for paired comparisons")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    train_base = ArchiveCorpus(bc["train_archives"])
    validation_base = ArchiveCorpus(bc["validation_archives"])
    selection_seed = derive_seed(seed, "behavior_cloning_manifest")
    validation_seed = derive_seed(seed, "behavior_cloning_validation")
    train_candidates = list(train_base.members)
    validation_candidates = list(validation_base.members)
    random.Random(selection_seed).shuffle(train_candidates)
    random.Random(validation_seed).shuffle(validation_candidates)

    data_workers = int(args.data_workers)
    replay_threads = int(args.replay_threads)
    train_manifest = validation_manifest = None
    models = {}
    results = {}
    initial_state = None
    initial_checkpoint = None
    if args.initial_checkpoint:
        initial_checkpoint = Path(args.initial_checkpoint)
        if (initial_checkpoint / "latest").is_file():
            initial_checkpoint = resolve_latest(initial_checkpoint)
        initial_state = restore(initial_checkpoint)["model"]
    for variant in variants:
        flags = _variant_flags(variant)
        model = _model(values, variant, device, seed)
        if initial_state is not None:
            model.load_state_dict(initial_state)
        optimizer = _optimizer(model, bc, fused=use_bf16)
        def layout(examples, current=flags):
            return transform_examples(
                examples,
                event_prefix=current["event_prefix"],
                stable_boundary=current["stable_boundary"],
            )
        started = perf_counter()
        initial_validation = None
        if initial_state is not None:
            initial_validation, selected_validation = _run_epoch(
                model, None, validation_base,
                manifest=validation_manifest,
                candidates=(
                    validation_candidates
                    if validation_manifest is None else validation_manifest
                ),
                maximum=int(bc["validation_decisions"]),
                config=bc, device=device, backend=profile.attention,
                use_bf16=use_bf16, train=False, data_workers=data_workers,
                replay_batch=int(args.replay_batch), replay_threads=replay_threads,
                example_transform=layout,
            )
            if validation_manifest is None:
                validation_manifest = selected_validation
        train_metrics, selected_train = _run_epoch(
            model, optimizer, train_base,
            manifest=train_manifest,
            candidates=train_candidates if train_manifest is None else train_manifest,
            maximum=int(bc["train_decisions"]),
            config=bc, device=device, backend=profile.attention,
            use_bf16=use_bf16, train=True, data_workers=data_workers,
            replay_batch=int(args.replay_batch), replay_threads=replay_threads,
            example_transform=layout,
        )
        if train_manifest is None:
            train_manifest = selected_train
        validation_metrics, selected_validation = _run_epoch(
            model, None, validation_base,
            manifest=validation_manifest,
            candidates=(
                validation_candidates
                if validation_manifest is None else validation_manifest
            ),
            maximum=int(bc["validation_decisions"]),
            config=bc, device=device, backend=profile.attention,
            use_bf16=use_bf16, train=False, data_workers=data_workers,
            replay_batch=int(args.replay_batch), replay_threads=replay_threads,
            example_transform=layout,
        )
        if validation_manifest is None:
            validation_manifest = selected_validation
        results[variant] = {
            "flags": flags,
            "initial_validation": initial_validation,
            "train": train_metrics,
            "validation": validation_metrics,
            "elapsed_seconds": perf_counter() - started,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
        }
        models[variant] = model
        torch.save(model.state_dict(), output / f"{variant}.model.pt")
        print(json.dumps({variant: results[variant]}, sort_keys=True), flush=True)

    paired = _paired_validation(
        models, validation_base, validation_manifest, bc=bc, device=device,
        backend=profile.attention, use_bf16=use_bf16,
        replay_threads=replay_threads,
    )
    mean_tokens = paired["corpus"]["mean_tokens"]
    mean_actions = paired["corpus"]["mean_actions"]
    action_layers = int(values["model"]["action_memory_layers"])
    cache_evidence = {
        "stable_prefix": "all event tokens within the current kyoku",
        "production_action_memory_tokens": mean_tokens + 36.0,
        "summary_action_memory_tokens": 38.0,
        "action_cross_attention_interactions_per_row_production": (
            mean_actions * (mean_tokens + 36.0) * action_layers
        ),
        "action_cross_attention_interactions_per_row_summary": (
            mean_actions * 38.0 * action_layers
        ),
        "estimated_action_cross_attention_reduction": (
            1.0 - 38.0 / (mean_tokens + 36.0)
        ),
    }
    report = {
        "schema": "zenith-cache-friendly-architecture-audit-v1",
        "config": str(Path(args.config).resolve()),
        "seed": seed,
        "device": device,
        "train_decisions": int(bc["train_decisions"]),
        "validation_decisions": int(bc["validation_decisions"]),
        "initial_checkpoint": (
            str(initial_checkpoint.resolve()) if initial_checkpoint else None
        ),
        "variants": results,
        "paired_validation": paired,
        "cache_evidence": cache_evidence,
        "train_manifest": train_manifest,
        "validation_manifest": validation_manifest,
    }
    temporary = output / ".report.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output / "report.json")
    print(json.dumps({"report": str(output / "report.json")}, sort_keys=True))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="training/configs/default.toml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-decisions", type=int, default=1_000_000)
    parser.add_argument("--validation-decisions", type=int, default=200_000)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS,
                        default=VARIANTS)
    parser.add_argument("--initial-checkpoint")
    parser.add_argument("--data-workers", type=int, default=0)
    parser.add_argument("--replay-batch", type=int, default=8)
    parser.add_argument("--replay-threads", type=int, default=8)
    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
