"""Matched architectural repair screen initialized from a PPO checkpoint.

The variants isolate two remaining hypotheses after counterfactual probes show
that the PPO actor already extracts dora, suji, kabe, riichi timing, and match
position:

* ``role_aware_tiles`` lets the shared canonical tile representation retain
  which semantic tile was called versus consumed by a multi-tile action;
* ``deeper_action_head`` adds two initially-identity action-memory blocks as a
  capacity control.

All variants see identical human decisions, start with identical policy
behavior, and use the same optimizer/update order.
"""

from __future__ import annotations

import argparse
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
from zenith_ppo.checkpoint import resolve_latest, restore
from zenith_ppo.cli.train_bc import _run_epoch
from zenith_ppo.config import load
from zenith_ppo.encoding.actions import segment_layout, segmented_entropy, segmented_log_softmax
from zenith_ppo.encoding.packing import model_batch, pack
from zenith_ppo.model.actor_critic import (
    ActorCritic,
    ActorOutput,
    CanonicalTileEmbedding,
)
from zenith_ppo.model.tile import current_public_tile_count_planes
from zenith_ppo.seeds import derive_seed


VARIANTS = (
    "baseline", "role_aware_tiles", "deeper_action_head", "post_action_shape",
)


class RoleAwareCanonicalTileEmbedding(CanonicalTileEmbedding):
    """Preserve tile role while retaining one shared tile identity table."""

    def __init__(self, d_model: int):
        import torch

        super().__init__(d_model)
        self.role_gate = torch.nn.Parameter(torch.zeros(4, d_model))

    def action_tiles(self, factors):
        semantic = factors[..., 7:11].long()
        valid = semantic.gt(0) & semantic.lt(136)
        tile_types = semantic.div(4, rounding_mode="floor")
        embedded = self._embed(tile_types, valid)
        gates = 1.0 + self.role_gate.to(embedded.dtype)
        count = valid.sum(-1, keepdim=True).clamp_min(1)
        return (
            embedded * gates[None, None] * valid[..., None]
        ).sum(-2) / count.sqrt()


class RoleAwareActorCritic(ActorCritic):
    def __init__(self, config):
        super().__init__(config)
        self.canonical_tile_embedding = RoleAwareCanonicalTileEmbedding(
            int(config["d_model"])
        )


class PostActionShapeActorCritic(ActorCritic):
    """Give each candidate an explicit encoding of its resulting hand."""

    def __init__(self, config):
        import torch

        super().__init__(config)
        width = int(config["d_model"])
        self.post_action_projection = torch.nn.Linear(
            width, width, bias=False
        )
        torch.nn.init.zeros_(self.post_action_projection.weight)

    def _post_action_residual(self, token_factors, public_lengths, padded):
        planes = current_public_tile_count_planes(
            token_factors, public_lengths
        )
        counts = planes[:, 0].flatten(1)[:, :34]
        batch, actions = padded.shape[:2]
        kinds = padded[..., 0].long()
        primary = padded[..., 1].long().clamp(0, 33)
        semantic = padded[..., 7:11].long()
        valid_tiles = semantic.gt(0) & semantic.lt(136)
        tile_types = semantic.div(4, rounding_mode="floor").clamp(0, 33)

        changes = counts.new_zeros(batch, actions, 34)
        removes_tiles = (
            kinds.ge(1) & kinds.le(7)
        )[..., None] & valid_tiles
        changes.scatter_add_(
            2, tile_types,
            -removes_tiles.to(changes.dtype),
        )
        # Chi, pon, and open-kan candidates include the externally called tile
        # among their sorted semantic tiles. It never leaves the concealed hand.
        called = kinds.ge(3) & kinds.le(5)
        changes.scatter_add_(
            2, primary[..., None], called[..., None].to(changes.dtype)
        )
        post = (counts[:, None] + changes).clamp_min(0)

        post_planes = counts.new_zeros(batch * actions, 1, 4, 9)
        post_planes.flatten(2)[:, 0, :34] = post.reshape(-1, 34)
        shape = self.action_memory.concealed_shape(post_planes)
        identity = self.canonical_tile_embedding.all_tiles()[None]
        tile_states = shape + identity.to(shape.dtype)
        weights = post.reshape(-1, 34)[..., None]
        summary = (tile_states * weights).sum(1) \
            / weights.sum(1).clamp_min(1).sqrt()
        return self.post_action_projection(summary).reshape(batch, actions, -1)

    def forward_actor(
        self, token_factors, action_factors, actor_query_indices, action_offsets,
        token_numeric=None, *, lengths=None, action_lengths=None,
        backend="sdpa", rank_boundary_features=None, decision_seats=None,
        strategic_features=None, compute_entropy=True, **_,
    ) -> ActorOutput:
        import torch

        if token_factors.shape[1] > self.context_tokens:
            raise ValueError(
                f"context overflow: {token_factors.shape[1]} > {self.context_tokens}"
            )
        if lengths is None:
            lengths = torch.full(
                (token_factors.shape[0],), token_factors.shape[1],
                dtype=torch.long, device=token_factors.device,
            )
        token_states = self.token_embedding(token_factors, token_numeric)
        token_states = self.token_tile_merge_norm(
            token_states
            + self.canonical_tile_embedding.tokens(token_factors).to(
                token_states.dtype
            )
        )
        history = self.backbone(token_states, lengths, backend)
        rows = torch.arange(history.shape[0], device=history.device)
        actor_state = history[rows, actor_query_indices]
        public_lengths = torch.minimum(lengths, actor_query_indices + 1)
        history_valid = self._valid(public_lengths, history.shape[1], history.device)
        if strategic_features is None:
            if rank_boundary_features is None or decision_seats is None:
                raise ValueError("actor requires boundary state and decision seat")
            strategic_features = self._strategic_features(
                rank_boundary_features, decision_seats
            )
        state, memory, memory_valid = self.action_memory.memory(
            actor_state, strategic_features, token_factors, public_lengths,
            history, history_valid,
            self.canonical_tile_embedding.all_tiles(),
        )
        padded, action_lengths = self._padded_actions(
            action_factors, action_lengths, action_offsets
        )
        action_valid = self._valid(action_lengths, padded.shape[1], padded.device)
        action_states = self.action_embedding(padded)
        canonical = (
            self.canonical_tile_embedding.action_tiles(padded)
            if self.share_all_action_tiles
            else self.canonical_tile_embedding.actions(padded)
        )
        action_states = self.action_tile_merge_norm(
            action_states + canonical.to(action_states.dtype)
        )
        action_states = action_states + self._post_action_residual(
            token_factors, public_lengths, padded
        ).to(action_states.dtype)
        action_states = self.action_memory(
            action_states, memory, memory_valid, action_valid
        )
        padded_logits = self.policy_head(action_states).squeeze(-1).float() \
            / self.policy_temperature
        logits = padded_logits[action_valid]
        layout = segment_layout(
            action_offsets, total=int(logits.shape[0]), device=logits.device
        )
        logp = segmented_log_softmax(logits, action_offsets, layout=layout)
        entropy = segmented_entropy(logp, action_offsets, layout=layout) \
            if compute_entropy else None
        return ActorOutput(logits, logp, entropy, state, action_states)


def _model(values, variant, device, seed):
    import torch

    torch.manual_seed(seed)
    if torch.device(device).type == "cuda":
        torch.cuda.manual_seed_all(seed)
    config = dict(values["model"])
    config["context_tokens"] = values["encoding"]["context_tokens"]
    if variant == "deeper_action_head":
        config["action_memory_layers"] = int(config["action_memory_layers"]) + 2
    cls = {
        "role_aware_tiles": RoleAwareActorCritic,
        "post_action_shape": PostActionShapeActorCritic,
    }.get(variant, ActorCritic)
    return cls(config).to(device)


def _load_initial(model, state, variant):
    import torch

    result = model.load_state_dict(state, strict=False)
    if result.unexpected_keys:
        raise RuntimeError(f"unexpected checkpoint keys: {result.unexpected_keys}")
    if variant == "baseline" and result.missing_keys:
        raise RuntimeError(f"baseline checkpoint is incomplete: {result.missing_keys}")
    if variant == "role_aware_tiles":
        expected = {"canonical_tile_embedding.role_gate"}
        if set(result.missing_keys) != expected:
            raise RuntimeError(f"unexpected role-aware missing keys: {result.missing_keys}")
    if variant == "post_action_shape":
        expected = {"post_action_projection.weight"}
        if set(result.missing_keys) != expected:
            raise RuntimeError(f"unexpected post-action missing keys: {result.missing_keys}")
    if variant == "deeper_action_head":
        original_layers = len(model.action_memory.blocks) - 2
        expected_prefixes = tuple(
            f"action_memory.blocks.{index}."
            for index in range(original_layers, original_layers + 2)
        )
        if not result.missing_keys or any(
            not key.startswith(expected_prefixes) for key in result.missing_keys
        ):
            raise RuntimeError(f"unexpected deeper-head missing keys: {result.missing_keys}")
        # Every added residual branch starts at zero, making both blocks exact
        # identities before adaptation while still allowing them to learn.
        for block in model.action_memory.blocks[original_layers:]:
            torch.nn.init.zeros_(block.memory_attention.out_proj.weight)
            torch.nn.init.zeros_(block.candidate_attention.out_proj.weight)
            torch.nn.init.zeros_(block.ffn.down.weight)


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


def _row_nll(model, examples, *, config, device, backend, use_bf16):
    import torch

    result = [None] * len(examples)
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
                actor = model.forward_actor(**inputs, compute_entropy=False)
            values = -actor.log_probabilities.index_select(0, selected)
            for index, row, value in zip(
                indices, rows, values.float().cpu(), strict=True
            ):
                result[index] = (row.family, float(value))
    if any(value is None for value in result):
        raise AssertionError("packed validation omitted an example")
    return tuple(result)


def _paired_validation(
    models, corpus, manifest, *, bc, device, backend, use_bf16, replay_threads,
):
    names = tuple(models)
    baseline = names[0]
    games = []
    totals = {name: {} for name in names}
    timings = {name: 0.0 for name in names}
    total_rows = total_actions = 0
    for archive, member, limit in manifest:
        values = corpus.examples_many(
            ((((Path(archive), member), int(limit))),),
            num_threads=replay_threads,
        )[0]
        if isinstance(values, Exception):
            continue
        examples = tuple(values)
        total_rows += len(examples)
        total_actions += sum(len(row.encoded.action_factors) for row in examples)
        per_name = {}
        for name, model in models.items():
            if str(device).startswith("cuda"):
                import torch
                torch.cuda.synchronize()
            started = perf_counter()
            rows = _row_nll(
                model, examples, config=bc, device=device, backend=backend,
                use_bf16=use_bf16,
            )
            if str(device).startswith("cuda"):
                import torch
                torch.cuda.synchronize()
            timings[name] += perf_counter() - started
            per_name[name] = rows
            for family, nll in rows:
                target = totals[name].setdefault(family, [0.0, 0])
                target[0] += nll
                target[1] += 1
        base = np.asarray([value for _, value in per_name[baseline]])
        games.append({
            "rows": len(examples),
            "differences": {
                name: float((
                    np.asarray([value for _, value in per_name[name]]) - base
                ).sum())
                for name in names[1:]
            },
        })

    def interval(name):
        rows = np.asarray([game["rows"] for game in games], dtype=np.float64)
        sums = np.asarray(
            [game["differences"][name] for game in games], dtype=np.float64
        )
        estimate = float(sums.sum() / rows.sum())
        influence = sums - estimate * rows
        count = len(games)
        standard_error = math.sqrt(
            count / max(count - 1, 1) * float(np.square(influence).sum())
        ) / float(rows.sum())
        radius = NormalDist().inv_cdf(0.975) * standard_error
        return {
            "mean_nll_difference_vs_baseline": estimate,
            "cluster_standard_error": standard_error,
            "ci95": [estimate - radius, estimate + radius],
            "games": count,
            "rows": int(rows.sum()),
        }

    return {
        "paired": {name: interval(name) for name in names[1:]},
        "family_nll": {
            name: {
                family: total / count
                for family, (total, count) in families.items()
            }
            for name, families in totals.items()
        },
        "corpus": {
            "games": len(games), "rows": total_rows,
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


def run(args):
    import torch

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
    if not variants or variants[0] != "baseline":
        raise ValueError("the first variant must be baseline")

    checkpoint = Path(args.initial_checkpoint)
    if (checkpoint / "latest").is_file():
        checkpoint = resolve_latest(checkpoint)
    initial_state = restore(checkpoint)["model"]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    train_corpus = ArchiveCorpus(bc["train_archives"])
    validation_corpus = ArchiveCorpus(bc["validation_archives"])
    train_candidates = list(train_corpus.members)
    validation_candidates = list(validation_corpus.members)
    random.Random(derive_seed(seed, "behavior_cloning_manifest")).shuffle(
        train_candidates
    )
    random.Random(derive_seed(seed, "behavior_cloning_validation")).shuffle(
        validation_candidates
    )
    train_manifest = validation_manifest = None
    models, results = {}, {}
    for variant in variants:
        model = _model(values, variant, device, seed)
        _load_initial(model, initial_state, variant)
        optimizer = _optimizer(model, bc, fused=use_bf16)
        started = perf_counter()
        initial_validation, selected_validation = _run_epoch(
            model, None, validation_corpus, manifest=validation_manifest,
            candidates=(
                validation_candidates
                if validation_manifest is None else validation_manifest
            ),
            maximum=int(bc["validation_decisions"]), config=bc,
            device=device, backend=profile.attention, use_bf16=use_bf16,
            train=False, data_workers=int(args.data_workers),
            replay_batch=int(args.replay_batch),
            replay_threads=int(args.replay_threads),
        )
        if validation_manifest is None:
            validation_manifest = selected_validation
        train_metrics, selected_train = _run_epoch(
            model, optimizer, train_corpus, manifest=train_manifest,
            candidates=train_candidates if train_manifest is None else train_manifest,
            maximum=int(bc["train_decisions"]), config=bc,
            device=device, backend=profile.attention, use_bf16=use_bf16,
            train=True, data_workers=int(args.data_workers),
            replay_batch=int(args.replay_batch),
            replay_threads=int(args.replay_threads),
        )
        if train_manifest is None:
            train_manifest = selected_train
        validation, _ = _run_epoch(
            model, None, validation_corpus, manifest=validation_manifest,
            candidates=validation_manifest,
            maximum=int(bc["validation_decisions"]), config=bc,
            device=device, backend=profile.attention, use_bf16=use_bf16,
            train=False, data_workers=int(args.data_workers),
            replay_batch=int(args.replay_batch),
            replay_threads=int(args.replay_threads),
        )
        diagnostics = {}
        if variant == "role_aware_tiles":
            diagnostics["role_gate_rms"] = float(
                model.canonical_tile_embedding.role_gate.detach().float()
                .square().mean().sqrt()
            )
        if variant == "post_action_shape":
            diagnostics["projection_rms"] = float(
                model.post_action_projection.weight.detach().float()
                .square().mean().sqrt()
            )
        results[variant] = {
            "initial_validation": initial_validation,
            "train": train_metrics,
            "validation": validation,
            "elapsed_seconds": perf_counter() - started,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "diagnostics": diagnostics,
        }
        models[variant] = model
        torch.save(model.state_dict(), output / f"{variant}.model.pt")
        print(json.dumps({variant: results[variant]}, sort_keys=True), flush=True)

    paired = _paired_validation(
        models, validation_corpus, validation_manifest, bc=bc, device=device,
        backend=profile.attention, use_bf16=use_bf16,
        replay_threads=int(args.replay_threads),
    )
    report = {
        "schema": "zenith-ppo-architecture-ablation-v1",
        "initial_checkpoint": str(checkpoint.resolve()),
        "config": str(Path(args.config).resolve()),
        "device": device,
        "train_decisions": int(bc["train_decisions"]),
        "validation_decisions": int(bc["validation_decisions"]),
        "variants": results,
        "paired_validation": paired,
        "train_manifest": train_manifest,
        "validation_manifest": validation_manifest,
    }
    temporary = output / ".report.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output / "report.json")
    print(json.dumps({"report": str(output / "report.json")}, sort_keys=True))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="training/configs/default.toml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--initial-checkpoint", required=True)
    parser.add_argument("--train-decisions", type=int, default=250_000)
    parser.add_argument("--validation-decisions", type=int, default=200_000)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=VARIANTS)
    parser.add_argument("--data-workers", type=int, default=8)
    parser.add_argument("--replay-batch", type=int, default=1)
    parser.add_argument("--replay-threads", type=int, default=1)
    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
