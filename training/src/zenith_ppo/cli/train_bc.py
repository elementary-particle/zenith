"""Streaming behavior cloning for the policy and boundary-rank critic."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path
from time import perf_counter

from ..capabilities import configure
from ..checkpoint import publish, resolve_latest, restore
from ..config import load


_WORKER_CORPUS = None
_WORKER_REPLAY_THREADS = 1


def _initialize_data_worker(archives, replay_threads):
    global _WORKER_CORPUS, _WORKER_REPLAY_THREADS
    from ..bc.data import ArchiveCorpus

    _WORKER_CORPUS = ArchiveCorpus(archives)
    _WORKER_REPLAY_THREADS = max(1, int(replay_threads))
    import torch

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _load_data_chunk(rows):
    requests = tuple(
        ((Path(archive), name), limit) for archive, name, limit in rows
    )
    results = _WORKER_CORPUS.examples_many(
        requests, num_threads=_WORKER_REPLAY_THREADS
    )
    return tuple(
        (False, f"{type(value).__name__}:{value}")
        if isinstance(value, Exception)
        else (True, value)
        for value in results
    )


def _architecture(_model):
    from ..model.factory import checkpoint_architecture

    return checkpoint_architecture(_model)


def _cpu_state(state):
    return {
        name: value.detach().cpu().clone() for name, value in state.items()
    }


def _selected_member(row):
    archive, name, limit = row
    return (Path(archive), name), int(limit)


class _Accumulator:
    """Keep sufficient statistics on-device until an epoch boundary."""

    def __init__(self, device):
        import torch

        self.device = torch.device(device)
        # policy nll, objective, objective weight, correct, rows
        self.policy = torch.zeros(5, dtype=torch.float64, device=self.device)
        # boundary nll, rows, correct, brier
        self.boundary = torch.zeros(4, dtype=torch.float64, device=self.device)
        self.families = {}

    def add(
        self, examples, row_nll, weights, correct, boundary_values,
    ):
        import torch

        nll = row_nll.detach().double()
        weights = weights.detach().double()
        correct = correct.detach().double()
        self.policy += torch.stack((
            nll.sum(),
            (nll * weights).sum(),
            weights.sum(),
            correct.sum(),
            nll.new_tensor(float(nll.numel())),
        ))
        self.boundary += boundary_values.detach().double()
        for family in {example.family for example in examples}:
            indices = torch.tensor(
                [
                    index for index, example in enumerate(examples)
                    if example.family == family
                ],
                dtype=torch.long,
                device=nll.device,
            )
            values = self.families.setdefault(
                family,
                torch.zeros(3, dtype=torch.float64, device=self.device),
            )
            values += torch.stack((
                nll.index_select(0, indices).sum(),
                correct.index_select(0, indices).sum(),
                nll.new_tensor(float(indices.numel())),
            ))

    def metrics(self):
        names = sorted(self.families)
        values = __import__("torch").cat((
            self.policy,
            self.boundary,
            *(self.families[name] for name in names),
        )).cpu().tolist()
        policy_nll, objective, objective_weight, correct, rows = values[:5]
        boundary_nll, boundary_rows, boundary_correct, boundary_brier = values[5:9]
        offset = 9
        families = {}
        for name in names:
            family_nll, family_correct, family_rows = values[offset:offset + 3]
            offset += 3
            families[name] = {
                "nll": family_nll / max(1, family_rows),
                "accuracy": family_correct / max(1, family_rows),
                "rows": int(family_rows),
            }
        return {
            "nll": policy_nll / max(1, rows),
            "objective_nll": objective / max(1, objective_weight),
            "accuracy": correct / max(1, rows),
            "rows": int(rows),
            "families": families,
            "rank_boundary": {
                "order_nll": boundary_nll / max(1, boundary_rows),
                "order_accuracy": boundary_correct / max(1, boundary_rows),
                "brier": boundary_brier / max(1, boundary_rows),
                "rows": int(boundary_rows),
            },
        }


def _regularized_policy_loss(
    log_probabilities, selected, action_offsets, *,
    label_smoothing=0.0, confidence_penalty_coefficient=0.0,
    weights=None,
):
    import torch

    selected_nll = -log_probabilities.index_select(0, selected)
    smoothing = float(label_smoothing)
    confidence = float(confidence_penalty_coefficient)
    objective = (1.0 - smoothing) * selected_nll
    for row in range(int(action_offsets.numel() - 1)):
        start = int(action_offsets[row])
        end = int(action_offsets[row + 1])
        segment = log_probabilities[start:end]
        if smoothing:
            objective[row] += -smoothing * segment.mean()
        if confidence:
            probabilities = segment.exp()
            objective[row] += confidence * (probabilities * segment).sum()
    if not torch.isfinite(objective).all():
        raise FloatingPointError("non-finite behavior-cloning objective")
    if weights is None:
        return objective.mean(), selected_nll
    weights = weights.to(device=objective.device, dtype=objective.dtype)
    if weights.shape != objective.shape or bool((weights <= 0).any()):
        raise ValueError("behavior-cloning weights must be positive per-row values")
    return (objective * weights).sum() / weights.sum(), selected_nll


def _run_examples(
    model, optimizer, examples, *, config, device, backend, use_bf16, train,
    accumulator,
):
    import torch
    from torch.nn import functional as F

    from ..encoding.actions import segmented_sample
    from ..encoding.packing import model_batch, pack

    examples = tuple(examples)
    packed = pack(
        [len(row.encoded.token_factors) for row in examples],
        int(config["token_budget"]),
        max_padding_fraction=0.10,
    )
    model.train(train)
    for indices in packed.batches:
        rows = tuple(examples[index] for index in indices)
        inputs = model_batch(
            [row.encoded for row in rows], device=device, backend=backend
        )
        targets = torch.tensor(
            [row.target for row in rows], dtype=torch.long, device=device
        )
        selected = inputs["action_offsets"][:-1] + targets
        configured_weights = config.get("family_weights", {})
        weights = torch.tensor(
            [float(configured_weights.get(row.family, 1.0)) for row in rows],
            dtype=torch.float32,
            device=device,
        )
        if train:
            optimizer.zero_grad(set_to_none=True)
        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            with torch.autocast(
                device_type=torch.device(device).type,
                dtype=torch.bfloat16,
                enabled=use_bf16,
            ):
                actor = model.forward_actor(**inputs)
                policy_loss, row_nll = _regularized_policy_loss(
                    actor.log_probabilities,
                    selected,
                    inputs["action_offsets"],
                    label_smoothing=(
                        float(config.get("label_smoothing", 0.0))
                        if train else 0.0
                    ),
                    confidence_penalty_coefficient=(
                        float(config.get("confidence_penalty_coefficient", 0.0))
                        if train else 0.0
                    ),
                    weights=weights if train else None,
                )
                critic = model.forward_critic(**inputs)
                boundary = torch.tensor(
                    [row.critic.rank_boundary_supervision for row in rows],
                    dtype=torch.bool,
                    device=device,
                )
                rank_loss = policy_loss * 0.0
                boundary_values = torch.zeros(
                    4, dtype=torch.float64, device=device
                )
                if bool(boundary.any()):
                    rank_targets = torch.tensor(
                        [row.critic.rank_order_target for row in rows],
                        dtype=torch.long,
                        device=device,
                    )[boundary]
                    rank_logits = critic.rank_order_logits[boundary].float()
                    rank_rows = int(rank_targets.numel())
                    rank_row_nll = F.cross_entropy(
                        rank_logits, rank_targets, reduction="none"
                    )
                    rank_loss = rank_row_nll.mean()
                    target_marginals = model.match_boundary_critic \
                        .order_to_marginals.index_select(0, rank_targets)
                    brier = (
                        critic.rank_marginals[boundary].float()
                        - target_marginals
                    ).square().sum((-1, -2))
                    boundary_values = torch.stack((
                        rank_row_nll.double().sum(),
                        rank_row_nll.new_tensor(rank_rows).double(),
                        rank_logits.argmax(-1).eq(rank_targets).double().sum(),
                        brier.double().sum(),
                    ))
                auxiliary_loss = policy_loss * 0.0
                auxiliary = getattr(model, "auxiliary_loss", None)
                if train and auxiliary is not None:
                    auxiliary_losses = auxiliary(actor, selected, rows)
                    for name, value in auxiliary_losses.items():
                        coefficient = float(
                            config.get(f"{name}_coefficient", 0.0)
                        )
                        if coefficient:
                            auxiliary_loss = auxiliary_loss + coefficient * value
                loss = policy_loss + float(
                    config["boundary_rank_coefficient"]
                ) * rank_loss + auxiliary_loss
            if train:
                loss.backward()
                maximum_norm = float(config["max_grad_norm"])
                if bool(config.get("separate_actor_critic_clipping", False)):
                    torch.nn.utils.clip_grad_norm_(
                        model.actor_parameters(), maximum_norm,
                        error_if_nonfinite=True,
                    )
                    torch.nn.utils.clip_grad_norm_(
                        model.critic_parameters(), maximum_norm,
                        error_if_nonfinite=True,
                    )
                else:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), maximum_norm,
                        error_if_nonfinite=True,
                    )
                optimizer.step()
        predicted = segmented_sample(
            actor.log_probabilities.detach(),
            inputs["action_offsets"],
            deterministic=True,
        )
        accumulator.add(
            rows,
            row_nll,
            weights,
            predicted.eq(selected),
            boundary_values,
        )
    return accumulator


def _chunks(rows, size):
    rows = tuple(rows)
    for start in range(0, len(rows), int(size)):
        yield rows[start:start + int(size)]


def _load_chunks(corpus, descriptors, *, workers, replay_batch, replay_threads):
    chunks = _chunks(descriptors, replay_batch)
    if int(workers) <= 0 or not hasattr(corpus, "examples_many"):
        for chunk in chunks:
            requests = tuple(
                ((Path(archive), name), limit)
                for archive, name, limit in chunk
            )
            values = corpus.examples_many(
                requests, num_threads=replay_threads
            )
            yield tuple(
                (False, f"{type(value).__name__}:{value}")
                if isinstance(value, Exception)
                else (True, value)
                for value in values
            )
        return
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=int(workers),
        mp_context=context,
        initializer=_initialize_data_worker,
        initargs=(tuple(map(str, corpus.archives)), int(replay_threads)),
    ) as executor:
        yield from executor.map(
            _load_data_chunk,
            chunks,
            chunksize=1,
            buffersize=max(2, int(workers) * 2),
        )


def _run_epoch(
    model, optimizer, corpus, *, manifest, candidates, maximum, config,
    device, backend, use_bf16, train, data_workers=0, replay_batch=8,
    replay_threads=1, example_transform=None,
):
    started = perf_counter()
    accumulator = _Accumulator(device)
    rejected = {}
    selected = [] if manifest is None else list(manifest)
    consumed = 0
    descriptors = (
        tuple((str(member[0]), member[1], None) for member in candidates)
        if manifest is None
        else tuple(
            (str(member[0]), member[1], limit)
            for member, limit in map(_selected_member, manifest)
        )
    )
    loaded = _load_chunks(
        corpus,
        descriptors,
        workers=data_workers,
        replay_batch=replay_batch,
        replay_threads=replay_threads,
    )
    descriptor_index = 0
    staged = []
    staged_tokens = 0
    staging_limit = int(config["token_budget"]) * int(
        os.environ.get("ZENITH_BC_STAGING_BATCHES", "4")
    )

    def flush():
        nonlocal staged, staged_tokens
        if staged:
            _run_examples(
                model,
                optimizer,
                staged,
                config=config,
                device=device,
                backend=backend,
                use_bf16=use_bf16,
                train=train,
                accumulator=accumulator,
            )
            staged = []
            staged_tokens = 0

    for chunk in loaded:
        for ok, value in chunk:
            archive, name, _ = descriptors[descriptor_index]
            descriptor_index += 1
            if maximum is not None and consumed >= maximum:
                break
            if not ok:
                rejected[value] = rejected.get(value, 0) + 1
                continue
            examples = tuple(value)
            if example_transform is not None:
                examples = tuple(example_transform(examples))
            if maximum is not None:
                examples = examples[:maximum - consumed]
            if not examples:
                rejected["ValueError:no_queryable_decisions"] = \
                    rejected.get("ValueError:no_queryable_decisions", 0) + 1
                continue
            staged.extend(examples)
            staged_tokens += sum(
                len(example.encoded.token_factors) for example in examples
            )
            if staged_tokens >= staging_limit:
                flush()
            if manifest is None:
                selected.append((archive, name, len(examples)))
            consumed += len(examples)
        if maximum is not None and consumed >= maximum:
            break
    flush()
    if not consumed:
        raise RuntimeError("behavior cloning found zero usable decisions")
    metrics = accumulator.metrics()
    metrics["combined_objective_nll"] = (
        metrics["objective_nll"]
        + float(config["boundary_rank_coefficient"])
        * metrics["rank_boundary"]["order_nll"]
    )
    metrics["throughput_rows_per_second"] = consumed / max(
        perf_counter() - started, 1e-9
    )
    metrics["rejected_games"] = rejected
    return metrics, tuple(selected)


def run(args):
    import random
    import torch

    from ..bc.data import ArchiveCorpus
    from ..model.factory import build_actor_critic
    from ..seeds import derive_seed

    config = load(args.config)
    values = config.values
    if "behavior_cloning" not in values:
        raise ValueError("configuration has no [behavior_cloning] group")
    bc = values["behavior_cloning"]
    for path in (*bc["train_archives"], *bc["validation_archives"]):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    profile = configure(values["run"]["profile"])
    device = "cuda" if profile.device == "cuda" else "cpu"
    use_bf16 = profile.precision == "bf16"
    seed = int(values["run"]["seed"])
    selection_seed = derive_seed(seed, "behavior_cloning_manifest")
    validation_seed = derive_seed(seed, "behavior_cloning_validation")
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    model_config = dict(values["model"])
    model_config["context_tokens"] = values["encoding"]["context_tokens"]
    model = build_actor_critic(model_config).to(device)
    initial_policy = None
    if args.initial_checkpoint:
        path = Path(args.initial_checkpoint)
        if (path / "latest").is_file():
            path = resolve_latest(path)
        restored = restore(path)
        model.load_state_dict(restored["model"])
        initial_policy = {
            "checkpoint_id": path.name,
            "checkpoint_path": str(path.resolve()),
        }
    actor_parameters = list(model.actor_parameters())
    # BC has exact labels for the public match-boundary rank prior.
    critic_parameters = list(model.match_boundary_critic.parameters())
    optimizer = torch.optim.AdamW(
        (
            {
                "params": actor_parameters,
                "lr": float(bc["learning_rate"]),
            },
            {
                "params": critic_parameters,
                "lr": float(bc["boundary_rank_learning_rate"]),
            },
        ),
        betas=(float(bc["adam_beta1"]), float(bc["adam_beta2"])),
        eps=float(bc["adam_epsilon"]),
        weight_decay=float(bc["weight_decay"]),
        fused=use_bf16,
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    train_corpus = ArchiveCorpus(bc["train_archives"])
    validation_corpus = ArchiveCorpus(bc["validation_archives"])
    default_workers = min(8, max(1, (os.cpu_count() or 1) // 2))
    data_workers = int(os.environ.get(
        "ZENITH_BC_DATA_WORKERS",
        str(default_workers) if device == "cuda" else "0",
    ))
    replay_batch = int(os.environ.get("ZENITH_BC_REPLAY_BATCH", "1"))
    replay_threads = int(os.environ.get("ZENITH_BC_REPLAY_THREADS", "1"))
    completed_epoch = 0
    train_manifest = validation_manifest = None
    best_score = float("inf")
    best_epoch = 0
    best_policy = None
    history = []
    seed_state = {
        "version": 1,
        "root": seed,
        "manifest": selection_seed,
        "validation": validation_seed,
    }
    if args.resume:
        path = Path(args.resume)
        if (path / "latest").is_file():
            path = resolve_latest(path)
        restored = restore(path)
        state = restored.get("trainer", {}).get("behavior_cloning") or {}
        if int(state.get("version", 0)) != 6:
            raise RuntimeError("behavior-cloning resume version mismatch")
        if state.get("config_digest") != config.digest:
            raise RuntimeError("behavior-cloning resume config mismatch")
        model.load_state_dict(restored["model"])
        optimizer.load_state_dict(restored["optimizer"]["policy"])
        completed_epoch = int(state["completed_epoch"])
        train_manifest = tuple(tuple(row) for row in state["train_manifest"])
        validation_manifest = tuple(
            tuple(row) for row in state["validation_manifest"]
        )
        best_score = float(state["best_score"])
        best_epoch = int(state["best_epoch"])
        best_policy = state["best_policy"]
        history = list(state.get("history", ()))
        initial_policy = state.get("initial_policy")
        if state.get("seed_state") != seed_state:
            raise RuntimeError("behavior-cloning resume seed mismatch")

    (output / "resolved-config.json").write_text(
        json.dumps(config.redacted(), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    started = perf_counter()
    for epoch in range(completed_epoch + 1, int(bc["epochs"]) + 1):
        if train_manifest is None:
            candidates = list(train_corpus.members)
            random.Random(selection_seed).shuffle(candidates)
        else:
            candidates = list(train_manifest)
            random.Random(
                derive_seed(seed, f"behavior_cloning_epoch_{epoch}")
            ).shuffle(candidates)
            train_manifest = tuple(candidates)
        train_limit = int(bc["train_decisions"])
        train_metrics, selected_train = _run_epoch(
            model,
            optimizer,
            train_corpus,
            manifest=train_manifest,
            candidates=candidates,
            maximum=None if train_limit == 0 else train_limit,
            config=bc,
            device=device,
            backend=profile.attention,
            use_bf16=use_bf16,
            train=True,
            data_workers=data_workers,
            replay_batch=replay_batch,
            replay_threads=replay_threads,
        )
        if train_manifest is None:
            train_manifest = selected_train
        validation_candidates = list(validation_corpus.members)
        if validation_manifest is None:
            random.Random(validation_seed).shuffle(validation_candidates)
        validation_metrics, selected_validation = _run_epoch(
            model,
            None,
            validation_corpus,
            manifest=validation_manifest,
            candidates=validation_candidates,
            maximum=int(bc["validation_decisions"]),
            config=bc,
            device=device,
            backend=profile.attention,
            use_bf16=use_bf16,
            train=False,
            data_workers=data_workers,
            replay_batch=replay_batch,
            replay_threads=replay_threads,
        )
        if validation_manifest is None:
            validation_manifest = selected_validation
        if validation_metrics["combined_objective_nll"] < best_score:
            best_score = float(validation_metrics["combined_objective_nll"])
            best_epoch = epoch
            best_policy = _cpu_state(model.state_dict())
        row = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation_metrics,
            "best_epoch": best_epoch,
            "best_validation_combined_objective_nll": best_score,
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        state = {
            "model": model.state_dict(),
            "optimizer": {
                "architecture": _architecture(model),
                "policy": optimizer.state_dict(),
            },
            "trainer": {"behavior_cloning": {
                "version": 6,
                "config_digest": config.digest,
                "completed_epoch": epoch,
                "train_manifest": train_manifest,
                "validation_manifest": validation_manifest,
                "best_score": best_score,
                "best_epoch": best_epoch,
                "best_policy": best_policy,
                "history": history,
                "seed_state": seed_state,
                "initial_policy": initial_policy,
            }},
            "state": {
                "architecture": _architecture(model),
                "phase": "behavior_cloning",
                "epoch": epoch,
                "selected_epoch": best_epoch,
            },
        }
        publish(
            output / "checkpoints",
            state,
            metadata={
                "config_digest": config.digest,
                "purpose": "behavior-cloning-epoch",
            },
        )

    if best_policy is None:
        raise RuntimeError("behavior cloning has no selected policy")
    model.load_state_dict(best_policy)
    final_state = {
        "model": model.state_dict(),
        "optimizer": {
            "architecture": _architecture(model),
            "policy": optimizer.state_dict(),
        },
        "trainer": {"behavior_cloning": {
            "version": 6,
            "config_digest": config.digest,
            "completed_epoch": int(bc["epochs"]),
            "train_manifest": train_manifest,
            "validation_manifest": validation_manifest,
            "best_score": best_score,
            "best_epoch": best_epoch,
            "best_policy": best_policy,
            "history": history,
            "seed_state": seed_state,
            "initial_policy": initial_policy,
        }},
        "state": {
            "architecture": _architecture(model),
            "phase": "behavior_cloning_complete",
            "epoch": int(bc["epochs"]),
            "selected_epoch": best_epoch,
        },
    }
    checkpoint_id = publish(
        output / "checkpoints",
        final_state,
        metadata={
            "config_digest": config.digest,
            "purpose": "behavior-cloned-policy-and-boundary-rank",
        },
    )
    summary = {
        "status": "completed",
        "config_digest": config.digest,
        "epochs": history,
        "selected_epoch": best_epoch,
        "best_validation_combined_objective_nll": best_score,
        "checkpoint_id": checkpoint_id,
        "checkpoint_path": str(output / "checkpoints" / checkpoint_id),
        "elapsed_seconds": perf_counter() - started,
        "train_manifest_games": len(train_manifest or ()),
        "validation_manifest_games": len(validation_manifest or ()),
        "data_loader": {
            "workers": data_workers,
            "replay_batch": replay_batch,
            "replay_threads": replay_threads,
        },
    }
    temporary = output / ".summary.json.tmp"
    temporary.write_text(
        json.dumps(summary, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output / "summary.json")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(prog="zenith-ppo-train-bc")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--initial-checkpoint")
    args = parser.parse_args(argv)
    if args.initial_checkpoint and args.resume:
        parser.error("--initial-checkpoint is mutually exclusive with --resume")
    run(args)


if __name__ == "__main__":
    main()
