"""Temporary legal-observation action teachers and their auxiliary losses."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp


DEFAULT_TEACHER_CONFIG = {
    "discard_coefficient": 0.10,
    "reaction_coefficient": 0.03,
    "riichi_coefficient": 0.02,
    "reaction_entropy_coefficient": 0.005,
    "anneal_end": 0.50,
    "discard_temperature": 1.0,
    "reaction_pass_target": 0.90,
    "reaction_call_target": 0.75,
    "reaction_call_pass_target": 0.25,
    "riichi_target": 0.80,
    "dama_target": 0.20,
    "supported_yaku": ("yakuhai", "open_tanyao"),
}


@dataclass(frozen=True, slots=True)
class DiscardTarget:
    group_sets: tuple[tuple[int, ...], ...]
    probabilities: tuple[float, ...]
    shanten: tuple[int, ...]
    ukeire: tuple[int, ...]
    costs: tuple[float, ...]
    group_to_option: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ReactionTarget:
    probabilities: tuple[float, ...]
    pass_group: int | None
    accepted_group: int | None
    call_improves: tuple[bool, ...]


@dataclass(frozen=True, slots=True)
class RiichiTarget:
    pairs: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]
    probabilities: tuple[float, float]


@dataclass(frozen=True, slots=True)
class TeacherTargets:
    discard: DiscardTarget | None = None
    reaction: ReactionTarget | None = None
    riichi: RiichiTarget | None = None


@dataclass(frozen=True, slots=True)
class AuxiliaryLosses:
    total: object
    discard_loss: object
    reaction_loss: object
    riichi_loss: object
    reaction_entropy_loss: object
    metrics: dict[str, float]


@dataclass(frozen=True, slots=True)
class PackedTeacherTargets:
    """Device-ready sparse teacher layout reused across PPO epochs."""

    tensors: dict[str, object]
    counts: dict[str, int]


def coefficients(config, progress: float = 0.0, *, guidance_scale=None) -> dict[str, float]:
    """Scale the full-strength objectives by a frozen rollout snapshot."""
    values = {**DEFAULT_TEACHER_CONFIG, **dict(config or {})}
    end = float(values.get("anneal_end", 1.0))
    if guidance_scale is not None:
        scale = float(guidance_scale)
    elif end <= 0:
        scale = 0.0
    else:
        scale = max(0.0, 1.0 - float(progress) / end)
    return {
        "discard": float(values["discard_coefficient"]) * scale,
        "reaction": float(values["reaction_coefficient"]) * scale,
        "riichi": float(values["riichi_coefficient"]) * scale,
        "reaction_entropy": float(values["reaction_entropy_coefficient"]) * scale,
    }


def build_targets(state, decision, encoded_actions, config=None) -> TeacherTargets:
    """Build targets from the acting player's ordinary state projection only."""
    return build_batch_targets(((state, decision, encoded_actions),), config)[0]


class _AnalysisBatch:
    """Deduplicated hand queries resolved by one native extension call."""

    def __init__(self):
        self.rows = []
        self.melds = []
        self.mapping = {}

    def add(self, counts, open_melds: int) -> int:
        key = (bytes(counts), int(open_melds))
        index = self.mapping.get(key)
        if index is None:
            index = len(self.rows)
            self.mapping[key] = index
            self.rows.append(counts)
            self.melds.append(int(open_melds))
        return index

    def run(self):
        return _analyze(self.rows, self.melds) if self.rows else None


def build_batch_targets(items, config=None) -> tuple[TeacherTargets, ...]:
    """Build a rollout frame's targets with one batched native analysis.

    The previous implementation called ``analyze_hands`` once for every
    decision, and again for each legal call.  A rollout frame contains many
    independent decisions, so queue all counterfactual hands and let the Rust
    extension process them in one call instead.
    """
    values = {**DEFAULT_TEACHER_CONFIG, **dict(config or {})}
    analysis = _AnalysisBatch()
    plans = []
    for state, decision, encoded_actions in items:
        actions = tuple(
            decision.actions[index] for index in encoded_actions.representatives
        )
        plans.append((
            _prepare_discard(state, decision, actions, analysis),
            _prepare_reaction(state, decision, actions, analysis),
            _riichi_target(actions, values),
        ))
    results = analysis.run()
    return tuple(
        TeacherTargets(
            discard=_finish_discard(discard, results, values),
            reaction=_finish_reaction(reaction, results, values),
            riichi=riichi,
        )
        for discard, reaction, riichi in plans
    )


def _semantic_tile(tile: int) -> tuple[int, bool]:
    tile_type, copy = divmod(int(tile), 4)
    return tile_type, tile_type in (4, 13, 22) and copy == 0


def _counts(decision):
    import numpy as np
    return np.frombuffer(bytes(decision.concealed_counts), dtype=np.uint8).copy()


def _open_meld_count(state, seat: int) -> int:
    return sum(int(meld.seat) == int(seat) for meld in state.melds)


def _visible_remaining(state, decision):
    """Copies unseen by the actor; never consults ``state.hidden``."""
    import numpy as np

    public_ids = {int(tile) for tile in state.dora_indicators}
    public_ids.update(int(row.tile) for row in state.rivers)
    public_ids.update(int(tile) for meld in state.melds for tile in meld.tiles)
    public = np.zeros(34, dtype=np.int16)
    for tile in public_ids:
        if 0 <= tile < 136:
            public[tile // 4] += 1
    remaining = 4 - _counts(decision).astype(np.int16) - public
    return np.maximum(remaining, 0).astype(np.uint8)


def _analyze(rows, melds):
    import numpy as np
    import riichi
    return riichi.analyze_hands(
        np.ascontiguousarray(rows, dtype=np.uint8),
        np.ascontiguousarray(melds, dtype=np.uint8),
    )


def _ukeire(mask: int, remaining) -> int:
    return sum(int(remaining[tile]) for tile in range(34) if int(mask) & (1 << tile))


def _discard_target(state, decision, actions, config) -> DiscardTarget | None:
    analysis = _AnalysisBatch()
    plan = _prepare_discard(state, decision, actions, analysis)
    return _finish_discard(plan, analysis.run(), config)


def _prepare_discard(state, decision, actions, analysis):
    options = {}
    for group, action in enumerate(actions):
        if int(action.kind) not in (1, 2) or not action.tiles:
            continue
        options.setdefault(_semantic_tile(action.tiles[0]), []).append(group)
    if not options:
        return None
    ordered = sorted(options)
    counts = _counts(decision)
    indices = []
    opened = _open_meld_count(state, decision.seat)
    for tile_type, _ in ordered:
        after = counts.copy()
        if not after[tile_type]:
            raise ValueError("discard teacher received an absent legal tile")
        after[tile_type] -= 1
        indices.append(analysis.add(after, opened))
    return options, ordered, tuple(indices), _visible_remaining(state, decision), len(actions)


def _finish_discard(plan, analysis, config) -> DiscardTarget | None:
    if plan is None:
        return None
    options, ordered, indices, remaining, action_count = plan
    shanten = tuple(int(analysis.shanten[index, 0]) for index in indices)
    ukeire = tuple(
        _ukeire(int(analysis.improving_type_mask[index]), remaining) for index in indices
    )
    best_shanten, best_ukeire = min(shanten), max(ukeire)
    costs = tuple(
        4.0 * (value - best_shanten) + (best_ukeire - waits) / 137.0
        for value, waits in zip(shanten, ukeire, strict=True)
    )
    temperature = float(config["discard_temperature"])
    weights = tuple(exp(-cost / temperature) for cost in costs)
    total = sum(weights)
    probabilities = tuple(weight / total for weight in weights)
    group_sets = tuple(tuple(options[key]) for key in ordered)
    group_to_option = [-1] * action_count
    for option, groups in enumerate(group_sets):
        for group in groups:
            group_to_option[group] = option
    return DiscardTarget(
        group_sets, probabilities, shanten, ukeire, costs, tuple(group_to_option)
    )


def _called_tile(state, action) -> int | None:
    source = action.source_seat
    if source is None:
        return None
    candidates = [
        row for row in state.rivers
        if int(row.seat) == int(source) and not bool(row.called)
    ]
    if not candidates:
        return None
    return int(max(candidates, key=lambda row: int(row.sequence)).tile)


def _kuikae_types(action, called: int) -> set[int]:
    called_type = called // 4
    forbidden = {called_type}
    if int(action.kind) == 3:
        types = sorted(int(tile) // 4 for tile in action.tiles)
        if called_type == types[0] and called_type > 0:
            forbidden.add(called_type - 1)
        elif called_type == types[-1] and called_type % 9 < 8:
            forbidden.add(called_type + 1)
    return forbidden


def _simple(tile_type: int) -> bool:
    return tile_type < 27 and tile_type % 9 not in (0, 8)


def _yakuhai_types(state, seat: int) -> set[int]:
    seat_wind = 27 + ((int(seat) + 4 - int(state.dealer)) % 4)
    return {31, 32, 33, 27 + int(state.round_wind), seat_wind}


def _guaranteed_yaku(state, decision, action, after_discard, supported) -> bool:
    supported = set(supported)
    melds = [meld for meld in state.melds if int(meld.seat) == int(decision.seat)]
    called_types = [int(tile) // 4 for tile in action.tiles]
    if "yakuhai" in supported:
        values = _yakuhai_types(state, decision.seat)
        for meld in melds:
            types = [int(tile) // 4 for tile in meld.tiles]
            if len(types) >= 3 and len(set(types)) == 1 and types[0] in values:
                return True
        if int(action.kind) in (4, 5) and called_types and called_types[0] in values:
            return True
    if "open_tanyao" in supported:
        if any(count and not _simple(tile) for tile, count in enumerate(after_discard)):
            return False
        if any(not _simple(int(tile) // 4) for meld in melds for tile in meld.tiles):
            return False
        if all(_simple(tile) for tile in called_types):
            return True
    return False


def _call_followup(state, decision, action):
    """Return best forced-discard (shanten, -ukeire, tile) and yaku status."""
    analysis = _AnalysisBatch()
    plan = _prepare_call_followup(state, decision, action, analysis)
    return _finish_call_followup(plan, analysis.run())


def _prepare_call_followup(state, decision, action, analysis):
    if int(action.kind) not in (3, 4):
        return None  # open kan depends on an unseen replacement draw
    called = _called_tile(state, action)
    if called is None:
        return None
    counts = _counts(decision)
    for tile in action.tiles:
        tile_type = int(tile) // 4
        if tile_type == called // 4 and int(tile) == called:
            continue
        if counts[tile_type] == 0:
            return None
        counts[tile_type] -= 1
    forbidden = _kuikae_types(action, called)
    candidates = [
        tile
        for tile, count in enumerate(counts)
        if count and tile not in forbidden
    ]
    if not candidates:
        return None
    rows = []
    for tile in candidates:
        after = counts.copy()
        after[tile] -= 1
        rows.append(after)
    open_melds = _open_meld_count(state, decision.seat)
    baseline = analysis.add(_counts(decision), open_melds)
    indices = tuple(analysis.add(row, open_melds + 1) for row in rows)
    return baseline, indices, tuple(candidates), tuple(rows), _visible_remaining(state, decision)


def _finish_call_followup(plan, analysis):
    if plan is None:
        return None
    baseline_index, indices, candidates, rows, remaining = plan
    baseline = int(analysis.shanten[baseline_index, 0])
    ranked = []
    for index, tile, row in zip(indices, candidates, rows, strict=True):
        shanten = int(analysis.shanten[index, 0])
        ukeire = _ukeire(int(analysis.improving_type_mask[index]), remaining)
        ranked.append(((shanten, -ukeire, tile), row))
    score, after = min(ranked, key=lambda item: item[0])
    return baseline, score, after


def call_is_supported(
    state,
    decision,
    action,
    supported=("yakuhai", "open_tanyao"),
) -> bool:
    result = _call_followup(state, decision, action)
    if result is None:
        return False
    baseline, score, after = result
    return score[0] < baseline and _guaranteed_yaku(
        state, decision, action, after, supported
    )


def _reaction_target(state, decision, actions, config) -> ReactionTarget | None:
    analysis = _AnalysisBatch()
    plan = _prepare_reaction(state, decision, actions, analysis)
    return _finish_reaction(plan, analysis.run(), config)


def _prepare_reaction(state, decision, actions, analysis):
    wins = [group for group, action in enumerate(actions) if int(action.kind) in (8, 9)]
    if wins:
        # A self-turn Tsumo offer has discards but no pass action.  Build the
        # unconditional win target before applying the call/pass shape checks.
        return state, decision, actions, None, tuple(wins), (None,) * len(actions)
    pass_groups = [group for group, action in enumerate(actions) if int(action.kind) == 0]
    if not pass_groups or any(int(action.kind) in (1, 2) for action in actions):
        return None
    pass_group = pass_groups[0]
    calls = tuple(
        _prepare_call_followup(state, decision, action, analysis)
        if int(action.kind) in (3, 4, 5) else None
        for action in actions
    )
    return state, decision, actions, pass_group, tuple(wins), calls


def _finish_reaction(plan, analysis, config) -> ReactionTarget | None:
    if plan is None:
        return None
    state, decision, actions, pass_group, wins, calls = plan
    probabilities = [0.0] * len(actions)
    improves = tuple(
        _call_plan_is_supported(
            state, decision, action, call, analysis, config["supported_yaku"]
        ) if call is not None else False
        for action, call in zip(actions, calls, strict=True)
    )
    accepted = None
    if wins:
        for group in wins:
            probabilities[group] = 1.0 / len(wins)
    else:
        accepted_groups = [group for group, value in enumerate(improves) if value]
        if accepted_groups:
            accepted = min(accepted_groups)
            probabilities[accepted] = float(config["reaction_call_target"])
            probabilities[pass_group] = float(config["reaction_call_pass_target"])
        else:
            other = [group for group in range(len(actions)) if group != pass_group]
            probabilities[pass_group] = float(config["reaction_pass_target"])
            if other:
                residual = 1.0 - probabilities[pass_group]
                for group in other:
                    probabilities[group] = residual / len(other)
            else:
                probabilities[pass_group] = 1.0
    return ReactionTarget(tuple(probabilities), pass_group, accepted, improves)


def _call_plan_is_supported(state, decision, action, plan, analysis, supported) -> bool:
    result = _finish_call_followup(plan, analysis)
    if result is None:
        return False
    baseline, score, after = result
    return score[0] < baseline and _guaranteed_yaku(
        state, decision, action, after, supported
    )


def _riichi_target(actions, config) -> RiichiTarget | None:
    ordinary, riichi = {}, {}
    for group, action in enumerate(actions):
        if int(action.kind) not in (1, 2) or not action.tiles:
            continue
        target = ordinary if int(action.kind) == 1 else riichi
        target.setdefault(_semantic_tile(action.tiles[0]), []).append(group)
    pairs = tuple(
        (tuple(ordinary[key]), tuple(riichi[key]))
        for key in sorted(set(ordinary) & set(riichi))
    )
    if not pairs:
        return None
    return RiichiTarget(
        pairs, (float(config["dama_target"]), float(config["riichi_target"]))
    )


def pack_targets(targets, action_counts, *, device="cpu") -> PackedTeacherTargets:
    """Convert immutable rollout targets to a sparse vectorized tensor layout."""
    import torch

    starts, end = [], 0
    for count in action_counts:
        starts.append(end)
        end += int(count)
    if len(starts) != len(targets):
        raise ValueError("teacher targets and action counts must have equal length")

    discard_actions, discard_options = [], []
    discard_option_rows, discard_probabilities = [], []
    reaction_actions, reaction_rows, reaction_probabilities = [], [], []
    riichi_actions, riichi_sides = [], []
    riichi_side_probabilities, riichi_pair_rows = [], []
    discard_count = reaction_count = riichi_count = option_count = side_count = pair_count = 0
    for start, target in zip(starts, targets, strict=True):
        if target.discard is not None:
            for groups, probability in zip(
                target.discard.group_sets, target.discard.probabilities, strict=True
            ):
                discard_actions.extend(start + int(group) for group in groups)
                discard_options.extend([option_count] * len(groups))
                discard_option_rows.append(discard_count)
                discard_probabilities.append(float(probability))
                option_count += 1
            discard_count += 1
        if target.reaction is not None:
            for group, probability in enumerate(target.reaction.probabilities):
                reaction_actions.append(start + group)
                reaction_rows.append(reaction_count)
                reaction_probabilities.append(float(probability))
            reaction_count += 1
        if target.riichi is not None:
            for ordinary, riichi in target.riichi.pairs:
                for groups, probability in zip(
                    (ordinary, riichi), target.riichi.probabilities, strict=True
                ):
                    riichi_actions.extend(start + int(group) for group in groups)
                    riichi_sides.extend([side_count] * len(groups))
                    riichi_side_probabilities.append(float(probability))
                    side_count += 1
                riichi_pair_rows.append(riichi_count)
                pair_count += 1
            riichi_count += 1

    def tensor(values, dtype):
        return torch.tensor(values, dtype=dtype, device=device)

    return PackedTeacherTargets(
        tensors={
            "discard_actions": tensor(discard_actions, torch.long),
            "discard_options": tensor(discard_options, torch.long),
            "discard_option_rows": tensor(discard_option_rows, torch.long),
            "discard_probabilities": tensor(discard_probabilities, torch.float32),
            "reaction_actions": tensor(reaction_actions, torch.long),
            "reaction_rows": tensor(reaction_rows, torch.long),
            "reaction_probabilities": tensor(reaction_probabilities, torch.float32),
            "riichi_actions": tensor(riichi_actions, torch.long),
            "riichi_sides": tensor(riichi_sides, torch.long),
            "riichi_side_probabilities": tensor(
                riichi_side_probabilities, torch.float32
            ),
            "riichi_pair_rows": tensor(riichi_pair_rows, torch.long),
        },
        counts={
            "discard_rows": discard_count,
            "discard_options": option_count,
            "reaction_rows": reaction_count,
            "riichi_rows": riichi_count,
            "riichi_sides": side_count,
            "riichi_pairs": pair_count,
        },
    )


def _segment_logsumexp(values, segments, count: int):
    import torch

    maxima = torch.full(
        (count,), -torch.inf, dtype=values.dtype, device=values.device
    )
    maxima.scatter_reduce_(0, segments, values, reduce="amax", include_self=True)
    totals = torch.zeros_like(maxima)
    totals.scatter_add_(0, segments, (values - maxima[segments]).exp())
    return maxima + totals.log()


def auxiliary_losses(logits, log_probabilities, offsets, targets, current_coefficients,
                     *, eligible=None) -> AuxiliaryLosses:
    """Compute vectorized row-averaged teacher losses with exact zero-row behavior."""
    import torch
    from torch.nn import functional as F

    if not isinstance(targets, PackedTeacherTargets):
        offset_values = offsets.detach().cpu().tolist()
        selected_rows = (
            range(len(targets))
            if eligible is None
            else eligible.nonzero(as_tuple=False).flatten().cpu().tolist()
        )
        selected_targets = tuple(targets[row] for row in selected_rows)
        action_counts = [offset_values[row + 1] - offset_values[row] for row in selected_rows]
        # Preserve the original global action indices when selecting a subset.
        if eligible is None:
            targets = pack_targets(selected_targets, action_counts, device=logits.device)
        else:
            rebased = pack_targets(selected_targets, action_counts, device=logits.device)
            global_actions = []
            for row in selected_rows:
                global_actions.extend(range(offset_values[row], offset_values[row + 1]))
            mapping = torch.tensor(global_actions, dtype=torch.long, device=logits.device)
            tensors = dict(rebased.tensors)
            for name in ("discard_actions", "reaction_actions", "riichi_actions"):
                tensors[name] = mapping[tensors[name]]
            targets = PackedTeacherTargets(tensors, rebased.counts)

    zero = logits.float().sum() * 0.0
    tensors, counts = targets.tensors, targets.counts

    if counts["discard_rows"]:
        option_logp = _segment_logsumexp(
            log_probabilities[tensors["discard_actions"]].float(),
            tensors["discard_options"], counts["discard_options"],
        )
        discard_normalizer = _segment_logsumexp(
            option_logp, tensors["discard_option_rows"], counts["discard_rows"]
        )
        normalized_option_logp = option_logp - discard_normalizer[
            tensors["discard_option_rows"]
        ]
        discard_by_row = torch.zeros(
            counts["discard_rows"], dtype=torch.float32, device=logits.device
        )
        discard_by_row.scatter_add_(
            0, tensors["discard_option_rows"],
            -tensors["discard_probabilities"] * normalized_option_logp,
        )
        discard_loss = discard_by_row.mean()
        predicted = torch.full_like(discard_by_row, -torch.inf)
        predicted.scatter_reduce_(
            0, tensors["discard_option_rows"], option_logp,
            reduce="amax", include_self=True,
        )
        truth = torch.full_like(discard_by_row, -torch.inf)
        truth.scatter_reduce_(
            0, tensors["discard_option_rows"], tensors["discard_probabilities"],
            reduce="amax", include_self=True,
        )
        agreement_options = (
            option_logp == predicted[tensors["discard_option_rows"]]
        ) & (
            tensors["discard_probabilities"] == truth[tensors["discard_option_rows"]]
        )
        agreement = torch.zeros_like(discard_by_row)
        agreement.scatter_reduce_(
            0, tensors["discard_option_rows"], agreement_options.float(),
            reduce="amax", include_self=True,
        )
        discard_agreement = agreement.mean()
    else:
        discard_loss = discard_agreement = zero

    if counts["reaction_rows"]:
        reaction_logp = log_probabilities[tensors["reaction_actions"]].float()
        reaction_by_row = torch.zeros(
            counts["reaction_rows"], dtype=torch.float32, device=logits.device
        )
        reaction_by_row.scatter_add_(
            0, tensors["reaction_rows"],
            -tensors["reaction_probabilities"] * reaction_logp,
        )
        reaction_loss = reaction_by_row.mean()
        entropy_by_row = torch.zeros_like(reaction_by_row)
        entropy_by_row.scatter_add_(
            0, tensors["reaction_rows"], -(reaction_logp.exp() * reaction_logp),
        )
        reaction_entropy = entropy_by_row.mean()
    else:
        reaction_loss = reaction_entropy = zero

    if counts["riichi_rows"]:
        side_logits = _segment_logsumexp(
            logits[tensors["riichi_actions"]].float(),
            tensors["riichi_sides"], counts["riichi_sides"],
        ).reshape(-1, 2)
        pair_losses = -(
            tensors["riichi_side_probabilities"].reshape(-1, 2)
            * F.log_softmax(side_logits, dim=1)
        ).sum(dim=1)
        riichi_by_row = torch.zeros(
            counts["riichi_rows"], dtype=torch.float32, device=logits.device
        )
        riichi_by_row.scatter_add_(0, tensors["riichi_pair_rows"], pair_losses)
        pair_counts = torch.zeros_like(riichi_by_row)
        pair_counts.scatter_add_(
            0, tensors["riichi_pair_rows"], torch.ones_like(pair_losses)
        )
        riichi_loss = (riichi_by_row / pair_counts).mean()
    else:
        riichi_loss = zero
    reaction_entropy_loss = -float(current_coefficients["reaction_entropy"]) * reaction_entropy
    total = (
        float(current_coefficients["discard"]) * discard_loss
        + float(current_coefficients["reaction"]) * reaction_loss
        + float(current_coefficients["riichi"]) * riichi_loss
        + reaction_entropy_loss
    )
    metrics = {
        "discard_teacher_loss": discard_loss.detach(),
        "discard_teacher_rows": counts["discard_rows"],
        "discard_teacher_agreement": discard_agreement.detach(),
        "reaction_teacher_loss": reaction_loss.detach(),
        "reaction_teacher_rows": counts["reaction_rows"],
        "reaction_entropy": reaction_entropy.detach(),
        "riichi_teacher_loss": riichi_loss.detach(),
        "riichi_teacher_rows": counts["riichi_rows"],
    }
    return AuxiliaryLosses(
        total, discard_loss, reaction_loss, riichi_loss, reaction_entropy_loss, metrics
    )


def rollout_metrics(samples, kyoku_completions: int) -> dict[str, float]:
    """Learner-only teacher diagnostics with explicit applicable denominators."""
    learners = [sample for sample in samples if sample.ppo_eligible]
    discard_rows = [
        sample for sample in learners if sample.encoded.teachers.discard is not None
    ]
    shanten_regret = ukeire_regret = worse = agreement = 0.0
    costs = 0.0
    for sample in discard_rows:
        target = sample.encoded.teachers.discard
        option = target.group_to_option[sample.selected_group]
        best_shanten, best_ukeire = min(target.shanten), max(target.ukeire)
        shanten_regret += target.shanten[option] - best_shanten
        ukeire_regret += best_ukeire - target.ukeire[option]
        costs += target.costs[option]
        worse += target.shanten[option] > best_shanten
        best_probability = max(target.probabilities)
        agreement += target.probabilities[option] == best_probability
    reaction_rows = [
        sample
        for sample in learners
        if sample.encoded.teachers.reaction is not None
    ]
    passes = calls = improving_calls = 0
    for sample in reaction_rows:
        target = sample.encoded.teachers.reaction
        kind = _selected_action_kind(sample)
        passes += kind == 0
        calls += kind in (3, 4, 5)
        improving_calls += kind in (3, 4, 5) and target.call_improves[sample.selected_group]
    riichi_rows = [
        sample for sample in learners if sample.encoded.teachers.riichi is not None
    ]
    # A decision is one conversion opportunity even when several discard
    # tiles each have ordinary and riichi variants.
    opportunities = len(riichi_rows)
    declarations = sum(_selected_action_kind(sample) == 2 for sample in riichi_rows)
    metrics = {
        "teacher/discard_applicable_rows": float(len(discard_rows)),
        "teacher/discard_agreement": agreement / max(1, len(discard_rows)),
        "teacher/discard_mean_shanten_regret": shanten_regret / max(1, len(discard_rows)),
        "teacher/discard_mean_ukeire_regret": ukeire_regret / max(1, len(discard_rows)),
        "teacher/discard_mean_cost": costs / max(1, len(discard_rows)),
        "teacher/discard_worse_shanten_rate": worse / max(1, len(discard_rows)),
        "teacher/reaction_applicable_rows": float(len(reaction_rows)),
        "teacher/reaction_pass_probability": passes / max(1, len(reaction_rows)),
        "teacher/reaction_call_probability": calls / max(1, len(reaction_rows)),
        "teacher/reaction_improving_call_rate": improving_calls / max(1, calls),
        "teacher/riichi_applicable_rows": float(len(riichi_rows)),
        "teacher/riichi_conversion_rate": declarations / max(1, opportunities),
    }
    if int(kyoku_completions) > 0:
        denominator = float(kyoku_completions)
        metrics.update({
            "teacher/reaction_calls_per_kyoku": calls / denominator,
            "teacher/reaction_improving_calls_per_kyoku": (
                improving_calls / denominator
            ),
            "teacher/riichi_legal_opportunities_per_kyoku": (
                opportunities / denominator
            ),
            "teacher/riichi_declarations_per_kyoku": declarations / denominator,
        })
    return metrics


def _selected_action_kind(sample) -> int:
    representative = sample.encoded.action_representatives[sample.selected_group]
    return int(sample.encoded.native_actions[representative].kind)
