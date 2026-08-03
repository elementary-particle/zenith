"""Batch-oriented native env rollout collection."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
import os
from pathlib import Path
import struct

from ..encoding.event_cache import EventPrefixCache
from ..encoding.critic import RANK_ORDER_INDEX
from ..encoding.packing import encode_native_batch, model_batch, pack
from ..profiling import StageProfiler
from ..inference import CONSERVATIVE_BOT_ID, ConservativeBot
from ..types import (
    ActionSegment, ObservationRecord, RolloutFrame,
    RolloutMatchOutcome, RolloutSample,
)


@dataclass(frozen=True, slots=True)
class CollectionResult:
    samples: tuple[RolloutSample, ...]
    decisions: int
    env_calls: int
    trajectory_digest: str
    kyoku_completions: int
    match_completions: int
    frames: tuple[RolloutFrame, ...] = ()
    match_outcomes: tuple[RolloutMatchOutcome, ...] = ()
    game_metrics: dict[str, float] = field(default_factory=dict)
    continuation: object | None = None
    model_queries: int = 0
    rust_resolved_decisions: int = 0


@dataclass(frozen=True, slots=True)
class _PendingDecision:
    row: object
    owner: tuple[str, bool]
    frame_index: int
    state: object


@dataclass(slots=True)
class _KyokuGameMetrics:
    learner_mask: int
    riichi_seats: set[int] = field(default_factory=set)
    open_seats: set[int] = field(default_factory=set)
    dealt_in_seats: set[int] = field(default_factory=set)
    discards: list[int] = field(default_factory=lambda: [0, 0, 0, 0])


class _GameMetricAccumulator:
    """Aggregate conventional player statistics from canonical core events."""

    def __init__(self, learner_mask):
        self._learner_mask = learner_mask
        self._kyoku: dict[tuple[int, int], _KyokuGameMetrics] = {}
        self.counts = {
            "kyoku": 0.0,
            "player_kyoku": 0.0,
            "wins": 0.0,
            "deal_ins": 0.0,
            "riichi_hands": 0.0,
            "calling_hands": 0.0,
            "tsumo_wins": 0.0,
            "dama_wins": 0.0,
            "winning_points": 0.0,
            "winning_point_events": 0.0,
            "deal_in_points": 0.0,
            "deal_in_point_events": 0.0,
            "winning_turns": 0.0,
            "winning_turn_events": 0.0,
            "exhaustive_ryukyoku": 0.0,
            "player_matches": 0.0,
            "bankrupt_matches": 0.0,
        }

    @staticmethod
    def _key(event):
        return int(event.environment_id), int(event.episode_generation)

    @staticmethod
    def _settlement(event):
        payload = bytes(event.payload or b"")
        return struct.unpack("<4i", payload[-16:]) if len(payload) >= 16 else None

    @staticmethod
    def _seat(event, name):
        value = getattr(event, name, None)
        return 255 if value is None else int(value)

    def observe(self, events):
        for event in events:
            kind = int(event.kind)
            key = self._key(event)
            if kind == 2:  # start_kyoku
                self._kyoku[key] = _KyokuGameMetrics(self._learner_mask(key))
                continue
            if kind == 16:  # end_game
                payload = bytes(event.payload or b"")
                if len(payload) == 20:
                    scores = struct.unpack("<4i", payload[:16])
                    mask = self._learner_mask(key)
                    self.counts["player_matches"] += mask.bit_count()
                    self.counts["bankrupt_matches"] += sum(
                        bool(mask & (1 << seat)) and scores[seat] < 0
                        for seat in range(4)
                    )
                continue
            kyoku = self._kyoku.get(key)
            if kyoku is None:
                continue
            actor = self._seat(event, "actor_seat")
            target = self._seat(event, "target_seat")
            if kind == 4 and 0 <= actor < 4:  # dahai
                kyoku.discards[actor] += 1
            elif kind in (5, 6, 7) and 0 <= actor < 4:  # chi/pon/daiminkan
                kyoku.open_seats.add(actor)
            elif kind == 11 and 0 <= actor < 4:  # reach declaration
                kyoku.riichi_seats.add(actor)
            elif kind == 13 and 0 <= actor < 4:  # hora
                if kyoku.learner_mask & (1 << actor):
                    self.counts["wins"] += 1
                    is_tsumo = actor == target
                    self.counts["tsumo_wins"] += int(is_tsumo)
                    self.counts["dama_wins"] += int(
                        actor not in kyoku.open_seats
                        and actor not in kyoku.riichi_seats
                    )
                    self.counts["winning_turns"] += (
                        kyoku.discards[actor] + 1
                        if is_tsumo else max(1, kyoku.discards[actor])
                    )
                    self.counts["winning_turn_events"] += 1
                    settlement = self._settlement(event)
                    if settlement is not None:
                        self.counts["winning_points"] += max(0, settlement[actor])
                        self.counts["winning_point_events"] += 1
                if (
                    actor != target
                    and 0 <= target < 4
                    and kyoku.learner_mask & (1 << target)
                    and target not in kyoku.dealt_in_seats
                ):
                    kyoku.dealt_in_seats.add(target)
                    self.counts["deal_ins"] += 1
                    settlement = self._settlement(event)
                    if settlement is not None:
                        self.counts["deal_in_points"] += max(0, -settlement[target])
                        self.counts["deal_in_point_events"] += 1
            elif kind == 14:  # ryukyoku
                args = getattr(event, "args", (0, 0, 0, 0))
                if int(args[0]) == 3:  # exhaustive draw
                    self.counts["exhaustive_ryukyoku"] += 1
            elif kind == 15:  # end_kyoku
                self.counts["kyoku"] += 1
                self.counts["player_kyoku"] += kyoku.learner_mask.bit_count()
                self.counts["riichi_hands"] += sum(
                    bool(kyoku.learner_mask & (1 << seat))
                    for seat in kyoku.riichi_seats
                )
                self.counts["calling_hands"] += sum(
                    bool(kyoku.learner_mask & (1 << seat))
                    for seat in kyoku.open_seats
                )
                del self._kyoku[key]

    def metrics(self):
        return _game_metric_values(self.counts)


class Collector:
    def __init__(self, adapter, model, action_generator, *, device="cpu",
                 backend="sdpa", inference_token_budget=65536, use_bf16=False,
                 max_padding_fraction=None,
                 lineups=None, lineup_provider=None, policy_models=None,
                 opponent_agents=None,
                 deterministic_policy_ids=None,
                 residency=None, model_loader=None, profiler=None, event_cache=None,
                 bot_policy=None, diagnostic_dir=None,
                 ):
        self.adapter = adapter
        self.model = model
        self.action_generator = action_generator
        self.device = device
        self.backend = backend
        self.inference_token_budget = int(inference_token_budget)
        self.max_padding_fraction = max_padding_fraction
        self.use_bf16 = bool(use_bf16 and device == "cuda")
        self.lineups = dict(lineups or {})
        self.lineup_provider = lineup_provider
        self.policy_models = dict(policy_models or {})
        self.opponent_agents = dict(opponent_agents or {})
        self.deterministic_policy_ids = frozenset(deterministic_policy_ids or ())
        self.residency = residency
        self.model_loader = model_loader
        self.profiler = profiler or StageProfiler()
        self.event_cache = event_cache or EventPrefixCache()
        self.bot_policy = bot_policy or ConservativeBot()
        self.diagnostic_dir = Path(
            diagnostic_dir or Path.cwd() / "native-env-diagnostics"
        )

    def collect(self, initial_batch, *, target_matches: int, curriculum, streams,
                current_policy_id="current",
                max_env_calls: int | None = None) -> CollectionResult:
        import torch

        if target_matches <= 0:
            raise ValueError("target_matches must be positive")
        batch = initial_batch
        launched = {
            (int(state.environment_id), int(state.episode_generation))
            for state in batch.transition.states if int(state.lifecycle) != 3
        }
        if not launched or len(launched) > int(target_matches):
            raise ValueError(
                "rollout must start with between one and target_matches fresh "
                f"matches, got {len(launched)}/{target_matches}"
            )
        launched_matches = len(launched)
        pending_matches = set(launched)
        samples: list[RolloutSample] = []
        frames: list[RolloutFrame] = []
        tails: dict[tuple[int, int, int], int] = {}
        kyoku_tails: dict[tuple[int, int, int], int] = {}
        kyoku_indices: dict[tuple[int, int], list[int]] = {}
        trajectory_indices: dict[tuple[int, int, int], list[int]] = {}
        sample_tails: dict[tuple[int, int, int], int] = {}
        sample_kyoku_tails: dict[tuple[int, int, int], int] = {}
        frame_queue = [initial_batch]
        pending_decisions: list[_PendingDecision] = []
        pending_decision_tokens = 0
        frame_counts = {key: 0 for key in launched}
        digest = sha256()
        env_calls = 0
        kyoku_completions = 0
        match_completions = 0
        match_outcomes = []
        model_queries = 0
        last_actions = ()
        game_metrics = _GameMetricAccumulator(self._learner_mask)
        self.model.eval()
        while pending_matches and (frame_queue or pending_decisions):
            batch = frame_queue.pop(0)
            game_metrics.observe(batch.transition.events)
            for state in batch.transition.states:
                key = (int(state.environment_id), int(state.episode_generation))
                frame_counts[key] = frame_counts.get(key, 0) + 1
                if max_env_calls is not None and frame_counts[key] > int(max_env_calls):
                    raise RuntimeError(
                        "rollout core-frame horizon exhausted before match completion: "
                        f"match={key}, frames={frame_counts[key]}"
                    )
            # A native transition is both an action frame and the receipt for
            # the preceding step. Consume its boundaries before asking the
            # model for another action. In particular, an all-terminal
            # transition legitimately contains no decisions.
            ended_kyoku, ended_matches = _apply_boundaries(
                batch, frames, tails, kyoku_tails=kyoku_tails,
                match_outcomes=match_outcomes,
                trajectory_indices=trajectory_indices,
                kyoku_indices=kyoku_indices,
            )
            _apply_boundaries(
                batch, samples, sample_tails,
                kyoku_tails=sample_kyoku_tails,
            )
            kyoku_completions += ended_kyoku
            match_completions += ended_matches
            terminal_matches = {
                (int(state.environment_id), int(state.episode_generation))
                for state in batch.transition.states
                if int(state.lifecycle) == 3
            }
            failed_matches = {
                (int(state.environment_id), int(state.episode_generation))
                for state in batch.transition.states
                if int(state.lifecycle) == 4
            }
            if failed_matches:
                raise RuntimeError(
                    f"native env failed for matches {sorted(failed_matches)}"
                )
            unexpected = terminal_matches - pending_matches
            if unexpected:
                raise RuntimeError(
                    "native env returned terminal matches outside the active launch set: "
                    f"{sorted(unexpected)}"
                )
            if terminal_matches and ended_matches != len(terminal_matches):
                raise RuntimeError(
                    "native terminal states are missing matching end_game events: "
                    f"terminal={sorted(terminal_matches)}, end_game_events={ended_matches}"
                )
            pending_matches.difference_update(terminal_matches)
            refill = None
            remaining_to_launch = int(target_matches) - launched_matches
            if terminal_matches and remaining_to_launch > 0:
                refill_ids = sorted({
                    int(environment_id)
                    for environment_id, _ in terminal_matches
                })[:remaining_to_launch]
                refill = self.adapter.reset(refill_ids)
                refill_keys = {
                    (int(state.environment_id), int(state.episode_generation))
                    for state in refill.transition.states
                    if int(state.lifecycle) not in (3, 4)
                }
                if len(refill_keys) != len(refill_ids):
                    raise RuntimeError(
                        "native env did not return one fresh match per refilled slot"
                    )
                pending_matches.update(refill_keys)
                launched_matches += len(refill_keys)
            with self.profiler.measure("rollout.encoding"):
                encoded = encode_native_batch(
                    batch,
                    self.adapter.histories,
                    event_cache=self.event_cache,
                )
            frame_encoded = encoded
            automatic_ids = tuple(
                int(state.environment_id)
                for state in batch.transition.states
                if int(state.lifecycle) not in (3, 4) and not state.action_spaces
            )
            frame_owners = [
                self._owner(row.binding, current_policy_id) for row in frame_encoded
            ]
            states_by_match = {
                (int(state.environment_id), int(state.episode_generation)): state
                for state in batch.transition.states
            }
            terminal_ranks = {}
            for event in batch.transition.events:
                payload = bytes(event.payload or b"")
                if int(event.kind) == 16 and len(payload) == 20:
                    terminal_ranks[(
                        int(event.environment_id),
                        int(event.episode_generation),
                    )] = tuple(int(rank) - 1 for rank in payload[16:20])
            for index, (row, owner) in enumerate(zip(
                frame_encoded, frame_owners, strict=True
            )):
                frame_index = len(frames)
                key = (
                    row.binding.environment_id,
                    row.binding.episode_generation,
                    row.binding.seat,
                )
                previous = tails.get(key)
                if previous is not None:
                    frames[previous].successor = frame_index
                frame = RolloutFrame(
                    binding=row.binding,
                    checkpoint_id=owner[0],
                    ppo_eligible=owner[1],
                    phase=int(next(
                        state.phase for state in batch.transition.states
                        if int(state.environment_id) == row.binding.environment_id
                    )),
                    genuine_action=bool(row.native_candidates),
                    old_boundary_rank_value=0.0,
                    old_boundary_rank_probabilities=(0.25, 0.25, 0.25, 0.25),
                    terminal=(terminal := (
                        row.binding.environment_id,
                        row.binding.episode_generation,
                    )) in terminal_ranks,
                    match_boundary=terminal in terminal_ranks,
                    terminal_placement=(
                        terminal_ranks[terminal][row.binding.seat]
                        if terminal in terminal_ranks else -1
                    ),
                    encoded=row,
                )
                frames.append(frame)
                tails[key] = frame_index
                kyoku_tails[key] = frame_index
                trajectory_indices.setdefault(key, []).append(frame_index)
                kyoku_indices.setdefault(key[:2], []).append(frame_index)
                if row.native_candidates:
                    pending_decisions.append(_PendingDecision(
                        row,
                        owner,
                        frame_index,
                        states_by_match[(
                            row.binding.environment_id,
                            row.binding.episode_generation,
                        )],
                    ))
                    pending_decision_tokens += len(row.token_factors)

            # Environments already waiting for policy actions remain paused
            # while disjoint automatic environments advance toward their next
            # decision.  Flush once the pending rows fill one inference pack,
            # or once no automatic work remains to enlarge the batch.
            next_batches = []
            if automatic_ids:
                next_batches.append(self.adapter.advance(automatic_ids))
                env_calls += 1
            if refill is not None:
                next_batches.append(refill)
            if next_batches:
                from ..env.adapter import EnvBatch
                frame_queue.append(EnvBatch.merge(*next_batches))
            if not pending_decisions:
                continue
            if (
                pending_decision_tokens < self.inference_token_budget
                and frame_queue
            ):
                continue

            encoded = tuple(item.row for item in pending_decisions)
            owners = [item.owner for item in pending_decisions]
            frame_by_binding = {
                item.row.binding: item.frame_index for item in pending_decisions
            }
            selected_groups = [0] * len(encoded)
            old_log_probabilities = [0.0] * len(encoded)
            grouped = {}
            for index, (checkpoint_id, _) in enumerate(owners):
                if (
                    checkpoint_id != CONSERVATIVE_BOT_ID
                    and checkpoint_id not in self.opponent_agents
                ):
                    grouped.setdefault(checkpoint_id, []).append(index)
                    model_queries += 1
            states = {
                (
                    item.row.binding.environment_id,
                    item.row.binding.episode_generation,
                ): item.state
                for item in pending_decisions
            }
            for index, (checkpoint_id, eligible) in enumerate(owners):
                if eligible and checkpoint_id in self.deterministic_policy_ids:
                    raise ValueError(
                        "deterministic neural policy rows cannot be PPO-eligible"
                    )
                if checkpoint_id != CONSERVATIVE_BOT_ID:
                    continue
                if eligible:
                    raise ValueError("deterministic bot rows cannot be PPO-eligible")
                binding = encoded[index].binding
                selected_groups[index] = self.bot_policy.select_group(
                    encoded[index],
                    state=states[(binding.environment_id, binding.episode_generation)],
                )
            for opponent_id, agent in self.opponent_agents.items():
                indices = [
                    index for index, (checkpoint_id, eligible) in enumerate(owners)
                    if checkpoint_id == opponent_id and not eligible
                ]
                if not indices:
                    continue
                chosen = agent.select([encoded[index] for index in indices])
                if len(chosen) != len(indices):
                    raise ValueError("opponent agent returned the wrong action count")
                model_queries += len(indices)
                for index, group in zip(indices, chosen, strict=True):
                    if not 0 <= int(group) < len(encoded[index].action_representatives):
                        raise ValueError("opponent agent selected an illegal action group")
                    selected_groups[index] = int(group)
            with torch.inference_mode():
                for checkpoint_id, indices in grouped.items():
                    policy = self._policy(checkpoint_id, current_policy_id)
                    policy.eval()
                    shards = pack(
                        [len(encoded[index].token_factors) for index in indices],
                        self.inference_token_budget,
                        max_padding_fraction=self.max_padding_fraction,
                    ).batches
                    with torch.autocast(
                        device_type=self.device,
                        dtype=torch.bfloat16,
                        enabled=self.use_bf16,
                    ):
                      for shard in shards:
                        shard_indices = [indices[local] for local in shard]
                        shard_lengths = [
                            len(encoded[index].token_factors)
                            for index in shard_indices
                        ]
                        self.profiler.observe(
                            "rollout.actor_rows_per_launch", len(shard_indices)
                        )
                        self.profiler.observe(
                            "rollout.actor_useful_tokens_per_launch",
                            sum(shard_lengths),
                        )
                        self.profiler.observe(
                            "rollout.actor_padded_tokens_per_launch",
                            max(shard_lengths) * len(shard_lengths),
                        )
                        with self.profiler.measure("rollout.batch_transfer"):
                            inputs = model_batch(
                                [encoded[index] for index in shard_indices],
                                device=self.device,
                                backend=self.backend,
                                pin_memory=(
                                    False if getattr(
                                        policy, "synchronous_model_batch", False
                                    ) else None
                                ),
                            )
                        with self.profiler.measure("rollout.inference_and_copy"):
                            with self.profiler.measure(
                                "rollout.actor_candidate_processing"
                            ):
                                # PPO recomputes entropy from the current
                                # policy. Rollout only needs the sampled action
                                # and its behavior log-probability.
                                output = policy.forward_actor(
                                    **inputs, compute_entropy=False,
                                )
                            selected_global = policy.sample(
                                output,
                                inputs["action_offsets"],
                                generator=self.action_generator,
                                deterministic=(
                                    checkpoint_id in self.deterministic_policy_ids
                                ),
                            )
                            copied = _copy_inference_results(
                                output, selected_global, inputs["action_offsets"],
                            )
                        for local, index in enumerate(shard_indices):
                            group, log_probability = (
                                column[local] for column in copied
                            )
                            selected_groups[index] = group
                            old_log_probabilities[index] = log_probability
            with self.profiler.measure("rollout.sample_materialization"):
                native_candidates = []
                for index, (row, group, owner) in enumerate(zip(
                    encoded, selected_groups, owners, strict=True,
                )):
                    representative = row.action_representatives[group]
                    native_candidates.append(row.native_candidates[representative])
                    store = self.adapter.histories.get(
                        row.binding.environment_id, row.binding.episode_generation
                    )
                    store_id = store.store_id
                    sample = RolloutSample(
                        binding=row.binding,
                        observation=ObservationRecord(
                            row.binding,
                            store_id,
                            store.next_sequence,
                            actor_query_offset=row.actor_query_index,
                            token_count=len(row.token_factors),
                        ),
                        actions=ActionSegment(
                            0,
                            len(row.native_candidates),
                            row.action_representatives,
                            row.action_factors,
                        ),
                        checkpoint_id=owner[0],
                        behavior_policy_version=curriculum.policy_version,
                        ppo_eligible=owner[1],
                        selected_group=group,
                        selected_native=representative,
                        old_log_probability=old_log_probabilities[index],
                        entropy=0.0,
                        old_boundary_rank_value=0.0,
                        frame_index=frame_by_binding[row.binding],
                        encoded=row,
                    )
                    key = (
                        row.binding.environment_id,
                        row.binding.episode_generation,
                        row.binding.seat,
                    )
                    previous = sample_tails.get(key)
                    if previous is not None:
                        samples[previous].successor = len(samples)
                    sample_tails[key] = len(samples)
                    sample_kyoku_tails[key] = len(samples)
                    samples.append(sample)
                    digest.update(
                        f"{row.binding}:{group}:{representative}:{store_id}".encode()
                    )
            with self.profiler.measure("rollout.env_step"):
                last_actions = tuple(native_candidates)
                try:
                    stepped = self.adapter.step(
                        [candidate.select() for candidate in native_candidates]
                    )
                except RuntimeError as error:
                    environment_ids = sorted({key[0] for key in pending_matches})
                    try:
                        failed_batch = self.adapter.inspect(
                            environment_ids, privileged=True
                        )
                    except Exception:
                        failed_batch = batch
                    diagnostic = self._write_stall_diagnostic(
                        failed_batch, pending_matches, last_actions
                    )
                    raise RuntimeError(
                        f"native env step failed: {error}; replay diagnostic: {diagnostic}"
                    ) from error
                env_calls += 1
                pending_decisions.clear()
                pending_decision_tokens = 0
                next_batches = [*frame_queue, stepped]
                from ..env.adapter import EnvBatch
                frame_queue[:] = [EnvBatch.merge(*next_batches)]
        if match_completions != target_matches:
            raise RuntimeError(
                "complete-match collector boundary count disagrees with its launch budget: "
                f"completed={match_completions}/{target_matches}"
            )
        if getattr(self.model, "verify_rollout_policy_replay", False):
            self._verify_rollout_policy_replay(
                samples, current_policy_id=current_policy_id,
            )
        model_queries += self._fill_deferred_critic_values(
            frames, current_policy_id
        )
        for sample in samples:
            frame = frames[sample.frame_index]
            sample.old_boundary_rank_value = frame.old_boundary_rank_value
            sample.terminal_placement = frame.terminal_placement
            sample.rank_boundary_supervision = frame.rank_boundary_supervision
            sample.rank_order_target = frame.rank_order_target
        if any(
            sample.frame_index < 0 or not frames[sample.frame_index].genuine_action
            for sample in samples
        ):
            raise RuntimeError("policy row is not bound to a genuine pre-action frame")
        return CollectionResult(
            samples=tuple(samples),
            frames=tuple(frames),
            decisions=len(samples),
            env_calls=env_calls,
            trajectory_digest=digest.hexdigest(),
            kyoku_completions=kyoku_completions,
            match_completions=match_completions,
            match_outcomes=tuple(match_outcomes),
            game_metrics=game_metrics.metrics(),
            continuation=batch,
            model_queries=model_queries,
        )

    def _verify_rollout_policy_replay(
        self, samples, *, current_policy_id,
    ):
        """Diagnose policy mismatch before target or optimizer code runs."""
        import torch
        from ..encoding.packing import model_batch, pack

        selected = tuple(
            sample for sample in samples
            if sample.ppo_eligible and sample.checkpoint_id == current_policy_id
        )
        differences = []
        policy = self._policy(current_policy_id, current_policy_id)
        policy.eval()
        plan = pack(
            [len(sample.encoded.token_factors) for sample in selected],
            self.inference_token_budget,
            max_padding_fraction=self.max_padding_fraction,
        )
        with torch.no_grad():
            for shard in plan.batches:
                rows = [selected[index] for index in shard]
                inputs = model_batch(
                    [sample.encoded for sample in rows], device=self.device,
                    backend=self.backend,
                    pin_memory=False,
                )
                with torch.autocast(
                    device_type=self.device, dtype=torch.bfloat16,
                    enabled=self.use_bf16,
                ):
                    output = policy.forward_actor(**inputs, policy_only=True)
                starts = inputs["action_offsets"][:-1]
                indices = starts + torch.tensor(
                    [sample.selected_group for sample in rows],
                    dtype=torch.long, device=starts.device,
                )
                replayed = output.log_probabilities.index_select(
                    0, indices
                ).float().cpu()
                old = torch.tensor([
                    sample.old_log_probability for sample in rows
                ])
                differences.append(replayed - old)
        difference = torch.cat(differences)
        replay_kl = (torch.expm1(difference) - difference).mean()
        if float(replay_kl) > 1e-3:
            raise RuntimeError(
                "rollout policy replay mismatch before target construction: "
                f"approximate_kl={float(replay_kl):.6g}, "
                f"mean={float(difference.mean()):.6g}, "
                f"abs_mean={float(difference.abs().mean()):.6g}, "
                f"max={float(difference.abs().max()):.6g}"
            )

    def _fill_deferred_critic_values(self, frames, current_policy_id):
        """Evaluate non-action public frames after rollout in large GPU packs."""
        import torch

        groups = {}
        for index, frame in enumerate(frames):
            if not frame.ppo_eligible:
                continue
            boundary_key = (
                frame.binding.environment_id,
                frame.binding.episode_generation,
                frame.binding.seat,
                frame.encoded.rank_boundary_features.tobytes(),
            )
            groups.setdefault(frame.checkpoint_id, {}).setdefault(
                boundary_key, []
            ).append(index)
        queries = 0
        with torch.no_grad():
            for checkpoint_id, boundary_groups in groups.items():
                policy = self._policy(checkpoint_id, current_policy_id)
                policy.eval()
                representatives = [
                    indices[0] for indices in boundary_groups.values()
                ]
                shards = pack(
                    [
                        len(frames[index].encoded.token_factors)
                        for index in representatives
                    ],
                    self.inference_token_budget,
                    max_padding_fraction=self.max_padding_fraction,
                ).batches
                for shard in shards:
                    shard_indices = [representatives[local] for local in shard]
                    self.profiler.observe(
                        "rollout.frame_critic_rows_per_launch", len(shard_indices)
                    )
                    with self.profiler.measure(
                        "rollout.frame_critic_batch_transfer"
                    ):
                        import numpy as np
                        inputs = {
                            "decision_seats": torch.as_tensor(
                                [frames[index].encoded.decision_seat
                                 for index in shard_indices],
                                dtype=torch.long, device=self.device,
                            ),
                            "rank_boundary_features": torch.as_tensor(
                                np.stack([
                                    frames[index].encoded.rank_boundary_features
                                    for index in shard_indices
                                ]),
                                dtype=torch.float32, device=self.device,
                            ),
                        }
                    with self.profiler.measure("rollout.frame_critic_forward"):
                        with torch.autocast(
                            device_type=self.device,
                            dtype=torch.bfloat16,
                            enabled=self.use_bf16,
                        ):
                            critic = policy.forward_critic(**inputs)
                    probabilities = (
                        critic.rank_probabilities.detach().float().cpu().tolist()
                    )
                    values = critic.rank_values.detach().cpu().tolist()
                    for local, representative in enumerate(shard_indices):
                        frame = frames[representative]
                        boundary_key = (
                            frame.binding.environment_id,
                            frame.binding.episode_generation,
                            frame.binding.seat,
                            frame.encoded.rank_boundary_features.tobytes(),
                        )
                        probability = tuple(
                            float(value) for value in probabilities[local]
                        )
                        for index in boundary_groups[boundary_key]:
                            frames[index].old_boundary_rank_value = float(
                                values[local]
                            )
                            frames[index].old_boundary_rank_probabilities = (
                                probability
                            )
                    queries += len(shard_indices)
        return queries

    def _write_stall_diagnostic(self, batch, pending_matches, last_actions):
        """Persist enough native state to replay a policy-dependent stall."""
        try:
            states = tuple(batch.transition.states)
            stalled = tuple(
                state for state in states
                if int(state.lifecycle) not in (3, 4) and not state.action_spaces
            )
            capture_ids = sorted({
                int(state.environment_id) for state in stalled
            } or {
                int(environment_id) for environment_id, _ in pending_matches
            })
            snapshots = self.adapter.snapshot(capture_ids)
            captured_keys = {
                (int(state.environment_id), int(state.episode_generation))
                for state in states if int(state.environment_id) in capture_ids
            }
            captured_keys.update(
                (int(environment_id), int(generation))
                for environment_id, generation in pending_matches
                if int(environment_id) in capture_ids
            )
            histories = {}
            for key in sorted(captured_keys):
                rows = self.adapter.histories.get(*key).rows[-64:]
                histories[f"{key[0]}:{key[1]}"] = [
                    {
                        **{name: value for name, value in row.items() if name != "payload"},
                        "payload_hex": bytes(row.get("payload", b"")).hex(),
                    }
                    for row in rows
                ]
            state_rows = [
                {
                    "environment_id": int(state.environment_id),
                    "episode_generation": int(state.episode_generation),
                    "lifecycle": int(state.lifecycle),
                    "phase": int(state.phase),
                    "frame_id": int(state.frame_id),
                    "eligible_mask": int(state.eligible_mask),
                    "decision_count": len(state.action_spaces),
                    "round_wind": int(state.round_wind),
                    "hand_number": int(state.hand_number),
                    "dealer": int(state.dealer),
                    "honba": int(state.honba),
                    "riichi_deposits": int(state.riichi_deposits),
                    "scores": list(map(int, state.scores)),
                    "live_wall_remaining": int(state.live_wall_remaining),
                }
                for state in states
            ]
            capture_set = set(capture_ids)
            action_rows = [
                {
                    "environment_id": int(candidate.select().environment_id),
                    "episode_generation": int(
                        candidate.select().episode_generation
                    ),
                    "frame_id": int(candidate.select().frame_id),
                    "seat": int(candidate.select().seat),
                    "candidate_index": int(candidate.candidate_index),
                    "kind": int(candidate.kind),
                    "source_seat": (
                        None
                        if candidate.source_seat is None
                        else int(candidate.source_seat)
                    ),
                    "tiles": list(map(int, candidate.tiles)),
                    "aux": int(candidate.aux),
                    "flags": int(candidate.flags),
                }
                for candidate in last_actions
                if int(candidate.select().environment_id) in capture_set
            ]
            import riichi

            payload = {
                "format": "zenith-native-env-stall-v1",
                "master_seed": int(self.adapter.env.master_seed),
                "rules_profile": str(self.adapter.env.rules_profile),
                "state_schema_version": int(riichi.STATE_SCHEMA_VERSION),
                "event_schema_version": int(riichi.EVENT_SCHEMA_VERSION),
                "decision_schema_version": int(riichi.DECISION_SCHEMA_VERSION),
                "snapshot_schema_version": int(riichi.SNAPSHOT_SCHEMA_VERSION),
                "transition_id": int(getattr(batch.transition, "transition_id", 0)),
                "pending_matches": [list(map(int, key)) for key in sorted(pending_matches)],
                "states": state_rows,
                "last_actions": action_rows,
                "recent_events": histories,
                "snapshots_hex": {
                    str(environment_id): snapshot.hex()
                    for environment_id, snapshot in sorted(snapshots.items())
                },
            }
            self.diagnostic_dir.mkdir(parents=True, exist_ok=True)
            identities = "-".join(map(str, capture_ids[:8])) or "none"
            path = self.diagnostic_dir / (
                f"stall-p{os.getpid()}-t{payload['transition_id']}-env{identities}.json"
            )
            temporary = path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8"
            )
            os.replace(temporary, path)
            return path
        except Exception as error:  # preserve the original rollout failure
            return f"<diagnostic capture failed: {type(error).__name__}: {error}>"

    def _owner(self, binding, current_policy_id):
        key = (binding.environment_id, binding.episode_generation)
        lineup = self.lineups.get(key)
        if lineup is None and self.lineup_provider is not None:
            lineup = self.lineup_provider(*key)
            self.lineups[key] = lineup
        if lineup is None:
            return current_policy_id, True
        checkpoint_id = lineup.seat_policy_ids[binding.seat]
        eligible = bool(lineup.learner_mask & (1 << binding.seat))
        if eligible and checkpoint_id != current_policy_id \
                and checkpoint_id not in self.policy_models:
            raise ValueError("learner lineup seat has no live league model")
        return checkpoint_id, eligible

    def _learner_mask(self, key):
        lineup = self.lineups.get(key)
        if lineup is None and self.lineup_provider is not None:
            lineup = self.lineup_provider(*key)
            self.lineups[key] = lineup
        return int(lineup.learner_mask) if lineup is not None else 0b1111

    def _policy(self, checkpoint_id, current_policy_id):
        if checkpoint_id == current_policy_id:
            return self.model
        if self.residency is not None:
            if self.model_loader is None:
                raise ValueError("historical residency requires a model loader")
            return self.residency.get(checkpoint_id, self.model_loader).model
        try:
            return self.policy_models[checkpoint_id]
        except KeyError as exc:
            raise ValueError(f"no model is available for checkpoint {checkpoint_id!r}") from exc


def _apply_boundaries(
    batch, samples, tails, *, kyoku_tails=None, match_outcomes=None,
    trajectory_indices=None, kyoku_indices=None,
):
    """Attach kyoku and match boundaries to each observer trajectory."""
    # Tests and external diagnostic callers historically supplied only match
    # tails. The collector supplies distinct kyoku tails so a seat that made no
    # decision in the current hand cannot attach its settlement to an action
    # from the previous hand.
    current_kyoku_tails = tails if kyoku_tails is None else kyoku_tails
    grouped = {}
    for event in batch.transition.events:
        grouped.setdefault(
            (event.environment_id, event.episode_generation), []
        ).append(event)
    kyoku_completions = 0
    match_completions = 0
    for (environment_id, generation), events in grouped.items():
        end_game = None
        end_scores = None
        completed_kyoku = None
        for event in events:
            payload = bytes(event.payload or b"")
            kind = int(event.kind)
            if kind == 16 and len(payload) == 20:
                end_scores = tuple(struct.unpack("<4i", payload[:16]))
                end_game = tuple(int(rank) - 1 for rank in payload[16:20])
                args = getattr(event, "args", (0, 0, 0, 0))
                completed_kyoku = int(args[0])
        if any(int(event.kind) == 15 for event in events):
            kyoku_completions += 1
            if kyoku_indices is not None:
                hand_rows = tuple(kyoku_indices.pop(
                    (environment_id, generation), ()
                ))
                _mark_boundary_rows(samples, hand_rows)
            for seat in range(4):
                key = (environment_id, generation, seat)
                index = current_kyoku_tails.get(key)
                if index is None:
                    continue
                samples[index].kyoku_boundary = True
            if kyoku_tails is not None:
                for seat in range(4):
                    kyoku_tails.pop((environment_id, generation, seat), None)
        if end_game is not None:
            match_completions += 1
            for seat in range(4):
                key = (environment_id, generation, seat)
                if trajectory_indices is None:
                    # Compatibility path for focused tests and external
                    # diagnostic callers that do not maintain the collector's
                    # trajectory-local index.
                    indices_for_seat = (
                        index for index, row in enumerate(samples)
                        if (
                            (binding := getattr(row, "binding", None)) is not None
                            and binding.environment_id == environment_id
                            and binding.episode_generation == generation
                            and binding.seat == seat
                        )
                    )
                else:
                    indices_for_seat = trajectory_indices.pop(key, ())
                for row_index in indices_for_seat:
                    samples[row_index].terminal_placement = int(end_game[seat])
                    encoded = getattr(samples[row_index], "encoded", None)
                    boundary = getattr(encoded, "rank_boundary_features", None)
                    if boundary is not None:
                        dealer = int(max(range(4), key=lambda value: boundary[4 + value]))
                        absolute_order = tuple(sorted(
                            range(4), key=lambda value: end_game[value]
                        ))
                        relative_order = tuple(
                            (value - dealer) % 4 for value in absolute_order
                        )
                        samples[row_index].rank_order_target = RANK_ORDER_INDEX[
                            relative_order
                        ]
                index = tails.get(key)
                if index is None:
                    continue
                sample = samples[index]
                sample.terminal_placement = int(end_game[seat])
                sample.match_boundary = True
                sample.terminal = True
            if match_outcomes is not None and end_scores is not None:
                if completed_kyoku is None or completed_kyoku <= 0:
                    raise ValueError("end_game event is missing its completed kyoku count")
                indices = [tails.get((environment_id, generation, seat)) for seat in range(4)]
                if all(index is not None for index in indices):
                    match_outcomes.append(RolloutMatchOutcome(
                        (int(environment_id), int(generation)),
                        tuple(samples[index].checkpoint_id for index in indices),
                        tuple(end_game),
                        tuple(end_scores),
                        completed_kyoku,
                    ))
    return kyoku_completions, match_completions


def _mark_boundary_rows(samples, indices):
    """Select one start-of-kyoku boundary row per learner policy."""
    if not indices:
        return
    supervised_policies = set()
    for index in indices:
        row = samples[index]
        if not getattr(row, "ppo_eligible", False):
            continue
        policy = getattr(row, "checkpoint_id", "single-policy")
        if policy in supervised_policies:
            continue
        row.rank_boundary_supervision = True
        supervised_policies.add(policy)


def _game_metric_values(counts):
    """Finalize player-opportunity and conditional gameplay statistics."""
    player_kyoku = float(counts["player_kyoku"])
    kyoku = float(counts["kyoku"])
    player_matches = float(counts["player_matches"])
    if player_kyoku <= 0 or kyoku <= 0 or player_matches <= 0:
        return {}
    return {
        "game/player_win_rate": counts["wins"] / player_kyoku,
        "game/player_deal_in_rate": counts["deal_ins"] / player_kyoku,
        "game/player_riichi_rate": counts["riichi_hands"] / player_kyoku,
        "game/player_calling_rate": counts["calling_hands"] / player_kyoku,
        "game/player_average_winning_points": counts["winning_points"] / max(
            1.0, counts["winning_point_events"]
        ),
        "game/player_average_deal_in_points": counts["deal_in_points"] / max(
            1.0, counts["deal_in_point_events"]
        ),
        "game/exhaustive_ryukyoku_rate": counts["exhaustive_ryukyoku"] / kyoku,
        "game/player_bankrupt_rate": counts["bankrupt_matches"] / player_matches,
        # These describe the composition of wins, not per-kyoku frequencies.
        "game/player_tsumo_rate": counts["tsumo_wins"] / max(1.0, counts["wins"]),
        "game/player_dama_rate": counts["dama_wins"] / max(1.0, counts["wins"]),
        "game/player_average_turns_before_winning": counts["winning_turns"] / max(
            1.0, counts["winning_turn_events"]
        ),
    }


def _copy_inference_results(
    output, selected_global, action_offsets,
):
    """Copy one inference shard to host with one synchronization point."""
    import torch

    if int(action_offsets[-1]) >= 1 << 24:
        raise ValueError("inference shard has too many candidates for packed result transfer")
    starts = torch.as_tensor(
        action_offsets[:-1], dtype=torch.long, device=selected_global.device
    )
    local_groups = selected_global - starts
    selected_logp = output.log_probabilities.index_select(0, selected_global)
    payload = torch.stack((
        local_groups.float(), selected_logp.float(),
    ), dim=1).detach().cpu().numpy()
    return (
        [int(row[0]) for row in payload],
        [float(row[1]) for row in payload],
    )
