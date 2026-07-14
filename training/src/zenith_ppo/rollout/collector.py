"""Batch-oriented native env rollout collection."""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import struct

import numpy as np

from ..encoding.event_cache import EventPrefixCache
from ..encoding.packing import encode_native_batch, model_batch, pack
from ..profiling import StageProfiler
from ..rewards.discard import DiscardScore, public_remaining, regret
from ..types import ActionSegment, ObservationRecord, RewardRecord, RolloutSample


@dataclass(frozen=True, slots=True)
class CollectionResult:
    samples: tuple[RolloutSample, ...]
    decisions: int
    env_calls: int
    trajectory_digest: str
    kyoku_completions: int
    match_completions: int
    kyoku_environment_coverage: float
    boundary_aligned: bool
    continuation: object | None = None


class Collector:
    def __init__(self, adapter, model, action_generator, *, device="cpu",
                 backend="sdpa", inference_token_budget=65536, use_bf16=False,
                 max_padding_fraction=None,
                 lineups=None, lineup_provider=None, policy_models=None,
                 residency=None, model_loader=None, profiler=None, event_cache=None):
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
        self.residency = residency
        self.model_loader = model_loader
        self.profiler = profiler or StageProfiler()
        self.event_cache = event_cache or EventPrefixCache()

    def collect(self, initial_batch, *, target_decisions: int, curriculum, streams,
                current_policy_id="current", critic_mode="privileged",
                complete_kyoku_per_env: bool = False,
                max_env_calls: int | None = None) -> CollectionResult:
        import torch

        if target_decisions <= 0:
            raise ValueError("target_decisions must be positive")
        batch = initial_batch
        samples: list[RolloutSample] = []
        eligible_decisions = 0
        tails: dict[tuple[int, int, int], int] = {}
        digest = sha256()
        env_calls = 0
        kyoku_completions = 0
        match_completions = 0
        initial_environments = {
            int(state.environment_id) for state in initial_batch.transition.states
        }
        crossed_kyoku = set()
        pending_kyoku = set(initial_environments) if complete_kyoku_per_env else set()
        self.model.eval()
        while eligible_decisions < target_decisions or pending_kyoku:
            if max_env_calls is not None and env_calls >= int(max_env_calls):
                raise RuntimeError(
                    "rollout frame horizon exhausted before decision/boundary target: "
                    f"eligible={eligible_decisions}/{target_decisions}, "
                    f"pending_kyoku={sorted(pending_kyoku)}"
                )
            with self.profiler.measure("rollout.encoding"):
                encoded = encode_native_batch(
                    batch,
                    self.adapter.histories,
                    critic_mode=critic_mode,
                    event_cache=self.event_cache,
                )
            if not encoded:
                raise RuntimeError("native env produced no decisions before rollout target")
            owners = [
                self._owner(row.binding, current_policy_id) for row in encoded
            ]
            selected_groups = [0] * len(encoded)
            old_log_probabilities = [0.0] * len(encoded)
            entropies = [0.0] * len(encoded)
            old_values = [0.0] * len(encoded)
            grouped = {}
            for index, (checkpoint_id, _) in enumerate(owners):
                grouped.setdefault(checkpoint_id, []).append(index)
            with torch.no_grad():
                for checkpoint_id, indices in grouped.items():
                    policy = self._policy(checkpoint_id, current_policy_id)
                    policy.eval()
                    shards = pack(
                        [len(encoded[index].token_factors) for index in indices],
                        self.inference_token_budget,
                        max_padding_fraction=self.max_padding_fraction,
                    ).batches
                    for shard in shards:
                        shard_indices = [indices[local] for local in shard]
                        with self.profiler.measure("rollout.batch_transfer"):
                            inputs = model_batch(
                                [encoded[index] for index in shard_indices],
                                device=self.device,
                                backend=self.backend,
                            )
                        with self.profiler.measure("rollout.inference_and_copy"):
                            with torch.autocast(
                                device_type=self.device,
                                dtype=torch.bfloat16,
                                enabled=self.use_bf16,
                            ):
                                output = policy(**inputs)
                            selected_global = policy.sample(
                                output,
                                inputs["action_offsets"],
                                generator=self.action_generator,
                            )
                            copied = _copy_inference_results(
                                output, selected_global, inputs["action_offsets"]
                            )
                        for local, index in enumerate(shard_indices):
                            group, log_probability, entropy, value = (
                                column[local] for column in copied
                            )
                            selected_groups[index] = group
                            old_log_probabilities[index] = log_probability
                            entropies[index] = entropy
                            old_values[index] = value
            with self.profiler.measure("rollout.discard_analysis"):
                discard_rewards = _discard_rewards(
                    batch, encoded, selected_groups, self.adapter.histories
                )
            with self.profiler.measure("rollout.sample_materialization"):
                native_actions = []
                for index, (decision, row, group, owner) in enumerate(
                    zip(batch.decisions, encoded, selected_groups, owners, strict=True)
                ):
                    representative = row.action_representatives[group]
                    native_actions.append(row.native_actions[representative])
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
                            len(row.native_actions),
                            row.action_representatives,
                            row.action_factors,
                        ),
                        checkpoint_id=owner[0],
                        behavior_policy_version=curriculum.policy_version,
                        ppo_eligible=owner[1],
                        selected_group=group,
                        selected_native=representative,
                        old_log_probability=old_log_probabilities[index],
                        entropy=entropies[index],
                        old_value=old_values[index],
                        reward=RewardRecord(
                            discard_reward=discard_rewards[index],
                            weights=curriculum.weights,
                            boundary_mode=curriculum.boundary_mode,
                        ),
                        encoded=row,
                    )
                    key = (
                        row.binding.environment_id,
                        row.binding.episode_generation,
                        row.binding.seat,
                    )
                    previous = tails.get(key)
                    if previous is not None:
                        samples[previous].successor = len(samples)
                    tails[key] = len(samples)
                    samples.append(sample)
                    eligible_decisions += int(sample.ppo_eligible)
                    digest.update(
                        f"{row.binding}:{group}:{representative}:{store_id}".encode()
                    )
            with self.profiler.measure("rollout.env_step"):
                batch = self.adapter.step(native_actions)
                env_calls += 1
                boundary_environments = {
                    int(event.environment_id)
                    for event in batch.transition.events
                    if int(event.kind) == 15
                }
                crossed_kyoku.update(boundary_environments & initial_environments)
                pending_kyoku.difference_update(boundary_environments)
                ended_kyoku, ended_matches = _apply_boundaries(batch, samples, tails)
                kyoku_completions += ended_kyoku
                match_completions += ended_matches
                batch = self.adapter.reset_completed(batch)
        for index in tails.values():
            if not samples[index].terminal and not samples[index].match_boundary:
                samples[index].truncated = True
                samples[index].bootstrap_value = samples[index].old_value
        coverage = len(crossed_kyoku) / max(1, len(initial_environments))
        return CollectionResult(
            samples=tuple(samples),
            decisions=len(samples),
            env_calls=env_calls,
            trajectory_digest=digest.hexdigest(),
            kyoku_completions=kyoku_completions,
            match_completions=match_completions,
            kyoku_environment_coverage=coverage,
            boundary_aligned=not pending_kyoku,
            continuation=batch,
        )

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
        if eligible and checkpoint_id != current_policy_id:
            raise ValueError("learner lineup seat is not bound to the current policy")
        return checkpoint_id, eligible

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


def _discard_rewards(batch, encoded, selected_groups, histories):
    import riichi

    unique: dict[tuple[bytes, int], int] = {}
    hands, melds = [], []
    candidates: list[list[tuple[int, int]]] = []
    open_melds = {
        (state.environment_id, decision.seat): sum(
            meld.seat == decision.seat for meld in state.melds
        )
        for state in batch.transition.states
        for decision in state.decisions
    }
    for decision, row in zip(batch.decisions, encoded, strict=True):
        rows = []
        counts = np.frombuffer(bytes(decision.concealed_counts), dtype=np.uint8).copy()
        for group, representative in enumerate(row.action_representatives):
            action = row.native_actions[representative]
            if int(action.kind) not in (1, 2) or not action.tiles:
                continue
            next_counts = counts.copy()
            next_counts[int(action.tiles[0]) // 4] -= 1
            opened = open_melds[(decision.environment_id, decision.seat)]
            key = (next_counts.tobytes(), opened)
            analysis_index = unique.get(key)
            if analysis_index is None:
                analysis_index = len(hands)
                unique[key] = analysis_index
                hands.append(next_counts)
                melds.append(opened)
            rows.append((group, analysis_index))
        candidates.append(rows)
    if not hands:
        return [0.0] * len(batch.decisions)
    analysis = riichi.analyze_hands(
        np.ascontiguousarray(hands, dtype=np.uint8),
        np.ascontiguousarray(melds, dtype=np.uint8),
    )
    # Multiple seats and action candidates commonly share one immutable event
    # store in a native frame. Materialize its current-kyoku public tiles once;
    # rescanning the full match for every candidate made rollout cost grow with
    # match age and incorrectly treated prior-kyoku tiles as still visible.
    public_by_store = {}
    rewards = []
    for decision, rows, selected in zip(batch.decisions, candidates, selected_groups, strict=True):
        if not rows or selected not in {group for group, _ in rows}:
            rewards.append(0.0)
            continue
        store_key = (decision.environment_id, decision.episode_generation)
        public_tiles = public_by_store.get(store_key)
        if public_tiles is None:
            public_tiles = _public_tiles(histories.get(*store_key).rows)
            public_by_store[store_key] = public_tiles
        scores = [
            DiscardScore(
                int(analysis.shanten[index, 0]),
                public_remaining(
                    decision.concealed_counts,
                    int(analysis.improving_type_mask[index]),
                    public_tiles,
                ),
            )
            for _, index in rows
        ]
        chosen = next(i for i, (group, _) in enumerate(rows) if group == selected)
        rewards.append(regret(scores, chosen))
    return rewards


def _public_tiles(events):
    """Return unique physical tiles made public during the current kyoku."""
    physical = set()
    start = 0
    for index in range(len(events) - 1, -1, -1):
        if int(events[index]["kind"]) == 2:
            start = index
            break
    for event in events[start:]:
        kind = int(event["kind"])
        if kind not in {4, 5, 6, 7, 8, 9, 10}:
            continue
        args = event.get("args", ())
        # Dahai's arg1 is the tsumogiri boolean, not a physical tile.
        # Dora likewise has only one tile argument. Call/kan events use every
        # physical argument and 255 for unused positions.
        values = args[:1] if kind in {4, 10} else args
        for value in values:
            value = int(value)
            if 0 <= value < 136:
                physical.add(value)
    return tuple(sorted(physical))


def _apply_boundaries(batch, samples, tails):
    """Attach terminal reward components to each seat's last decision."""
    grouped = {}
    for event in batch.transition.events:
        grouped.setdefault(
            (event.environment_id, event.episode_generation), []
        ).append(event)
    kyoku_completions = 0
    match_completions = 0
    for (environment_id, generation), events in grouped.items():
        settlement = None
        end_game = None
        for event in events:
            payload = bytes(event.payload or b"")
            if int(event.kind) in {13, 14} and len(payload) >= 16:
                candidate = struct.unpack("<4i", payload[-16:])
                if settlement is None:
                    settlement = candidate
            elif int(event.kind) == 16 and len(payload) == 20:
                end_game = tuple(int(rank) - 1 for rank in payload[16:20])
        if any(int(event.kind) == 15 for event in events):
            kyoku_completions += 1
            delta = settlement or (0, 0, 0, 0)
            for seat in range(4):
                index = tails.get((environment_id, generation, seat))
                if index is None:
                    continue
                sample = samples[index]
                sample.reward = replace(
                    sample.reward, kyoku_delta=delta[seat] / 1000.0
                )
                sample.kyoku_boundary = True
        if end_game is not None:
            match_completions += 1
            from ..rewards.ranking import rewards as ranking_rewards

            terminal_rewards = ranking_rewards(end_game)
            for seat in range(4):
                index = tails.get((environment_id, generation, seat))
                if index is None:
                    continue
                sample = samples[index]
                sample.reward = replace(
                    sample.reward, rank_reward=terminal_rewards[seat]
                )
                sample.match_boundary = True
                sample.terminal = True
    return kyoku_completions, match_completions


def _copy_inference_results(output, selected_global, action_offsets):
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
        local_groups.float(),
        selected_logp.float(),
        output.entropy.float(),
        output.values.float(),
    ), dim=1).detach().cpu().tolist()
    return (
        [int(row[0]) for row in payload],
        [float(row[1]) for row in payload],
        [float(row[2]) for row in payload],
        [float(row[3]) for row in payload],
    )
