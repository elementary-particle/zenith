"""Batch-oriented native env rollout collection."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from hashlib import sha256
import json
import os
from pathlib import Path
import struct

from ..encoding.event_cache import EventPrefixCache
from ..encoding.packing import encode_native_batch, model_batch, pack
from ..profiling import StageProfiler
from ..inference import CONSERVATIVE_BOT_ID, ConservativeBot
from ..types import (
    ActionSegment, ObservationRecord, RewardRecord, RolloutMatchOutcome, RolloutSample,
)


@dataclass(frozen=True, slots=True)
class CollectionResult:
    samples: tuple[RolloutSample, ...]
    decisions: int
    env_calls: int
    trajectory_digest: str
    kyoku_completions: int
    match_completions: int
    match_outcomes: tuple[RolloutMatchOutcome, ...] = ()
    game_metrics: dict[str, float] = field(default_factory=dict)
    continuation: object | None = None
    model_queries: int = 0
    rust_resolved_decisions: int = 0


class Collector:
    def __init__(self, adapter, model, action_generator, *, device="cpu",
                 backend="sdpa", inference_token_budget=65536, use_bf16=False,
                 max_padding_fraction=None,
                 lineups=None, lineup_provider=None, policy_models=None,
                 residency=None, model_loader=None, profiler=None, event_cache=None,
                 teacher_config=None, bot_policy=None, diagnostic_dir=None):
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
        self.teacher_config = dict(teacher_config or {})
        self.bot_policy = bot_policy or ConservativeBot()
        self.diagnostic_dir = Path(
            diagnostic_dir or Path.cwd() / "native-env-diagnostics"
        )

    def collect(self, initial_batch, *, target_matches: int, curriculum, streams,
                current_policy_id="current", critic_mode="privileged",
                max_env_calls: int | None = None) -> CollectionResult:
        import torch

        if target_matches <= 0:
            raise ValueError("target_matches must be positive")
        batch = initial_batch
        launched = {
            (int(state.environment_id), int(state.episode_generation))
            for state in batch.transition.states if int(state.lifecycle) != 3
        }
        if len(launched) != int(target_matches):
            raise ValueError(
                f"rollout must launch exactly {target_matches} fresh matches, got {len(launched)}"
            )
        pending_matches = set(launched)
        samples: list[RolloutSample] = []
        tails: dict[tuple[int, int, int], int] = {}
        kyoku_tails: dict[tuple[int, int, int], int] = {}
        digest = sha256()
        env_calls = 0
        kyoku_completions = 0
        match_completions = 0
        match_outcomes = []
        model_queries = 0
        last_actions = ()
        game_metrics = {
            "open_wins": 0.0,
            "closed_wins": 0.0,
            "deal_ins_after_opponent_riichi": 0.0,
            "exhaustive_ryukyoku": 0.0,
            "exhaustive_ryukyoku_tenpai_score": 0.0,
        }
        self.model.eval()
        while pending_matches:
            # A native transition is both an action frame and the receipt for
            # the preceding step. Consume its boundaries before asking the
            # model for another action. In particular, an all-terminal
            # transition legitimately contains no decisions.
            ended_kyoku, ended_matches = _apply_boundaries(
                batch, samples, tails, kyoku_tails=kyoku_tails,
                match_outcomes=match_outcomes,
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
                    f"native env returned terminal matches outside the frozen launch set: "
                    f"{sorted(unexpected)}"
                )
            if terminal_matches and ended_matches != len(terminal_matches):
                raise RuntimeError(
                    "native terminal states are missing matching end_game events: "
                    f"terminal={sorted(terminal_matches)}, end_game_events={ended_matches}"
                )
            pending_matches.difference_update(terminal_matches)
            if not pending_matches:
                break
            if max_env_calls is not None and env_calls >= int(max_env_calls):
                raise RuntimeError(
                    "rollout frame horizon exhausted before every match completed: "
                    f"completed={target_matches - len(pending_matches)}/{target_matches}"
                )
            with self.profiler.measure("rollout.encoding"):
                encoded = encode_native_batch(
                    batch,
                    self.adapter.histories,
                    critic_mode=critic_mode,
                    event_cache=self.event_cache,
                    teacher_config=self.teacher_config,
                    teacher_selector=lambda binding: (
                        (owner := self._owner(binding, current_policy_id))[1]
                        or owner[0] == CONSERVATIVE_BOT_ID
                    ),
                    profiler=self.profiler,
                )
            if not encoded:
                diagnostic = self._write_stall_diagnostic(
                    batch, pending_matches, last_actions
                )
                raise RuntimeError(
                    "native env produced no queryable decisions before match completion; "
                    f"replay diagnostic: {diagnostic}"
                )
            owners = [
                self._owner(row.binding, current_policy_id) for row in encoded
            ]
            selected_groups = [0] * len(encoded)
            old_log_probabilities = [0.0] * len(encoded)
            entropies = [0.0] * len(encoded)
            old_score_values = [0.0] * len(encoded)
            old_rank_values = [0.0] * len(encoded)
            grouped = {}
            for index, (checkpoint_id, _) in enumerate(owners):
                if checkpoint_id != CONSERVATIVE_BOT_ID:
                    grouped.setdefault(checkpoint_id, []).append(index)
                    model_queries += 1
            states = {
                (int(state.environment_id), int(state.episode_generation)): state
                for state in batch.transition.states
            }
            for index, (checkpoint_id, eligible) in enumerate(owners):
                if checkpoint_id != CONSERVATIVE_BOT_ID:
                    continue
                if eligible:
                    raise ValueError("deterministic bot rows cannot be PPO-eligible")
                binding = encoded[index].binding
                selected_groups[index] = self.bot_policy.select_group(
                    encoded[index],
                    state=states[(binding.environment_id, binding.episode_generation)],
                )
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
                                with self.profiler.measure(
                                    "rollout.actor_candidate_processing"
                                ):
                                    output = policy.forward_actor(**inputs)
                                if checkpoint_id == current_policy_id:
                                    with self.profiler.measure("rollout.oracle_forward"):
                                        critic = policy.forward_critic(**inputs)
                                    from types import SimpleNamespace
                                    output = SimpleNamespace(
                                        **output.__dict__, **critic.__dict__
                                    )
                            selected_global = policy.sample(
                                output,
                                inputs["action_offsets"],
                                generator=self.action_generator,
                            )
                            copied = _copy_inference_results(
                                output, selected_global, inputs["action_offsets"]
                            )
                        for local, index in enumerate(shard_indices):
                            group, log_probability, entropy, score_value, rank_value = (
                                column[local] for column in copied
                            )
                            selected_groups[index] = group
                            old_log_probabilities[index] = log_probability
                            entropies[index] = entropy
                            old_score_values[index] = score_value
                            old_rank_values[index] = rank_value
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
                        old_score_value=old_score_values[index],
                        old_rank_value=old_rank_values[index],
                        reward=RewardRecord(
                            weights=curriculum.weights,
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
                    kyoku_tails[key] = len(samples)
                    samples.append(sample)
                    kind = int(native_actions[-1].kind)
                    if sample.ppo_eligible and kind in (8, 9):
                        state = states[(row.binding.environment_id, row.binding.episode_generation)]
                        is_open = any(
                            int(meld.seat) == row.binding.seat and int(meld.kind) != 4
                            for meld in state.melds
                        )
                        game_metrics["open_wins" if is_open else "closed_wins"] += 1
                    digest.update(
                        f"{row.binding}:{group}:{representative}:{store_id}".encode()
                    )
            with self.profiler.measure("rollout.env_step"):
                dealt_in = set()
                for action, row in zip(native_actions, encoded, strict=True):
                    if int(action.kind) != 8 or action.source_seat is None:
                        continue
                    key = (row.binding.environment_id, row.binding.episode_generation)
                    loser = int(action.source_seat)
                    lineup = self.lineups.get(key)
                    loser_is_learner = lineup is None or bool(lineup.learner_mask & (1 << loser))
                    state = states[key]
                    if loser_is_learner and any(
                        seat != loser and int(state.seat_flags[seat]) & 0b11 for seat in range(4)
                    ):
                        dealt_in.add((*key, loser))
                game_metrics["deal_ins_after_opponent_riichi"] += len(dealt_in)
                last_actions = tuple(native_actions)
                try:
                    batch = self.adapter.step(native_actions)
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
                for event in batch.transition.events:
                    args = getattr(event, "args", (0, 0, 0, 0))
                    if int(event.kind) != 14 or int(args[0]) != 3:
                        continue
                    game_metrics["exhaustive_ryukyoku"] += 1
                    payload = bytes(event.payload or b"")
                    if len(payload) >= 16:
                        deltas = struct.unpack("<4i", payload[-16:])
                        key = (int(event.environment_id), int(event.episode_generation))
                        lineup = self.lineups.get(key)
                        mask = lineup.learner_mask if lineup is not None else 0b1111
                        game_metrics["exhaustive_ryukyoku_tenpai_score"] += sum(
                            deltas[seat] / 1000.0 for seat in range(4) if mask & (1 << seat)
                        )
        if match_completions != target_matches:
            raise RuntimeError(
                "complete-match collector boundary count disagrees with its frozen launch set: "
                f"completed={match_completions}/{target_matches}"
            )
        open_learner_tails = [
            index for index in tails.values()
            if samples[index].ppo_eligible and not samples[index].terminal
        ]
        if open_learner_tails:
            raise RuntimeError(f"completed rollout contains open learner tails: {open_learner_tails}")
        if any(sample.truncated or sample.successor is None and not sample.terminal
               for sample in samples if sample.ppo_eligible):
            raise RuntimeError("complete-match rollout contains a bootstrap trajectory")
        return CollectionResult(
            samples=tuple(samples),
            decisions=len(samples),
            env_calls=env_calls,
            trajectory_digest=digest.hexdigest(),
            kyoku_completions=kyoku_completions,
            match_completions=match_completions,
            match_outcomes=tuple(match_outcomes),
            game_metrics=_game_metrics_per_kyoku(game_metrics, kyoku_completions),
            continuation=batch,
            model_queries=model_queries,
        )

    def _write_stall_diagnostic(self, batch, pending_matches, last_actions):
        """Persist enough native state to replay a policy-dependent stall."""
        try:
            states = tuple(batch.transition.states)
            stalled = tuple(
                state for state in states
                if int(state.lifecycle) not in (3, 4) and not state.decisions
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
                    "decision_count": len(state.decisions),
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
                    "environment_id": int(action.environment_id),
                    "episode_generation": int(action.episode_generation),
                    "frame_id": int(action.frame_id),
                    "seat": int(action.seat),
                    "action_index": int(action.action_index),
                    "kind": int(action.kind),
                    "source_seat": (
                        None if action.source_seat is None else int(action.source_seat)
                    ),
                    "tiles": list(map(int, action.tiles)),
                    "aux": int(action.aux),
                    "flags": int(action.flags),
                }
                for action in last_actions
                if int(action.environment_id) in capture_set
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


def _apply_boundaries(
    batch, samples, tails, *, kyoku_tails=None, match_outcomes=None,
):
    """Attach score changes and terminal rewards to each seat's last decision.

    Settlements contain the score movement at the end of a kyoku, but riichi
    deposits are deducted earlier when the declaration is accepted.  Recording
    that event here makes the accumulated kyoku reward equal the actual score
    delta instead of treating riichi sticks as free.
    """
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
        settlement = None
        end_game = None
        end_scores = None
        completed_kyoku = None
        for event in events:
            payload = bytes(event.payload or b"")
            kind = int(event.kind)
            if kind == 12:
                seat = int(event.actor_seat)
                index = current_kyoku_tails.get((environment_id, generation, seat))
                if index is not None:
                    sample = samples[index]
                    sample.reward = replace(
                        sample.reward,
                        kyoku_delta=sample.reward.kyoku_delta - 1.0,
                    )
            elif kind in {13, 14} and len(payload) >= 16:
                candidate = struct.unpack("<4i", payload[-16:])
                if settlement is None:
                    settlement = candidate
            elif kind == 16 and len(payload) == 20:
                end_scores = tuple(struct.unpack("<4i", payload[:16]))
                end_game = tuple(int(rank) - 1 for rank in payload[16:20])
                args = getattr(event, "args", (0, 0, 0, 0))
                completed_kyoku = int(args[0])
        if any(int(event.kind) == 15 for event in events):
            kyoku_completions += 1
            delta = settlement or (0, 0, 0, 0)
            for seat in range(4):
                key = (environment_id, generation, seat)
                index = current_kyoku_tails.get(key)
                if index is None:
                    continue
                sample = samples[index]
                sample.reward = replace(
                    sample.reward,
                    kyoku_delta=sample.reward.kyoku_delta + delta[seat] / 1000.0,
                )
                sample.kyoku_boundary = True
            if kyoku_tails is not None:
                for seat in range(4):
                    kyoku_tails.pop((environment_id, generation, seat), None)
        if end_game is not None:
            match_completions += 1
            from ..rewards.ranking import rewards as ranking_rewards

            terminal_rewards = ranking_rewards(end_game)
            for seat in range(4):
                for row in samples:
                    binding = getattr(row, "binding", None)
                    if (
                        binding is not None
                        and binding.environment_id == environment_id
                        and binding.episode_generation == generation
                        and binding.seat == seat
                    ):
                        row.terminal_placement = int(end_game[seat])
                index = tails.get((environment_id, generation, seat))
                if index is None:
                    continue
                sample = samples[index]
                sample.terminal_placement = int(end_game[seat])
                sample.reward = replace(
                    sample.reward, rank_reward=terminal_rewards[seat]
                )
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


def _game_metrics_per_kyoku(counts, kyoku_completions: int):
    """Normalize learner gameplay events only when the denominator exists."""
    completed = int(kyoku_completions)
    if completed <= 0:
        return {}
    denominator = float(completed)
    return {
        "game/open_wins_per_kyoku": counts["open_wins"] / denominator,
        "game/closed_wins_per_kyoku": counts["closed_wins"] / denominator,
        "game/deal_ins_after_opponent_riichi_per_kyoku": (
            counts["deal_ins_after_opponent_riichi"] / denominator
        ),
        "game/exhaustive_ryukyoku_rate": (
            counts["exhaustive_ryukyoku"] / denominator
        ),
        "game/exhaustive_ryukyoku_tenpai_score_per_kyoku": (
            counts["exhaustive_ryukyoku_tenpai_score"] / denominator
        ),
    }


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
        output.score_values.float(),
        output.rank_values.float(),
    ), dim=1).detach().cpu().tolist()
    return (
        [int(row[0]) for row in payload],
        [float(row[1]) for row in payload],
        [float(row[2]) for row in payload],
        [float(row[3]) for row in payload],
        [float(row[4]) for row in payload],
    )
