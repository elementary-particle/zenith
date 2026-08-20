"""Streaming Tenhou-to-MJAI ingestion through the native replay boundary."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
import gzip
import json
import os
from pathlib import Path
import zipfile
from types import SimpleNamespace

from ..encoding.event_cache import EventPrefixCache
from ..encoding.packing import encode_native_batch
from ..env.adapter import EnvAdapter, EnvBatch


_HONORS = {"E": 27, "S": 28, "W": 29, "N": 30, "P": 31, "F": 32, "C": 33}
_FAMILY = {
    0: "pass", 1: "discard", 2: "riichi", 3: "call", 4: "call",
    5: "kan", 6: "kan", 7: "kan", 8: "win", 9: "win", 10: "abortive",
}
_CRITIC_FIELDS = (
    "terminal_placement", "rank_boundary_supervision", "rank_order_target",
    "hand_outcome_target", "hand_score_delta",
)

HAND_OUTCOME_DRAW = 0
HAND_OUTCOME_WIN = 1
HAND_OUTCOME_DEAL_IN = 2
HAND_OUTCOME_OTHER_WIN = 3


def _validate_game(events):
    start = next(
        (event for event in events if event.get("type") == "start_game"),
        None,
    )
    if start is None:
        raise ValueError("missing_start_game")
    if (
        not start.get("aka_flag", False)
        or int(start.get("kyoku_first", -1)) != 0
    ):
        raise ValueError("unsupported_game_profile")


@dataclass(frozen=True, slots=True)
class BCExample:
    encoded: object
    target: int
    family: str
    critic: object | None = None


@dataclass(frozen=True, slots=True)
class PhysicalKyoku:
    hanchan: dict
    events: tuple[dict, ...]


def tile_type(tile: str) -> int:
    tile = str(tile)
    if tile in _HONORS:
        return _HONORS[tile]
    if len(tile) not in (2, 3) or tile[0] not in "123456789" \
            or tile[1] not in "mps" or (len(tile) == 3 and tile[2] != "r"):
        raise ValueError(f"unsupported MJAI tile {tile!r}")
    return "mps".index(tile[1]) * 9 + int(tile[0]) - 1


def _is_red(tile: str) -> bool:
    return str(tile).endswith("r")


def _matches(physical: int, tile: str) -> bool:
    kind, copy = divmod(int(physical), 4)
    return kind == tile_type(tile) and (
        _is_red(tile) == (kind in (4, 13, 22) and copy == 0)
    )


class _Allocator:
    def __init__(self):
        self.available = {}
        for kind in range(34):
            ids = [kind * 4 + copy for copy in range(4)]
            if kind in (4, 13, 22):
                self.available[(kind, True)] = ids[:1]
                self.available[(kind, False)] = ids[1:]
            else:
                self.available[(kind, False)] = ids

    def take(self, tile: str) -> int:
        key = (tile_type(tile), _is_red(tile))
        values = self.available.get(key)
        if not values:
            raise ValueError(f"tile conservation exceeded for {tile!r}")
        return values.pop(0)

    def remaining(self):
        return sorted(tile for values in self.available.values() for tile in values)


def _remove(hand, tile, *, current=None):
    if current is not None and current in hand and _matches(current, tile):
        hand.remove(current)
        return current
    for physical in hand:
        if _matches(physical, tile):
            hand.remove(physical)
            return physical
    raise ValueError(f"recorded tile {tile!r} is absent from concealed hand")


def _split_kyoku(events):
    current = None
    for event in events:
        if event.get("type") == "start_kyoku":
            if current is not None:
                yield tuple(current)
            current = [event]
        elif current is not None:
            current.append(event)
            if event.get("type") == "end_kyoku":
                yield tuple(current)
                current = None
    if current is not None:
        yield tuple(current)


def physicalize_kyoku(rows, *, environment_id=0, completed_kyoku=0):
    """Convert one complete MJAI kyoku to an exact physical wall and events."""
    if not rows or rows[0].get("type") != "start_kyoku":
        raise ValueError("kyoku must begin with start_kyoku")
    start = rows[0]
    allocator = _Allocator()

    dora_names = [start["dora_marker"]] + [
        row["dora_marker"] for row in rows if row.get("type") == "dora"
    ]
    ura_names = []
    for row in rows:
        candidate = list(row.get("ura_markers", ()))
        if len(candidate) > len(ura_names):
            if ura_names and candidate[:len(ura_names)] != ura_names:
                raise ValueError("inconsistent ura markers in multiple hora records")
            ura_names = candidate
        elif candidate and candidate != ura_names[:len(candidate)]:
            raise ValueError("inconsistent ura markers in multiple hora records")
    dora_ids = [allocator.take(tile) for tile in dora_names]
    ura_ids = [allocator.take(tile) for tile in ura_names]

    hands = [[allocator.take(tile) for tile in hand] for hand in start["tehais"]]
    if len(hands) != 4 or any(len(hand) != 13 for hand in hands):
        raise ValueError("replay requires four thirteen-tile starting hands")
    deal = []
    for packet in range(3):
        for hand in hands:
            deal.extend(hand[packet * 4:(packet + 1) * 4])
    deal.extend(hand[12] for hand in hands)

    live_draws, rinshan_draws, output = [], [], []
    current_draw = [None] * 4
    last_discard = None
    melds = [[] for _ in range(4)]
    expect_rinshan = False
    pending_kan = None
    dora_index = 1
    for source in rows[1:]:
        name = source.get("type")
        if name in {"end_kyoku", "end_game"}:
            continue
        row = {
            "environment_id": int(environment_id), "kind": name,
            "actor_seat": source.get("actor"), "target_seat": source.get("target"),
            "tile": None, "consumed": (),
            "tsumogiri": bool(source.get("tsumogiri", False)),
            "deltas": tuple(source.get("deltas", (0, 0, 0, 0))),
            "dora_marker": None, "ura_markers": (),
        }
        if name == "tsumo":
            actor = int(source["actor"])
            physical = allocator.take(source["pai"])
            hands[actor].append(physical)
            current_draw[actor] = physical
            (rinshan_draws if expect_rinshan else live_draws).append(physical)
            expect_rinshan = False
            pending_kan = None
            row["tile"] = physical
        elif name == "dahai":
            actor = int(source["actor"])
            physical = _remove(
                hands[actor], source["pai"],
                current=current_draw[actor] if source.get("tsumogiri") else None,
            )
            row["tile"] = physical
            current_draw[actor] = None
            last_discard = (actor, physical)
        elif name in {"chi", "pon", "daiminkan"}:
            actor = int(source["actor"])
            if last_discard is None or last_discard[0] != int(source["target"]):
                raise ValueError("call does not target the latest discard")
            called = last_discard[1]
            if not _matches(called, source["pai"]):
                raise ValueError("called tile does not match latest discard")
            consumed = [_remove(hands[actor], tile) for tile in source["consumed"]]
            row["tile"], row["consumed"] = called, tuple(consumed)
            melds[actor].append({"kind": name, "tiles": [called, *consumed]})
            current_draw[actor] = None
            expect_rinshan = name == "daiminkan"
        elif name == "ankan":
            actor = int(source["actor"])
            consumed = [_remove(hands[actor], tile) for tile in source["consumed"]]
            row["consumed"] = tuple(consumed)
            row["tile"] = consumed[0]
            melds[actor].append({"kind": name, "tiles": consumed})
            current_draw[actor] = None
            expect_rinshan = True
            pending_kan = (actor, consumed[0])
        elif name == "kakan":
            actor = int(source["actor"])
            added = _remove(hands[actor], source["pai"])
            target = next((meld for meld in reversed(melds[actor])
                           if meld["kind"] == "pon"
                           and meld["tiles"][0] // 4 == added // 4), None)
            if target is None:
                raise ValueError("kakan has no preceding pon")
            target["kind"] = "kakan"
            target["tiles"].append(added)
            row["tile"] = added
            row["consumed"] = tuple(target["tiles"][:3])
            current_draw[actor] = None
            expect_rinshan = True
            pending_kan = (actor, added)
        elif name == "dora":
            if dora_index >= len(dora_ids):
                raise ValueError("unexpected dora event")
            row["dora_marker"] = dora_ids[dora_index]
            dora_index += 1
        elif name == "hora":
            actor, target = int(source["actor"]), int(source["target"])
            if actor == target:
                tile = current_draw[actor]
            elif pending_kan is not None and pending_kan[0] == target:
                # Chankan wins on the proposed added/concealed-kan tile, not
                # on the older public discard retained in last_discard.
                tile = pending_kan[1]
            else:
                tile = None if last_discard is None else last_discard[1]
            if tile is None:
                raise ValueError("hora is missing its winning tile")
            row["tile"] = tile
            row["ura_markers"] = tuple(
                ura_ids[index] for index, _ in enumerate(source.get("ura_markers", ()))
            )
        elif name in {"reach", "reach_accepted", "ryukyoku"}:
            pass
        else:
            raise ValueError(f"unsupported in-kyoku MJAI event {name!r}")
        output.append(row)

    wall = [None] * 136
    wall[:52] = deal
    wall[52:52 + len(live_draws)] = live_draws
    for index, physical in enumerate(rinshan_draws):
        wall[135 - index] = physical
    for index, physical in enumerate(dora_ids):
        wall[130 - index * 2] = physical
    for index, physical in enumerate(ura_ids):
        wall[131 - index * 2] = physical
    remaining = iter(allocator.remaining())
    for index, value in enumerate(wall):
        if value is None:
            wall[index] = next(remaining)
    try:
        next(remaining)
    except StopIteration:
        pass
    else:
        raise AssertionError("replay wall did not consume every physical tile")
    if sorted(wall) != list(range(136)):
        raise ValueError("reconstructed wall is not a physical tile permutation")

    return PhysicalKyoku(
        {
            "environment_id": int(environment_id),
            "round_wind": {"E": 0, "S": 1, "W": 2, "N": 3}[start["bakaze"]],
            "hand_number": int(start["kyoku"]) - 1,
            "dealer": int(start["oya"]),
            "honba": int(start["honba"]),
            "riichi_deposits": int(start["kyotaku"]),
            "completed_kyoku": int(completed_kyoku),
            "scores": tuple(map(int, start["scores"])),
            "wall": tuple(wall),
        },
        tuple(output),
    )


def _native_hanchan(riichi, row):
    return riichi.ReplayHanchan(**row)


def _native_event(riichi, row):
    return riichi.ReplayEvent(**row)


def _event_group(events, index, phase):
    first = events[index]["kind"]
    if first == "reach":
        return events[index:index + 2]
    if first == "hora":
        end = index + 1
        while end < len(events) and events[end]["kind"] == "hora":
            end += 1
        return events[index:end]
    if first in {"ankan", "kakan", "daiminkan"}:
        # Submit only the declaration. A kan may resolve immediately (and
        # emit its dora/draw automatically) or pause at a genuine chankan
        # decision. The emitted-event count advances over automatic output in
        # the former case; keeping the source draw out of this input lets the
        # latter case resolve on the following replay call.
        return events[index:index + 1]
    if first in {"reach_accepted", "dora"}:
        end = index + 1
        while end < len(events) and events[end]["kind"] in {
            "reach_accepted", "dora",
        }:
            end += 1
        if end < len(events) and events[end]["kind"] in {
            "tsumo", "dahai", "chi", "pon", "daiminkan", "ankan",
            "kakan", "hora",
        }:
            if events[end]["kind"] == "hora":
                while end < len(events) and events[end]["kind"] == "hora":
                    end += 1
            else:
                end += 1
        return events[index:end]
    return events[index:index + 1]


def _terminal_draw_metadata(state):
    """Recover exhaustive-draw labels omitted by ReplayEvent.

    The replay event schema preserves settlement deltas but not terminal reason
    or the tenpai mask.  At the terminal privileged state those labels are
    authoritative: an exhausted live wall identifies an exhaustive draw, and
    native shanten analysis recovers the exact four-seat configuration.  This
    metadata supervises critics only; it is never added to actor observations.
    """
    if int(state.live_wall_remaining) != 0:
        return False, None
    import numpy as np
    import riichi

    hidden = state.hidden
    if hidden is None:
        raise ValueError("privileged replay state lacks terminal concealed hands")
    counts = np.ascontiguousarray(np.stack([
        np.frombuffer(row, dtype=np.uint8) if isinstance(row, bytes)
        else np.asarray(row, dtype=np.uint8)
        for row in hidden.concealed_counts
    ]))
    open_melds = np.zeros(4, dtype=np.uint8)
    for meld in state.melds:
        open_melds[int(meld.seat)] += 1
    shanten = np.asarray(
        riichi.evaluate_hand_efficiency(counts, open_melds).shanten
    )
    mask = sum(int(int(shanten[seat, 0]) == 0) << seat for seat in range(4))
    return True, mask


def _hand_outcome_targets(events):
    """Return the public terminal outcome class for each seat in one kyoku."""
    wins = {
        int(event["actor_seat"])
        for event in events
        if event["kind"] == "hora" and event.get("actor_seat") is not None
    }
    deal_ins = {
        int(event["target_seat"])
        for event in events
        if event["kind"] == "hora"
        and event.get("actor_seat") is not None
        and event.get("target_seat") is not None
        and int(event["actor_seat"]) != int(event["target_seat"])
    }
    if not wins:
        return (HAND_OUTCOME_DRAW,) * 4
    return tuple(
        HAND_OUTCOME_WIN if seat in wins else (
            HAND_OUTCOME_DEAL_IN if seat in deal_ins
            else HAND_OUTCOME_OTHER_WIN
        )
        for seat in range(4)
    )


def _set_hand_supervision(rows, indices, kyoku, final_scores):
    """Attach authoritative selected-action outcomes to every hand decision.

    These labels are training-only hindsight. They never enter the public
    observation and therefore cannot leak through actor inference.
    """
    outcomes = _hand_outcome_targets(kyoku.events)
    initial_scores = tuple(map(int, kyoku.hanchan["scores"]))
    deltas = tuple(
        int(final) - initial
        for initial, final in zip(initial_scores, final_scores, strict=True)
    )
    for index in indices:
        row = rows[index]
        seat = int(row.binding.seat)
        row.hand_outcome_target = outcomes[seat]
        row.hand_score_delta = deltas[seat]


def replay_game(
    events, *, maximum=None, include_public_snapshot=True,
):
    """Replay and buffer one game; any mismatch rejects the whole game."""
    import riichi

    env = riichi.Env(
        1, master_seed=0, num_threads=1,
        rules_profile=riichi.TENHOU_RULES_PROFILE, privileged=True,
    )
    adapter = EnvAdapter(env)
    event_cache = EventPrefixCache(max_entries=64)
    examples = []
    critic_rows = []
    final_scores = None
    completed = 0
    try:
        source_kyokus = tuple(_split_kyoku(events))
        for kyoku_position, source_rows in enumerate(source_kyokus):
            kyoku = physicalize_kyoku(
                source_rows, environment_id=0, completed_kyoku=completed
            )
            batch = adapter.load_hanchan([_native_hanchan(riichi, kyoku.hanchan)])
            index = 0
            hand_indices = []
            while index < len(kyoku.events):
                state = batch.transition.states[0]
                encoded = ()
                decisions = tuple(state.action_spaces)
                if decisions:
                    encoded = encode_native_batch(
                        batch, adapter.histories,
                        event_cache=event_cache,
                        include_public_snapshot=include_public_snapshot,
                    )
                group = _event_group(kyoku.events, index, int(state.phase))
                result = adapter.apply_events([
                    _native_event(riichi, row) for row in group
                ])
                emitted = tuple(result.transition.events)
                source_tail = kyoku.events[index:index + len(emitted)]
                if len(source_tail) != len(emitted):
                    raise ValueError("native replay emitted events beyond the source record")
                for native_event, source_event in zip(emitted, source_tail, strict=True):
                    if native_event.kind_name != source_event["kind"]:
                        raise ValueError(
                            "native/source automatic event mismatch: "
                            f"native={native_event.kind_name}, source={source_event['kind']}"
                        )
                    for field in ("actor_seat", "target_seat"):
                        expected = source_event[field]
                        if expected is not None and getattr(native_event, field) != expected:
                            raise ValueError(
                                f"native/source {field} mismatch for {native_event.kind_name}"
                            )
                if decisions:
                    selected = {
                        (int(action.frame_id), int(action.seat)): int(action.candidate_index)
                        for action in result.transition.applied_selections
                    }
                    for decision, row in zip(decisions, encoded, strict=True):
                        native = selected[(int(decision.frame_id), int(decision.seat))]
                        target = next(group_index for group_index, members in enumerate(row.action_members)
                                      if native in members)
                        family = _FAMILY[int(decision.candidates[native].kind)]
                        critic = SimpleNamespace(
                            binding=row.binding,
                            encoded=row,
                            ppo_eligible=True,
                            rank_boundary_supervision=False,
                            rank_order_target=-1,
                            terminal_placement=-1,
                            hand_outcome_target=-1,
                            hand_score_delta=0,
                        )
                        examples.append(BCExample(row, target, family, critic))
                        critic_rows.append(critic)
                        hand_indices.append(len(critic_rows) - 1)
                batch = result
                index += len(emitted)
            if hand_indices:
                critic_rows[hand_indices[0]].rank_boundary_supervision = True
            final_scores = tuple(int(value) for value in batch.transition.states[0].scores)
            _set_hand_supervision(
                critic_rows, hand_indices, kyoku, final_scores
            )
            completed += 1
        if final_scores is not None:
            order = tuple(sorted(range(4), key=lambda seat: (-final_scores[seat], seat)))
            placements = {seat: rank for rank, seat in enumerate(order)}
            from ..encoding.critic import RANK_ORDER_INDEX
            for row in critic_rows:
                row.terminal_placement = placements[int(row.binding.seat)]
                boundary = row.encoded.rank_boundary_features
                dealer = int(max(range(4), key=lambda value: boundary[4 + value]))
                relative_order = tuple((seat - dealer) % 4 for seat in order)
                row.rank_order_target = RANK_ORDER_INDEX[relative_order]
        if maximum is not None:
            examples = examples[:int(maximum)]
        return tuple(examples)
    finally:
        env.close()


@dataclass(slots=True)
class _ReplayContext:
    game_index: int
    environment_id: int
    kyokus: tuple[PhysicalKyoku, ...]
    maximum: int | None = None
    kyoku_index: int = 0
    event_index: int = 0
    examples: list = field(default_factory=list)
    critic_rows: list = field(default_factory=list)
    hand_indices: list = field(default_factory=list)
    final_scores: tuple[int, ...] | None = None

    @property
    def kyoku(self):
        return self.kyokus[self.kyoku_index]


def _encoding_batch(states, transition_id):
    states = tuple(sorted(states, key=lambda row: int(row.environment_id)))
    transition = SimpleNamespace(
        states=states,
        events=(),
        transition_id=int(transition_id),
    )
    decisions = tuple(decision for state in states for decision in state.action_spaces)
    return EnvBatch(transition, {}, decisions, ())


def _finish_replay_context(context):
    if context.final_scores is not None:
        order = tuple(sorted(
            range(4), key=lambda seat: (-context.final_scores[seat], seat)
        ))
        placements = {seat: rank for rank, seat in enumerate(order)}
        from ..encoding.critic import RANK_ORDER_INDEX
        for row in context.critic_rows:
            row.terminal_placement = placements[int(row.binding.seat)]
            boundary = row.encoded.rank_boundary_features
            dealer = int(max(range(4), key=lambda value: boundary[4 + value]))
            relative_order = tuple((seat - dealer) % 4 for seat in order)
            row.rank_order_target = RANK_ORDER_INDEX[relative_order]
    examples = tuple(context.examples)
    if context.maximum is not None:
        examples = examples[:int(context.maximum)]
    return examples


def replay_games(
    games, *, maximums=None, num_threads=None,
    include_public_snapshot=True,
):
    """Replay several complete games through one native batched environment."""
    import riichi

    games = tuple(games)
    if not games:
        return ()
    if maximums is None:
        maximums = (None,) * len(games)
    maximums = tuple(maximums)
    if len(maximums) != len(games):
        raise ValueError("one replay limit is required per game")
    contexts = []
    for game_index, (events, maximum) in enumerate(zip(games, maximums, strict=True)):
        kyokus = tuple(
            physicalize_kyoku(
                rows, environment_id=game_index, completed_kyoku=index,
            )
            for index, rows in enumerate(_split_kyoku(events))
        )
        if not kyokus:
            raise ValueError("no_kyoku")
        contexts.append(_ReplayContext(
            game_index, game_index, kyokus,
            None if maximum is None else int(maximum),
        ))

    threads = min(len(games), os.cpu_count() or 1) if num_threads is None \
        else max(1, int(num_threads))
    env = riichi.Env(
        len(games), master_seed=0, num_threads=threads,
        rules_profile=riichi.TENHOU_RULES_PROFILE, privileged=True,
    )
    adapter = EnvAdapter(env)
    event_cache = EventPrefixCache(max_entries=max(64, len(games) * 64))
    current_states = {}
    active = {context.environment_id: context for context in contexts}
    try:
        loaded = adapter.load_hanchan([
            _native_hanchan(riichi, context.kyoku.hanchan)
            for context in contexts
        ])
        current_states.update(
            (int(state.environment_id), state)
            for state in loaded.transition.states
        )
        transition_id = int(loaded.transition.transition_id)
        while active:
            states = [current_states[environment_id] for environment_id in active]
            batch = _encoding_batch(states, transition_id)
            encoded = encode_native_batch(
                batch,
                adapter.histories,
                event_cache=event_cache,
                include_public_snapshot=include_public_snapshot,
            ) if batch.action_spaces else ()
            encoded_by_binding = {
                (
                    int(row.binding.environment_id),
                    int(row.binding.frame_id),
                    int(row.binding.seat),
                ): row
                for row in encoded
            }
            source_groups = {}
            native_events = []
            for environment_id, context in active.items():
                state = current_states[environment_id]
                group = _event_group(
                    context.kyoku.events,
                    context.event_index,
                    int(state.phase),
                )
                if not group:
                    raise ValueError("replay kyoku has no event to apply")
                source_groups[environment_id] = group
                native_events.extend(_native_event(riichi, row) for row in group)

            result = adapter.apply_events(native_events)
            transition_id = int(result.transition.transition_id)
            current_states.update(
                (int(state.environment_id), state)
                for state in result.transition.states
            )
            emitted_by_environment = {environment_id: [] for environment_id in active}
            for event in result.transition.events:
                emitted_by_environment[int(event.environment_id)].append(event)
            selected = {
                (int(action.environment_id), int(action.frame_id), int(action.seat)):
                    int(action.candidate_index)
                for action in result.transition.applied_selections
            }
            completed = []
            for environment_id, context in tuple(active.items()):
                state = batch.transition.states[
                    next(index for index, row in enumerate(batch.transition.states)
                         if int(row.environment_id) == environment_id)
                ]
                decisions = tuple(state.action_spaces)
                emitted = tuple(emitted_by_environment[environment_id])
                source_tail = context.kyoku.events[
                    context.event_index:context.event_index + len(emitted)
                ]
                if len(source_tail) != len(emitted):
                    raise ValueError("native replay emitted events beyond the source record")
                for native_event, source_event in zip(emitted, source_tail, strict=True):
                    if native_event.kind_name != source_event["kind"]:
                        raise ValueError(
                            "native/source automatic event mismatch: "
                            f"native={native_event.kind_name}, "
                            f"source={source_event['kind']}"
                        )
                    for name in ("actor_seat", "target_seat"):
                        expected = source_event[name]
                        if expected is not None and getattr(native_event, name) != expected:
                            raise ValueError(
                                f"native/source {name} mismatch for "
                                f"{native_event.kind_name}"
                            )
                for decision in decisions:
                    key = (
                        environment_id,
                        int(decision.frame_id),
                        int(decision.seat),
                    )
                    row = encoded_by_binding[key]
                    native = selected[key]
                    target = next(
                        group_index
                        for group_index, members in enumerate(row.action_members)
                        if native in members
                    )
                    family = _FAMILY[int(decision.candidates[native].kind)]
                    critic = SimpleNamespace(
                        binding=row.binding,
                        encoded=row,
                        ppo_eligible=True,
                        rank_boundary_supervision=False,
                        rank_order_target=-1,
                        terminal_placement=-1,
                        hand_outcome_target=-1,
                        hand_score_delta=0,
                    )
                    context.examples.append(BCExample(row, target, family, critic))
                    context.critic_rows.append(critic)
                    context.hand_indices.append(len(context.critic_rows) - 1)
                context.event_index += len(emitted)
                if context.event_index >= len(context.kyoku.events):
                    if context.hand_indices:
                        context.critic_rows[
                            context.hand_indices[0]
                        ].rank_boundary_supervision = True
                    context.final_scores = tuple(
                        int(value) for value in current_states[environment_id].scores
                    )
                    _set_hand_supervision(
                        context.critic_rows,
                        context.hand_indices,
                        context.kyoku,
                        context.final_scores,
                    )
                    context.kyoku_index += 1
                    context.event_index = 0
                    context.hand_indices = []
                    if context.kyoku_index >= len(context.kyokus):
                        completed.append(environment_id)

            for environment_id in completed:
                active.pop(environment_id)
                current_states.pop(environment_id, None)
            continuing = [
                context for context in active.values()
                if context.event_index == 0
            ]
            if continuing:
                loaded = adapter.load_hanchan([
                    _native_hanchan(riichi, context.kyoku.hanchan)
                    for context in continuing
                ])
                transition_id = int(loaded.transition.transition_id)
                current_states.update(
                    (int(state.environment_id), state)
                    for state in loaded.transition.states
                )
        return tuple(
            _finish_replay_context(context)
            for context in sorted(contexts, key=lambda row: row.game_index)
        )
    finally:
        env.close()


class ArchiveCorpus:
    """Stream original archive members through deterministic native replay."""

    def __init__(self, archives):
        self.archives = tuple(Path(path) for path in archives)
        self._sources = {
            archive: zipfile.ZipFile(archive) for archive in self.archives
        }
        self.members = tuple(
            (archive, name)
            for archive in self.archives
            for name in self._names(self._sources[archive])
        )

    @staticmethod
    def _names(source):
        return sorted(name for name in source.namelist() if name.endswith(".mjson"))

    def close(self):
        for source in self._sources.values():
            source.close()
        self._sources.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()

    def read(self, member):
        archive, name = member
        source = self._sources.get(Path(archive))
        if source is None:
            raise ValueError(f"archive is outside this corpus: {archive}")
        payload = source.read(name)
        if payload.startswith(b"\x1f\x8b"):
            payload = gzip.decompress(payload)
        return tuple(json.loads(line) for line in payload.splitlines())

    @staticmethod
    def _compact(examples):
        compact = []
        for example in examples:
            encoded = replace(
                example.encoded,
                action_representatives=(),
                action_members=(),
                native_candidates=(),
            )
            critic = None if example.critic is None else SimpleNamespace(**{
                field: getattr(example.critic, field) for field in _CRITIC_FIELDS
            })
            compact.append(BCExample(encoded, example.target, example.family, critic))
        return tuple(compact)

    def examples(self, member, *, maximum=None):
        examples = self._compact(self.replay(self.read(member)))
        return examples if maximum is None else examples[:int(maximum)]

    def examples_many(self, requests, *, num_threads=None):
        """Read and batch-replay original archive members in request order."""
        requests = tuple(requests)
        results = [None] * len(requests)
        source_rows = []
        for index, (member, maximum) in enumerate(requests):
            try:
                events = self.read(member)
                _validate_game(events)
            except Exception as exc:
                results[index] = exc
                continue
            source_rows.append((index, member, maximum, events))

        def resolve(rows):
            if not rows:
                return
            try:
                replayed = replay_games(
                    [row[3] for row in rows], num_threads=num_threads,
                )
            except Exception:
                if len(rows) > 1:
                    middle = len(rows) // 2
                    resolve(rows[:middle])
                    resolve(rows[middle:])
                    return
                try:
                    replayed = (replay_game(rows[0][3]),)
                except Exception as single:
                    results[rows[0][0]] = single
                    return
            for row, examples in zip(rows, replayed, strict=True):
                index, _member, maximum, _events = row
                examples = self._compact(examples)
                results[index] = (
                    examples if maximum is None else examples[:int(maximum)]
                )

        resolve(source_rows)
        return tuple(results)

    def collect(self, members, *, maximum):
        rows = []
        rejected = Counter()
        accepted_members = []
        for member in members:
            if len(rows) >= int(maximum):
                break
            try:
                game_rows = self.examples(
                    member, maximum=int(maximum) - len(rows)
                )
                if not game_rows:
                    raise ValueError("no_queryable_decisions")
            except Exception as exc:
                rejected[f"{type(exc).__name__}:{exc}"] += 1
                continue
            accepted_members.append((str(member[0]), member[1], len(game_rows)))
            rows.extend(game_rows)
        return tuple(rows[:int(maximum)]), tuple(accepted_members), dict(rejected)

    @staticmethod
    def replay(events, *, maximum=None):
        _validate_game(events)
        return replay_game(events, maximum=maximum)
