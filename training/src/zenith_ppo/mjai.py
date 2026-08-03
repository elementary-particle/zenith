"""RiichiEnv observation bridge and RiichiLab MJAI WebSocket client."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
from pathlib import Path


LOGGER = logging.getLogger(__name__)
_ABSENT = 255
_EVENT_KINDS = {
    "start_game": 1,
    "start_kyoku": 2,
    "tsumo": 3,
    "dahai": 4,
    "chi": 5,
    "pon": 6,
    "daiminkan": 7,
    "ankan": 8,
    "kakan": 9,
    "dora": 10,
    "reach": 11,
    "reach_accepted": 12,
    "hora": 13,
    "ryukyoku": 14,
    "end_kyoku": 15,
    "end_game": 16,
}
_WINDS = {"E": 0, "S": 1, "W": 2, "N": 3}


@dataclass(frozen=True, slots=True)
class _Candidate:
    action: object
    row: dict
    riichi_tile: int | None = None


def _tile(value) -> int:
    if value in (None, "?"):
        return _ABSENT
    from riichienv.convert import mjai_to_tid

    return int(mjai_to_tid(str(value)))


def _event_identity(event: dict) -> tuple:
    """Semantic identity used to merge RiichiEnv's rolling event windows."""
    return (
        event.get("type"), event.get("actor"), event.get("target"),
        event.get("pai"), bool(event.get("tsumogiri", False)),
        tuple(event.get("consumed", ())), event.get("dora_marker"),
        event.get("bakaze"), event.get("kyoku"), event.get("honba"),
        event.get("kyotaku"), event.get("oya"),
        tuple(event.get("deltas", ())), bool(event.get("tsumo", False)),
    )


def _observation_events(observation) -> list[dict]:
    raw = observation.to_dict().get("events", ())
    result = []
    for value in raw:
        event = json.loads(value) if isinstance(value, str) else dict(value)
        if event.get("type") in _EVENT_KINDS:
            result.append(event)
    return result


@dataclass(slots=True)
class _InferenceContext:
    """Complete public current-kyoku context for stateful MJAI inference.

    RiichiEnv observations expose a rolling event window, not the complete
    history used by training. WebSocket events are therefore accumulated here;
    observation windows fill any gap around reconnects or direct ``act`` calls.
    """

    events: list[dict] = field(default_factory=list)
    start_boundary: dict | None = None
    current_scores: list[int] | None = None
    riichi_sticks: int = 0
    authoritative_stream: bool = False

    def reset(self) -> None:
        self.events.clear()
        self.start_boundary = None
        self.current_scores = None
        self.riichi_sticks = 0
        self.authoritative_stream = False

    @staticmethod
    def _start_identity(event: dict) -> tuple:
        return tuple(event.get(name) for name in (
            "bakaze", "kyoku", "honba", "kyotaku", "oya",
        ))

    def _begin_kyoku(self, event: dict) -> None:
        scores = tuple(int(value) for value in event.get("scores", ()))
        self.events.clear()
        self.start_boundary = {
            "scores": scores,
            "round_wind": _WINDS.get(str(event.get("bakaze", "E")), 0),
            "hand_number": int(event.get("kyoku", 1)) - 1,
            "dealer": int(event.get("oya", 0)),
            "honba": int(event.get("honba", 0)),
            "riichi_deposits": int(event.get("kyotaku", 0)),
        }
        self.current_scores = list(scores) if len(scores) == 4 else None
        self.riichi_sticks = int(event.get("kyotaku", 0))

    def _append(self, source: dict) -> None:
        event = dict(source)
        name = event.get("type")
        if name == "start_game":
            self.reset()
            return
        if name == "start_kyoku":
            self._begin_kyoku(event)
        elif self.start_boundary is None:
            # A reconnect without a start boundary can still use the exact
            # current-state suffix; rank-boundary and progress fall back to
            # the observation until the next start_kyoku.
            pass
        if name == "reach_accepted":
            actor = int(event.get("actor", _ABSENT))
            explicit = event.get("scores")
            if explicit is not None and len(explicit) == 4:
                self.current_scores = list(map(int, explicit))
            elif self.current_scores is not None and 0 <= actor < 4:
                self.current_scores[actor] -= 1_000
            self.riichi_sticks = int(event.get(
                "kyotaku", self.riichi_sticks + 1,
            ))
            if self.current_scores is not None:
                event["scores"] = tuple(self.current_scores)
            event["kyotaku"] = self.riichi_sticks
        elif name in {"hora", "ryukyoku"}:
            deltas = tuple(int(value) for value in event.get("deltas", ()))
            if self.current_scores is not None and len(deltas) == 4:
                self.current_scores = [
                    score + delta
                    for score, delta in zip(self.current_scores, deltas, strict=True)
                ]
        self.events.append(event)

    def observe_event(self, event: dict) -> None:
        if event.get("type") in _EVENT_KINDS:
            self._append(event)
            # start_game alone is not proof that the connection supplied the
            # complete current-kyoku prefix.  Once start_kyoku has arrived,
            # subsequent top-level events are authoritative.
            self.authoritative_stream = self.complete

    def synchronize(self, observation) -> None:
        """Merge one rolling observation window into the complete prefix."""
        if self.authoritative_stream and self.complete:
            return
        incoming = _observation_events(observation)
        if not incoming:
            return
        starts = [
            index for index, event in enumerate(incoming)
            if event.get("type") == "start_kyoku"
        ]
        if starts:
            incoming = incoming[starts[-1]:]
            current_start = self.events[0] if self.events else None
            if (
                current_start is None
                or current_start.get("type") != "start_kyoku"
                or self._start_identity(current_start)
                    != self._start_identity(incoming[0])
            ):
                self._begin_kyoku(incoming[0])
                self.events.append(dict(incoming[0]))
                for event in incoming[1:]:
                    self._append(event)
                return
            common = 0
            limit = min(len(self.events), len(incoming))
            while common < limit and _event_identity(self.events[common]) \
                    == _event_identity(incoming[common]):
                common += 1
            if common == len(incoming):
                return
            if common == len(self.events):
                for event in incoming[common:]:
                    self._append(event)
                return

        identities = [_event_identity(event) for event in incoming]
        stored = [_event_identity(event) for event in self.events]
        overlap = 0
        for size in range(min(len(stored), len(identities)), 0, -1):
            if stored[-size:] == identities[:size]:
                overlap = size
                break
        for event in incoming[overlap:]:
            self._append(event)

    @property
    def complete(self) -> bool:
        return bool(
            self.start_boundary is not None
            and self.events
            and self.events[0].get("type") == "start_kyoku"
        )

    @property
    def live_wall_remaining(self) -> int | None:
        if not self.complete:
            return None
        return max(0, 70 - sum(
            event.get("type") == "tsumo" for event in self.events
        ))

    def boundary_frame(self, current: dict) -> dict:
        result = dict(current)
        if self.start_boundary is not None:
            result.update(self.start_boundary)
        return result

    def public_rivers(self) -> tuple[dict, ...] | None:
        if not self.complete:
            return None
        rivers = [[] for _ in range(4)]
        pending_riichi = set()
        sequence = 0
        for event in self.events:
            name = event.get("type")
            actor = int(event.get("actor", _ABSENT))
            if name == "reach" and 0 <= actor < 4:
                pending_riichi.add(actor)
            elif name == "dahai" and 0 <= actor < 4:
                rivers[actor].append({
                    "seat": actor,
                    "tile": _tile(event.get("pai")),
                    "sequence": sequence,
                    "riichi_declaration": actor in pending_riichi,
                    "called": False,
                    "tsumogiri": bool(event.get("tsumogiri", False)),
                })
                pending_riichi.discard(actor)
                sequence += 1
            elif name in {"chi", "pon", "daiminkan"}:
                target = int(event.get("target", _ABSENT))
                if 0 <= target < 4:
                    for river in reversed(rivers[target]):
                        if not river["called"]:
                            river["called"] = True
                            break
        return tuple(row for seat in rivers for row in seat)


def _event_rows(observation, events=None) -> tuple[list[dict], list[dict]]:
    events = (
        _observation_events(observation)
        if events is None else [dict(value) for value in events]
    )
    rows = []
    for event in events:
        name = event.get("type")
        kind = _EVENT_KINDS.get(name)
        if kind is None:
            # RiichiLab explicitly permits future informational event types.
            continue
        actor = int(event.get("actor", _ABSENT))
        target = int(event.get("target", _ABSENT))
        # Native events use ABSENT for unused tile arguments.  Zero is the
        # first physical 1m and changes chi shape/detail factorization.
        args = [_ABSENT, _ABSENT, _ABSENT, _ABSENT]
        if name == "start_kyoku":
            args = [
                _WINDS.get(str(event.get("bakaze", "E")), 0),
                int(event.get("kyoku", 1)),
                int(event.get("honba", 0)),
                int(event.get("oya", 0)),
            ]
            visibility = 0
        elif name == "tsumo":
            args[0] = _tile(event.get("pai"))
            visibility = 1 << actor if 0 <= actor < 4 else 0
        else:
            visibility = 0b1111
            if name == "dora":
                args[0] = _tile(event.get("dora_marker"))
            elif name in {"dahai", "chi", "pon", "daiminkan", "kakan", "hora"}:
                args[0] = _tile(event.get("pai"))
            if name == "dahai":
                args[1] = int(bool(event.get("tsumogiri", False)))
            if name in {"chi", "pon", "daiminkan"}:
                for index, value in enumerate(event.get("consumed", ())[:3], 1):
                    args[index] = _tile(value)
            elif name == "ankan":
                for index, value in enumerate(event.get("consumed", ())[:4]):
                    args[index] = _tile(value)
            elif name == "reach_accepted":
                scores = event.get("scores")
                if scores is not None and 0 <= actor < len(scores):
                    args[0] = int(scores[actor])
                elif 0 <= actor < len(observation.scores):
                    args[0] = int(observation.scores[actor])
                args[1] = int(event.get("kyotaku", observation.riichi_sticks))
        rows.append({
            "environment_id": 0,
            "episode_generation": 1,
            "sequence": len(rows),
            "kind": kind,
            "actor_seat": actor,
            "target_seat": target,
            "visibility_mask": visibility,
            "args": tuple(args),
            "payload": b"",
        })
    return rows, events


def _last_source(events) -> int:
    for event in reversed(events):
        if event.get("type") in {"dahai", "ankan", "kakan"}:
            return int(event.get("actor", _ABSENT))
    return _ABSENT


def _decision_flags(observation, events) -> int:
    seat = int(observation.player_id)
    declared = False
    accepted = False
    ippatsu = False
    for event in events:
        name = event.get("type")
        actor = int(event.get("actor", _ABSENT))
        if actor == seat and name == "reach":
            declared = True
        elif actor == seat and name == "reach_accepted":
            declared = False
            accepted = True
            ippatsu = True
        if name in {"chi", "pon", "daiminkan", "ankan", "kakan"}:
            ippatsu = False
        elif actor == seat and name == "dahai" and accepted:
            ippatsu = False
    accepted = accepted or bool(observation.riichi_declared[seat])
    return int(declared and not accepted) | (int(accepted) << 1) | (int(ippatsu) << 2)


def _public_seat_flags(observation, events) -> tuple[int, int, int, int]:
    flags = []
    for seat in range(4):
        declared = accepted = ippatsu = False
        for event in events:
            name = event.get("type")
            actor = int(event.get("actor", _ABSENT))
            if actor == seat and name == "reach":
                declared = True
            elif actor == seat and name == "reach_accepted":
                declared = False
                accepted = True
                ippatsu = True
            if name in {"chi", "pon", "daiminkan", "ankan", "kakan"}:
                ippatsu = False
            elif actor == seat and name == "dahai" and accepted:
                ippatsu = False
        accepted = accepted or bool(observation.riichi_declared[seat])
        flags.append(
            int(declared and not accepted)
            | (int(accepted) << 1)
            | (int(ippatsu) << 2)
        )
    return tuple(flags)


def _public_rivers(observation) -> tuple[dict, ...]:
    """Use RiichiEnv's current rivers; called discards are already removed."""
    rivers = []
    sequence = 0
    for seat, discards in enumerate(observation.discards):
        tsumogiri = tuple(observation.tsumogiri_flags[seat])
        riichi_tile = observation.riichi_sutehais[seat]
        for index, tile in enumerate(discards):
            rivers.append({
                "seat": seat,
                "tile": int(tile),
                "sequence": sequence,
                "riichi_declaration": (
                    riichi_tile is not None and int(tile) == int(riichi_tile)
                ),
                "called": False,
                "tsumogiri": bool(tsumogiri[index])
                    if index < len(tsumogiri) else False,
            })
            sequence += 1
    return tuple(rivers)


def _public_melds(observation) -> tuple[dict, ...]:
    result = []
    for seat, melds in enumerate(observation.melds):
        for meld in melds:
            result.append({
                "seat": seat,
                "kind": int(meld.meld_type) + 1,
                # Core snapshots canonicalize meld members by physical tile.
                # MJAI observations retain called-tile-first ordering.
                "tiles": tuple(sorted(int(tile) for tile in meld.tiles)),
            })
    return tuple(result)


def _action_row(action, observation, events) -> dict:
    action_type = int(action.action_type)
    kinds = {
        0: 1,   # discard
        1: 3,   # chi
        2: 4,   # pon
        3: 5,   # daiminkan
        4: 8,   # ron
        6: 9,   # tsumo
        7: 0,   # pass
        8: 6,   # ankan
        9: 7,   # kakan
        10: 10, # kyuushu kyuuhai
    }
    if action_type not in kinds:
        raise ValueError(f"unsupported RiichiEnv action type {action_type}")
    tile = None if action.tile is None else int(action.tile)
    consumed = tuple(int(value) for value in action.consume_tiles)
    source = _last_source(events) if action_type in {1, 2, 3, 4} else _ABSENT
    if action_type in {1, 2, 3}:
        tiles = tuple(sorted((*consumed, tile)))
    elif action_type == 8:
        tiles = tuple(sorted(consumed))
    elif tile is not None:
        tiles = (tile,)
    else:
        tiles = ()
    aux = 1 if action_type == 10 else 0
    if action_type == 9 and tile is not None:
        tile_type = tile // 4
        melds = observation.melds[int(observation.player_id)]
        aux = next((
            index for index, meld in enumerate(melds)
            if int(meld.meld_type) == 1
            and meld.tiles and int(meld.tiles[0]) // 4 == tile_type
        ), 0)
    return {
        "kind": kinds[action_type],
        "primary_tile_type": _ABSENT if not tiles else (
            tile // 4 if tile is not None else tiles[0] // 4
        ),
        "source_seat": source,
        "tiles": tiles,
        "aux": aux,
        "flags": 0,
    }


def _candidates(observation, events) -> tuple[_Candidate, ...]:
    from riichienv import check_riichi_candidates

    legal = tuple(observation.legal_actions())
    reach = next((action for action in legal if int(action.action_type) == 5), None)
    candidates = [
        _Candidate(action, _action_row(action, observation, events))
        for action in legal if int(action.action_type) != 5
    ]
    # Native training descriptors are ordered by semantic ActionKind. Preserve
    # that order because action memory is sequence-aware (RiichiEnv commonly
    # returns call before pass on reaction frames).
    candidates.sort(key=lambda candidate: int(candidate.row["kind"]))
    if reach is not None:
        legal_discards = [action for action in legal if int(action.action_type) == 0]
        for tile in check_riichi_candidates(list(observation.hand)):
            tile = int(tile)
            representative = next((
                action for action in legal_discards
                if int(action.tile) == tile
            ), None)
            if representative is None:
                representative = next((
                    action for action in legal_discards
                    if int(action.tile) // 4 == tile // 4
                    and (int(action.tile) in (16, 52, 88)) == (tile in (16, 52, 88))
                ), None)
            if representative is None:
                raise ValueError("RiichiEnv returned a non-discardable riichi candidate")
            row = dict(_action_row(representative, observation, events), kind=2)
            candidates.append(_Candidate(reach, row, riichi_tile=int(representative.tile)))
    if not candidates:
        raise ValueError("request_action observation contains no legal actions")
    return tuple(candidates)


def _decision_phase(observation, events) -> int:
    """Recover the native HandPhase represented by one legal-action set."""
    action_types = {int(action.action_type) for action in observation.legal_actions()}
    if 7 in action_types:  # pass exists only on discard/kan reaction frames
        latest = next((
            event.get("type") for event in reversed(events)
            if event.get("type") not in {"tsumo", "dora"}
        ), None)
        return 3 if latest in {"ankan", "kakan"} else 2
    # ReplacementTurn is an automatic core transition.  Once the rinshan tile
    # has been drawn and legal actions are exposed, the native decision is a
    # regular SelfTurnDecision again.
    return 1


def encode_observation(observation, *, context: _InferenceContext | None = None):
    """Encode one RiichiEnv observation with Zenith's public actor schema."""
    import numpy as np

    from .encoding.actions import encode_actions
    from .encoding.critic import encode_rank_boundary
    from .encoding.events import encode_history
    from .encoding.packing import EncodedActionSpace
    from .encoding.schema import Segment, TokenKind, numeric_features
    from .encoding.state import (
        encode_match_state_factors,
        encode_tactical_state_factors,
    )
    from .types import ActionSpaceBinding

    observer = int(observation.player_id)
    if context is not None:
        context.synchronize(observation)
    rows, events = _event_rows(
        observation,
        context.events if context is not None and context.events else None,
    )
    history = encode_history(rows, observer=observer, generation=1)
    event_factors = np.asarray(
        [token.categorical() for token in history], dtype=np.uint8
    ).reshape(-1, 10)
    event_numeric = np.asarray(
        [numeric_features(token) for token in history], dtype=np.float32
    ).reshape(-1, 8)
    current_kyoku = events[next((
        index for index in range(len(events) - 1, -1, -1)
        if events[index].get("type") == "start_kyoku"
    ), 0):]
    rolling_wall = max(
        0, 70 - sum(event.get("type") == "tsumo" for event in current_kyoku)
    )
    context_wall = context.live_wall_remaining if context is not None else None
    context_rivers = context.public_rivers() if context is not None else None
    frame = {
        "scores": tuple(int(value) for value in observation.scores),
        "round_wind": int(observation.round_wind),
        "hand_number": int(observation.kyoku_index),
        "dealer": int(observation.oya),
        "honba": int(observation.honba),
        "riichi_deposits": int(observation.riichi_sticks),
        "live_wall_remaining": (
            rolling_wall if context_wall is None else context_wall
        ),
        "dora_indicators": tuple(int(value) for value in observation.dora_indicators),
        "seat_flags": _public_seat_flags(observation, current_kyoku),
        "rivers": (
            _public_rivers(observation)
            if context_rivers is None else context_rivers
        ),
        "melds": _public_melds(observation),
    }
    counts = np.bincount(
        np.asarray(observation.hand, dtype=np.int64) // 4, minlength=34
    ).astype(np.uint8)
    actor = {
        "concealed_counts": tuple(int(value) for value in counts),
        "flags": _decision_flags(observation, current_kyoku),
    }
    match_factors, match_numeric = encode_match_state_factors(frame, observer=observer)
    tactical_factors, tactical_numeric = encode_tactical_state_factors(
        frame, actor, observer=observer, include_public_snapshot=True
    )
    queries = np.asarray([
        (Segment.MATCH_SUMMARY, TokenKind.QUERY, 2, 1, 0, 0, 0, 0, 0, 0),
        (Segment.KYOKU_SUMMARY, TokenKind.QUERY, 3, 1, 0, 0, 0, 0, 0, 0),
        (Segment.ACTOR_QUERY, TokenKind.QUERY, 1, 1, 0, 0, 0, 0, 0, 0),
    ], dtype=np.uint8)
    queries[2, 8] = _decision_phase(observation, current_kyoku)
    queries[2, 9] = 1
    zero = np.zeros((1, 8), dtype=np.float32)
    actor_query = len(match_factors) + 1 + len(event_factors) + len(tactical_factors) + 1
    candidates = _candidates(observation, current_kyoku)
    actions = encode_actions([candidate.row for candidate in candidates], observer=observer)
    return EncodedActionSpace(
        binding=ActionSpaceBinding(0, 1, 1, observer),
        token_factors=np.concatenate((
            match_factors, queries[:1], event_factors,
            tactical_factors, queries[1:2], queries[2:3],
        )),
        token_numeric=np.concatenate((
            match_numeric, zero, event_numeric,
            tactical_numeric, zero, zero,
        )),
        actor_query_index=actor_query,
        rank_boundary_features=encode_rank_boundary(
            frame if context is None else context.boundary_frame(frame)
        ),
        decision_seat=(observer - int(observation.oya)) % 4,
        action_factors=np.asarray(
            actions.factors, dtype=np.uint8
        ).reshape(-1, 15),
        action_representatives=actions.representatives,
        action_members=actions.members,
        native_candidates=candidates,
    )


class CheckpointAgent:
    """Greedy public-policy inference over RiichiEnv observations."""

    def __init__(self, model, *, device="cpu", backend="sdpa", use_bf16=False):
        self.model = model
        self.device = str(device)
        self.backend = str(backend)
        self.use_bf16 = bool(use_bf16 and self.device.startswith("cuda"))
        self.pending_riichi_tile: int | None = None
        self.encoding_context = _InferenceContext()

    def reset(self) -> None:
        self.pending_riichi_tile = None
        self.encoding_context.reset()

    def cancel_pending_action(self) -> None:
        self.pending_riichi_tile = None

    def observe_event(self, event: dict) -> None:
        self.encoding_context.observe_event(event)

    def act(self, observation):
        import torch

        legal = tuple(observation.legal_actions())
        if self.pending_riichi_tile is not None:
            expected = self.pending_riichi_tile
            self.pending_riichi_tile = None
            action = next((
                value for value in legal
                if int(value.action_type) == 0 and int(value.tile) == expected
            ), None)
            if action is None:
                raise RuntimeError("riichi discard request did not offer the selected tile")
            return action

        from .encoding.packing import model_batch

        encoded = encode_observation(
            observation, context=self.encoding_context,
        )
        inputs = model_batch([encoded], device=self.device, backend=self.backend)
        self.model.eval()
        with torch.no_grad(), torch.autocast(
            device_type=torch.device(self.device).type,
            dtype=torch.bfloat16,
            enabled=self.use_bf16,
        ):
            output = self.model.forward_actor(**inputs)
        group = int(output.log_probabilities.argmax().item())
        candidate = encoded.native_candidates[encoded.action_representatives[group]]
        if candidate.riichi_tile is not None:
            self.pending_riichi_tile = candidate.riichi_tile
        return candidate.action


def load_checkpoint_agent(config, checkpoint, *, device="cpu", backend="sdpa",
                          use_bf16=False) -> CheckpointAgent:
    """Build an inference-only agent from a durable Zenith checkpoint."""
    from .checkpoint import resolve_latest, restore
    from .model.actor_critic import ActorCritic

    path = Path(checkpoint)
    if (path / "latest").is_file():
        path = resolve_latest(path)
    restored = restore(path)
    model_config = dict(config.values["model"])
    model_config["context_tokens"] = int(config.values["encoding"]["context_tokens"])
    model = ActorCritic(model_config)
    actual_architecture = restored.get("state", {}).get("architecture")
    compatible_architectures = {
        "shared-shape-rank-v-bc-v1",
        "shared-shape-emagnet-current-kyoku-ppo-v1",
    }
    if actual_architecture not in compatible_architectures:
        raise RuntimeError(
            f"checkpoint architecture {actual_architecture!r} does not match "
            f"a compatible configuration architecture "
            f"{sorted(compatible_architectures)!r}"
        )
    model.load_state_dict(restored["model"])
    model.to(device).eval()
    return CheckpointAgent(
        model, device=device, backend=backend, use_bf16=use_bf16
    )


def action_response(action, observation) -> dict:
    response = json.loads(action.to_mjai())
    if response.get("type") == "dahai":
        response["tsumogiri"] = (
            observation.drawn_tile is not None
            and action.tile is not None
            and int(action.tile) == int(observation.drawn_tile)
        )
    if observation.select_action_from_mjai(response) is None:
        raise RuntimeError("agent produced an action rejected by its RiichiEnv observation")
    return response


def _possible_match(response: dict, option: dict) -> bool:
    if response.get("type") != option.get("type"):
        return False
    name = response.get("type")
    if name in {"dahai", "chi", "pon", "daiminkan", "kakan"} \
            and response.get("pai") != option.get("pai"):
        return False
    if name in {"chi", "pon", "daiminkan", "ankan"}:
        return sorted(response.get("consumed", ())) == sorted(option.get("consumed", ()))
    return True


def validate_possible_action(response: dict, possible_actions) -> None:
    if possible_actions is None:
        return
    if not any(_possible_match(response, dict(option)) for option in possible_actions):
        raise RuntimeError("agent response does not match request_action.possible_actions")


async def play_connection(websocket, agent, *, observation_type=None):
    """Play one server-driven RiichiLab connection."""
    if observation_type is None:
        from riichienv import Observation
        observation_type = Observation
    async for payload in websocket:
        if isinstance(payload, bytes):
            continue
        message = json.loads(payload)
        if "error" in message:
            raise RuntimeError(str(message["error"]))
        name = message.get("type")
        if name == "start_game":
            agent.reset()
        observer = getattr(agent, "observe_event", None)
        if observer is not None and name in _EVENT_KINDS:
            observer(message)
        if name == "request_action":
            observation = observation_type.deserialize_from_base64(
                message["observation"]
            )
            action = agent.act(observation)
            response = action_response(action, observation)
            validate_possible_action(response, message.get("possible_actions"))
            if "request_id" in message:
                response["request_id"] = message["request_id"]
            await websocket.send(json.dumps(response, separators=(",", ":")))
        elif name == "action_ack":
            status = message.get("status")
            if status in {"rejected", "unparseable"}:
                raise RuntimeError(f"server rejected action: {message}")
            if status in {"stale", "defaulted"}:
                agent.cancel_pending_action()
        elif name == "end_game":
            return message
    raise ConnectionError("RiichiLab connection closed before end_game")


async def connect(url: str, token: str, agent):
    import websockets

    headers = {"Authorization": f"Bearer {token}"}
    async with websockets.connect(url, additional_headers=headers) as websocket:
        LOGGER.info("connected to %s", url)
        return await play_connection(websocket, agent)


async def run_matches(
    url: str,
    token: str,
    agent,
    *,
    max_games: int | None = None,
    retry_initial_seconds: float = 1.0,
    retry_max_seconds: float = 30.0,
    connect_one=None,
    sleep=None,
    stop_requested=None,
) -> int:
    """Keep reconnecting until stopped or ``max_games`` complete.

    A stop requested while connected takes effect only after the active match
    reaches ``end_game``.
    """
    import asyncio

    if max_games is not None and int(max_games) <= 0:
        raise ValueError("max_games must be positive or None")
    if retry_initial_seconds < 0 or retry_max_seconds < retry_initial_seconds:
        raise ValueError("invalid reconnect delay bounds")
    connect_one = connect if connect_one is None else connect_one
    sleep = asyncio.sleep if sleep is None else sleep
    stop_requested = (lambda: False) if stop_requested is None else stop_requested
    completed = 0
    delay = float(retry_initial_seconds)
    while (
        not stop_requested()
        and (max_games is None or completed < int(max_games))
    ):
        try:
            result = await connect_one(url, token, agent)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if stop_requested():
                break
            LOGGER.warning(
                "connection failed; retrying in %.1fs: %s", delay, exc
            )
            await sleep(delay)
            delay = min(float(retry_max_seconds), max(delay * 2.0, 0.001))
            continue
        completed += 1
        delay = float(retry_initial_seconds)
        LOGGER.info(
            "match %d completed: scores=%s",
            completed,
            result.get("scores") if isinstance(result, dict) else None,
        )
    return completed
