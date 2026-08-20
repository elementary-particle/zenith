"""Akagi JSONL/WebSocket adapter for the Zenith MJAI policy.

Akagi supplies public MJAI events rather than RiichiLab's serialized
``riichienv.Observation``.  This module maintains the small amount of live
state needed to materialize that observation and deliberately delegates model
encoding and action selection to :mod:`zenith_ppo.mjai`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations, product
import json
import logging

from .mjai import action_response


LOGGER = logging.getLogger(__name__)
_WINDS = {"E": 0, "S": 1, "W": 2, "N": 3}
_TERMINAL_TYPES = frozenset((0, 8, 9, 17, 18, 26, *range(27, 34)))


class AkagiProtocolError(ValueError):
    """The Akagi client supplied an invalid or unsupported event stream."""


def _tile_id(pai: str) -> int:
    from riichienv.convert import mjai_to_tid

    return int(mjai_to_tid(pai))


def _tile_name(tile: int) -> str:
    from riichienv.convert import tid_to_mjai

    return str(tid_to_mjai(int(tile)))


def _is_red(tile: int) -> bool:
    return int(tile) in (16, 52, 88)


def _physical_candidates(pai: str) -> tuple[int, ...]:
    canonical = _tile_id(pai)
    if pai.endswith("r"):
        return (canonical,)
    tile_type = canonical // 4
    values = tuple(range(tile_type * 4, tile_type * 4 + 4))
    if tile_type in (4, 13, 22):
        values = tuple(value for value in values if not _is_red(value))
    return values


def _take_named(hand: list[int], pai: str) -> int:
    candidates = set(_physical_candidates(pai))
    for index, tile in enumerate(hand):
        if tile in candidates:
            return hand.pop(index)
    raise AkagiProtocolError(f"own hand does not contain {pai}")


def _allocate_hand(pais) -> list[int]:
    result: list[int] = []
    for pai in pais:
        if pai == "?":
            continue
        used = set(result)
        tile = next(
            (value for value in _physical_candidates(str(pai)) if value not in used),
            None,
        )
        if tile is None:
            raise AkagiProtocolError(f"too many copies of tile {pai} in own hand")
        result.append(tile)
    return sorted(result)


def _public_tile(pai: str) -> int:
    """Return a stable physical representative for a public tile."""
    return _physical_candidates(pai)[0]


def _meld_kind(name: str):
    from riichienv import MeldType

    return {
        "chi": MeldType.Chi,
        "pon": MeldType.Pon,
        "daiminkan": MeldType.Daiminkan,
        "ankan": MeldType.Ankan,
        "kakan": MeldType.Kakan,
    }[name]


@dataclass(slots=True)
class LiveMjaiState:
    """Public four-player state reconstructed from Akagi's MJAI stream."""

    player_id: int | None = None
    names: list[str] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    hand: list[int] = field(default_factory=list)
    melds: list[list[object]] = field(default_factory=lambda: [[] for _ in range(4)])
    discards: list[list[int]] = field(default_factory=lambda: [[] for _ in range(4)])
    discard_tsumogiri: list[list[bool]] = field(
        default_factory=lambda: [[] for _ in range(4)]
    )
    dora_indicators: list[int] = field(default_factory=list)
    scores: list[int] = field(default_factory=lambda: [25_000] * 4)
    riichi_declared: list[bool] = field(default_factory=lambda: [False] * 4)
    riichi_sutehais: list[int | None] = field(default_factory=lambda: [None] * 4)
    last_tedashis: list[int | None] = field(default_factory=lambda: [None] * 4)
    pending_reach: set[int] = field(default_factory=set)
    round_wind: int = 0
    kyoku_index: int = 0
    oya: int = 0
    honba: int = 0
    riichi_sticks: int = 0
    drawn_tile: int | None = None
    unknown_own_tiles: int = 0
    own_draw_hidden: bool = False
    last_discard: tuple[int, int] | None = None
    tsumo_count: int = 0
    call_count: int = 0
    own_discard_count: int = 0
    forbidden_discard_types: set[int] = field(default_factory=set)
    active: bool = False

    def reset(self) -> None:
        player_id = self.player_id
        names = self.names
        fresh = type(self)(player_id=player_id, names=names)
        for name in self.__dataclass_fields__:
            setattr(self, name, getattr(fresh, name))

    def consume(self, event: dict) -> None:
        if not isinstance(event, dict) or not isinstance(event.get("type"), str):
            raise AkagiProtocolError("each MJAI event must be an object with a type")
        name = event["type"]
        if name == "start_game":
            players = int(event.get("num_players", 4))
            if players != 4:
                raise AkagiProtocolError("Zenith currently supports four-player games only")
            if event.get("id") is None:
                raise AkagiProtocolError("start_game is missing the bot seat id")
            self.player_id = int(event["id"])
            if self.player_id not in range(4):
                raise AkagiProtocolError("start_game contains an invalid bot seat")
            self.names = [str(value) for value in event.get("names", ())]
            self.reset()
            self.events.append(dict(event))
            return
        if self.player_id is None:
            raise AkagiProtocolError("start_game must precede all other events")
        self.events.append(dict(event))
        if name == "start_kyoku":
            self._start_kyoku(event)
        elif name == "tsumo":
            self._tsumo(event)
        elif name == "dahai":
            self._dahai(event)
        elif name in {"chi", "pon", "daiminkan"}:
            self._open_call(event)
        elif name == "ankan":
            self._ankan(event)
        elif name == "kakan":
            self._kakan(event)
        elif name == "dora":
            self.dora_indicators.append(_public_tile(str(event["dora_marker"])))
        elif name == "reach":
            self.pending_reach.add(int(event["actor"]))
        elif name == "reach_accepted":
            self._reach_accepted(event)
        elif name in {"hora", "ryukyoku"}:
            self._settlement(event)
        elif name == "end_kyoku":
            self.active = False
            self.drawn_tile = None
            self.unknown_own_tiles = 0
            self.own_draw_hidden = False
        elif name == "end_game":
            self.active = False

    def _start_kyoku(self, event: dict) -> None:
        tehais = event.get("tehais")
        if not isinstance(tehais, list) or len(tehais) != 4:
            raise AkagiProtocolError("start_kyoku must contain four tehais")
        scores = list(map(int, event.get("scores", ())))
        if len(scores) != 4:
            raise AkagiProtocolError("start_kyoku must contain four scores")
        self.hand = _allocate_hand(tehais[self.player_id])
        if len(self.hand) != 13:
            raise AkagiProtocolError("the bot's start_kyoku hand must contain 13 tiles")
        self.melds = [[] for _ in range(4)]
        self.discards = [[] for _ in range(4)]
        self.discard_tsumogiri = [[] for _ in range(4)]
        self.dora_indicators = [_public_tile(str(event["dora_marker"]))]
        self.scores = scores
        self.riichi_declared = [False] * 4
        self.riichi_sutehais = [None] * 4
        self.last_tedashis = [None] * 4
        self.pending_reach.clear()
        self.round_wind = _WINDS.get(str(event.get("bakaze", "E")), 0)
        self.kyoku_index = int(event.get("kyoku", 1)) - 1
        self.oya = int(event.get("oya", 0))
        self.honba = int(event.get("honba", 0))
        self.riichi_sticks = int(event.get("kyotaku", 0))
        self.drawn_tile = None
        self.unknown_own_tiles = 0
        self.own_draw_hidden = False
        self.last_discard = None
        self.tsumo_count = 0
        self.call_count = 0
        self.own_discard_count = 0
        self.forbidden_discard_types.clear()
        self.active = True

    def _tsumo(self, event: dict) -> None:
        actor = int(event["actor"])
        self.tsumo_count += 1
        self.last_discard = None
        if actor == self.player_id:
            pai = str(event["pai"])
            if pai == "?":
                # Akagi normally exposes the local player's draw, but its
                # Majsoul bridge maps an absent/empty ActionDealTile tile to
                # "?".  We cannot advise on an unknown 14th tile.  Keep the
                # authoritative 13-tile concealed hand and wait for the echo:
                # a tsumogiri reveals the missing tile without changing it.
                self.drawn_tile = None
                self.unknown_own_tiles += 1
                self.own_draw_hidden = True
                LOGGER.info(
                    "Akagi hid the bot's own draw; skipping this decision "
                    "and waiting for the discard echo"
                )
                return
            used = set(self.hand)
            tile = next(
                (value for value in _physical_candidates(pai) if value not in used),
                None,
            )
            if tile is None:
                raise AkagiProtocolError(f"cannot allocate own drawn tile {pai}")
            self.hand.append(tile)
            self.hand.sort()
            self.drawn_tile = tile
            self.own_draw_hidden = False
            self.forbidden_discard_types.clear()

    def _dahai(self, event: dict) -> None:
        actor = int(event["actor"])
        pai = str(event["pai"])
        tsumogiri = bool(event.get("tsumogiri", False))
        tile = _public_tile(pai)
        if actor == self.player_id:
            if self.own_draw_hidden and tsumogiri:
                # The missing draw immediately left the hand, so the original
                # known concealed hand remains exact.
                self.unknown_own_tiles -= 1
            else:
                tile = self._take_owned_tile(pai)
            self.drawn_tile = None
            self.own_draw_hidden = False
            self.own_discard_count += 1
            self.forbidden_discard_types.clear()
        self.discards[actor].append(tile)
        self.discard_tsumogiri[actor].append(tsumogiri)
        if not tsumogiri:
            self.last_tedashis[actor] = tile
        if actor in self.pending_reach:
            self.riichi_sutehais[actor] = tile
            self.pending_reach.discard(actor)
        self.last_discard = (actor, tile)

    def _open_call(self, event: dict) -> None:
        from riichienv import Meld

        name = str(event["type"])
        actor = int(event["actor"])
        target = int(event["target"])
        consumed_names = [str(value) for value in event.get("consumed", ())]
        if actor == self.player_id:
            consumed = [self._take_owned_tile(value) for value in consumed_names]
            self.drawn_tile = None
            self.own_draw_hidden = False
        else:
            consumed = [_public_tile(value) for value in consumed_names]
        called = _public_tile(str(event["pai"]))
        self._mark_called_discard(target)
        self.melds[actor].append(
            Meld(_meld_kind(name), [called, *consumed], True, target, called)
        )
        if actor == self.player_id:
            called_type = called // 4
            forbidden = {called_type}
            if name == "chi":
                sequence = sorted([called_type, *(tile // 4 for tile in consumed)])
                suit_start = (called_type // 9) * 9
                suit_end = suit_start + 8
                if called_type == sequence[0] and sequence[-1] < suit_end:
                    forbidden.add(sequence[-1] + 1)
                elif called_type == sequence[-1] and sequence[0] > suit_start:
                    forbidden.add(sequence[0] - 1)
            self.forbidden_discard_types = forbidden
        self.call_count += 1
        self.last_discard = None

    def _ankan(self, event: dict) -> None:
        from riichienv import Meld

        actor = int(event["actor"])
        names = [str(value) for value in event.get("consumed", ())]
        if actor == self.player_id:
            tiles = [self._take_owned_tile(value) for value in names]
            self.drawn_tile = None
            self.own_draw_hidden = False
        else:
            tiles = [_public_tile(value) for value in names]
        self.melds[actor].append(Meld(_meld_kind("ankan"), tiles, False, actor))
        self.call_count += 1

    def _kakan(self, event: dict) -> None:
        from riichienv import Meld

        actor = int(event["actor"])
        pai = str(event["pai"])
        added = (
            self._take_owned_tile(pai)
            if actor == self.player_id else _public_tile(pai)
        )
        tile_type = added // 4
        for index, meld in enumerate(self.melds[actor]):
            if int(meld.meld_type) == 1 and meld.tiles and int(meld.tiles[0]) // 4 == tile_type:
                self.melds[actor][index] = Meld(
                    _meld_kind("kakan"), [*meld.tiles, added], True,
                    int(meld.from_who), meld.called_tile,
                )
                break
        else:
            raise AkagiProtocolError("kakan does not match an existing pon")
        if actor == self.player_id:
            self.drawn_tile = None
            self.own_draw_hidden = False
        self.call_count += 1

    def _reach_accepted(self, event: dict) -> None:
        actor = int(event["actor"])
        self.riichi_declared[actor] = True
        explicit = event.get("scores")
        if isinstance(explicit, list) and len(explicit) == 4:
            self.scores = list(map(int, explicit))
        else:
            self.scores[actor] -= 1_000
        self.riichi_sticks = int(event.get("kyotaku", self.riichi_sticks + 1))

    def _settlement(self, event: dict) -> None:
        deltas = event.get("deltas")
        if isinstance(deltas, list) and len(deltas) == 4:
            self.scores = [
                score + int(delta)
                for score, delta in zip(self.scores, deltas, strict=True)
            ]
        self.drawn_tile = None
        self.unknown_own_tiles = 0
        self.own_draw_hidden = False

    def _take_owned_tile(self, pai: str) -> int:
        """Remove a revealed tile from the known or unresolved own hand.

        A known matching tile is preferred because an unresolved tile could
        still be anything.  If no known copy matches, the event identifies one
        previously hidden tile and removes that uncertainty exactly.
        """
        candidates = set(_physical_candidates(pai))
        for index, tile in enumerate(self.hand):
            if tile in candidates:
                return self.hand.pop(index)
        if self.unknown_own_tiles > 0:
            self.unknown_own_tiles -= 1
            return _public_tile(pai)
        raise AkagiProtocolError(f"own hand does not contain {pai}")

    def _mark_called_discard(self, seat: int) -> None:
        # RiichiEnv observations omit called tiles from ``discards``. Zenith's
        # authoritative inference context separately preserves the river entry
        # and marks it called, so removing it here matches that API contract.
        if self.discards[seat]:
            self.discards[seat].pop()
            self.discard_tsumogiri[seat].pop()

    @property
    def live_wall_remaining(self) -> int:
        return max(0, 70 - self.tsumo_count)

    def _closed(self) -> bool:
        return all(not bool(meld.opened) for meld in self.melds[self.player_id])

    def _hand_analysis(self) -> tuple[list[int], bool]:
        from riichienv import HandEvaluator

        if self.unknown_own_tiles:
            return [], False
        try:
            evaluator = HandEvaluator(self.hand, self.melds[self.player_id])
            waits = list(map(int, evaluator.get_waits()))
            return waits, bool(evaluator.is_tenpai())
        except (RuntimeError, ValueError):
            return [], False

    def _can_win(self, win_tile: int, *, tsumo: bool, chankan: bool = False) -> bool:
        from riichienv import Conditions, HandEvaluator

        tiles = list(self.hand)
        if not tsumo:
            tiles.append(int(win_tile))
        conditions = Conditions(
            tsumo=tsumo,
            riichi=self.riichi_declared[self.player_id],
            houtei=not tsumo and self.live_wall_remaining == 0,
            haitei=tsumo and self.live_wall_remaining == 0,
            chankan=chankan,
            player_wind=(self.player_id - self.oya) % 4,
            round_wind=self.round_wind,
            riichi_sticks=self.riichi_sticks,
            honba=self.honba,
        )
        try:
            result = HandEvaluator(tiles, self.melds[self.player_id]).calc(
                int(win_tile), self.dora_indicators, conditions,
            )
        except (RuntimeError, ValueError):
            return False
        return bool(result.is_win)

    def _self_actions(self, *, discard_only: bool = False):
        from riichienv import Action, ActionType, check_riichi_candidates

        actor = self.player_id
        if self.drawn_tile is None and not self.hand:
            return []
        accepted = self.riichi_declared[actor]
        discard_tiles = [self.drawn_tile] if accepted and self.drawn_tile is not None else [
            tile for tile in self.hand
            if tile // 4 not in self.forbidden_discard_types
        ]
        actions = [Action(ActionType.DISCARD, tile, [], actor) for tile in discard_tiles]
        if discard_only:
            return actions
        if self.drawn_tile is not None and self._can_win(self.drawn_tile, tsumo=True):
            actions.append(Action(ActionType.TSUMO, self.drawn_tile, [], actor))
        if not accepted:
            by_type: dict[int, list[int]] = {}
            for tile in self.hand:
                by_type.setdefault(tile // 4, []).append(tile)
            for values in by_type.values():
                if len(values) == 4 and self.live_wall_remaining > 0:
                    actions.append(Action(ActionType.ANKAN, values[0], values, actor))
            pon_types = {
                int(meld.tiles[0]) // 4
                for meld in self.melds[actor]
                if int(meld.meld_type) == 1 and meld.tiles
            }
            for tile_type in sorted(pon_types):
                added = next((tile for tile in self.hand if tile // 4 == tile_type), None)
                if added is not None and self.live_wall_remaining > 0:
                    pon = next(
                        meld for meld in self.melds[actor]
                        if int(meld.meld_type) == 1
                        and int(meld.tiles[0]) // 4 == tile_type
                    )
                    actions.append(
                        Action(ActionType.KAKAN, added, list(pon.tiles), actor)
                    )
            if (
                self._closed()
                and self.scores[actor] >= 1_000
                and self.live_wall_remaining >= 4
                and check_riichi_candidates(list(self.hand))
            ):
                actions.append(Action(ActionType.RIICHI, actor=actor))
            if self.own_discard_count == 0 and self.call_count == 0:
                unique = {tile // 4 for tile in self.hand if tile // 4 in _TERMINAL_TYPES}
                if len(unique) >= 9:
                    actions.append(Action(ActionType.KYUSHU_KYUHAI, actor=actor))
        return actions

    def _reaction_actions(self, event: dict):
        from riichienv import Action, ActionType

        actor = self.player_id
        source = int(event.get("actor", -1))
        actions = [Action(ActionType.PASS, actor=actor)]
        if source == actor:
            return actions
        name = str(event["type"])
        pai = str(event.get("pai", "?"))
        if pai == "?":
            return actions
        called = _public_tile(pai)
        if self._can_win(called, tsumo=False, chankan=name in {"ankan", "kakan"}):
            actions.insert(0, Action(ActionType.RON, called, [], actor))
        if name != "dahai" or self.riichi_declared[actor]:
            return actions
        matches = [tile for tile in self.hand if tile // 4 == called // 4]
        for consumed in self._unique_combinations(matches, 2):
            actions.insert(-1, Action(ActionType.PON, called, consumed, actor))
        if len(matches) >= 3 and self.live_wall_remaining > 0:
            for consumed in self._unique_combinations(matches, 3):
                actions.insert(-1, Action(ActionType.DAIMINKAN, called, consumed, actor))
        if source == (actor + 3) % 4 and called // 4 < 27:
            for consumed in self._chi_consumed(called):
                actions.insert(-1, Action(ActionType.CHI, called, consumed, actor))
        return actions

    @staticmethod
    def _unique_combinations(values: list[int], count: int) -> list[list[int]]:
        unique = {}
        for choice in combinations(values, count):
            key = tuple(sorted((tile // 4, _is_red(tile)) for tile in choice))
            unique.setdefault(key, list(choice))
        return list(unique.values())

    def _chi_consumed(self, called: int) -> list[list[int]]:
        tile_type = called // 4
        suit_start = (tile_type // 9) * 9
        number = tile_type - suit_start
        result = {}
        for low in range(number - 2, number + 1):
            sequence = (low, low + 1, low + 2)
            if low < 0 or sequence[-1] > 8:
                continue
            required = [suit_start + value for value in sequence if value != number]
            choices = [
                [tile for tile in self.hand if tile // 4 == required_type]
                for required_type in required
            ]
            if any(not values for values in choices):
                continue
            for pair in product(*choices):
                key = tuple((tile // 4, _is_red(tile)) for tile in pair)
                result.setdefault(key, list(pair))
        return list(result.values())

    def observation(self, trigger: dict):
        """Return a RiichiEnv observation when ``trigger`` is a bot decision."""
        from riichienv import Observation

        if not self.active or self.player_id is None:
            return None
        if self.unknown_own_tiles:
            # The model must never choose from an ambiguous concealed hand.
            # Public play remains tracked, and a later discard/call can reveal
            # the unresolved tile and automatically restore exact inference.
            return None
        name = str(trigger.get("type", ""))
        actor = int(trigger.get("actor", -1))
        if (name == "tsumo" and actor == self.player_id) or (
            name in {"chi", "pon"} and actor == self.player_id
        ) or (name == "reach" and actor == self.player_id):
            legal = self._self_actions(discard_only=name in {"chi", "pon", "reach"})
        elif name in {"dahai", "ankan", "kakan"} and actor != self.player_id:
            legal = self._reaction_actions(trigger)
        else:
            return None
        if not legal:
            return None
        waits, is_tenpai = self._hand_analysis()
        hands = [[] for _ in range(4)]
        hands[self.player_id] = list(self.hand)
        return Observation(
            self.player_id,
            hands,
            self.melds,
            self.discards,
            self.dora_indicators,
            self.scores,
            self.riichi_declared,
            legal,
            [json.dumps(event, separators=(",", ":")) for event in self.events],
            self.honba,
            self.riichi_sticks,
            self.round_wind,
            self.oya,
            self.kyoku_index,
            waits,
            is_tenpai,
            self.riichi_sutehais,
            self.last_tedashis,
            None if self.last_discard is None else self.last_discard[1],
            self.drawn_tile,
        )


class AkagiSession:
    """One Akagi game backed by one mutable checkpoint-agent session."""

    def __init__(self, agent, *, state: LiveMjaiState | None = None):
        self.agent = agent
        self.state = LiveMjaiState() if state is None else state
        self.ended = False
        self.desynchronized = False

    def react(self, events) -> dict:
        if not isinstance(events, list) or not events:
            raise AkagiProtocolError("a reaction request must be a non-empty event array")
        if self.desynchronized:
            boundary = max(
                (
                    index for index, event in enumerate(events)
                    if isinstance(event, dict)
                    and event.get("type") in {"start_game", "start_kyoku"}
                ),
                default=-1,
            )
            if boundary < 0:
                return {"type": "none"}
            events = events[boundary:]
            if events[0].get("type") == "start_kyoku":
                # start_kyoku carries a complete current hand and scores, so
                # it is a safe recovery point after a malformed live hand.
                self.state.reset()
                self.agent.reset()
            self.desynchronized = False
        trigger = None
        for event in events:
            if not isinstance(event, dict):
                raise AkagiProtocolError("every event in a batch must be an object")
            if event.get("type") == "start_game":
                self.agent.reset()
                self.ended = False
                self.desynchronized = False
            else:
                self._reconcile_pending_recommendation(event)
            self.state.consume(event)
            observer = getattr(self.agent, "observe_event", None)
            if observer is not None:
                observer(event)
            trigger = event
            if event.get("type") == "end_game":
                self.ended = True
        if trigger is None or self.ended:
            return {"type": "none"}
        observation = self.state.observation(trigger)
        if observation is None:
            return {"type": "none"}
        action = self.agent.act(observation)
        response = action_response(action, observation)
        name = response.get("type")
        if name == "none":
            return {"type": "none"}
        if name == "ryukyoku":
            return {"type": "ryukyoku"}
        response["actor"] = self.state.player_id
        if name in {"chi", "pon", "daiminkan"} and self.state.last_discard is not None:
            response["target"] = self.state.last_discard[0]
        elif name == "hora":
            response["target"] = (
                self.state.player_id
                if trigger.get("type") == "tsumo"
                else int(trigger.get("actor", self.state.player_id))
            )
            response.pop("pai", None)
        elif name == "reach":
            pending = getattr(self.agent, "pending_riichi_tile", None)
            if pending is not None:
                response["pai"] = _tile_name(pending)
            # Akagi presents the combined riichi + discard recommendation in
            # one HUD response and doesn't request RiichiLab's second protocol
            # step. The live echo later tells us what the human actually did.
            cancel = getattr(self.agent, "cancel_pending_action", None)
            if cancel is not None:
                cancel()
            else:
                self.agent.pending_riichi_tile = None
        elif name == "ankan":
            response.pop("pai", None)
        diagnostics = getattr(self.agent, "last_diagnostics", None)
        if diagnostics:
            response["meta"] = self._diagnostic_meta(diagnostics)
        return response

    def desynchronize(self) -> None:
        """Fail this hand closed until an authoritative round boundary."""
        self.desynchronized = True
        self.state.active = False
        cancel = getattr(self.agent, "cancel_pending_action", None)
        if cancel is not None:
            cancel()
        elif hasattr(self.agent, "pending_riichi_tile"):
            self.agent.pending_riichi_tile = None

    def _reconcile_pending_recommendation(self, event: dict) -> None:
        """Keep forced riichi state only when the live game confirms riichi.

        RiichiLab executes the policy response, while Akagi normally presents
        it as advice to a human. If the human takes another action, the next
        authoritative event must cancel the policy's pending follow-up discard
        instead of applying it to an unrelated decision frame.
        """
        if getattr(self.agent, "pending_riichi_tile", None) is None:
            return
        confirms_riichi = (
            event.get("type") == "reach"
            and int(event.get("actor", -1)) == self.state.player_id
        )
        if confirms_riichi:
            return
        cancel = getattr(self.agent, "cancel_pending_action", None)
        if cancel is not None:
            cancel()
        else:
            self.agent.pending_riichi_tile = None

    def _diagnostic_meta(self, diagnostics: dict) -> dict:
        waits, is_tenpai = self.state._hand_analysis()
        state = {
            "live_wall_remaining": self.state.live_wall_remaining,
            "is_tenpai": is_tenpai,
            "waits": [_tile_name(tile) for tile in waits],
            "scores": list(self.state.scores),
        }
        top_actions = diagnostics.get("top_actions", ())
        items = [self._diagnostic_item(item) for item in top_actions]
        selected = diagnostics.get("selected_action", {})
        confidence = float(selected.get("policy_probability", 0.0))
        return {
            "confidence": confidence,
            "policy": diagnostics,
            "state": state,
            "show": {
                "title": "Zenith policy — top actions",
                "items": items,
            },
        }

    @staticmethod
    def _diagnostic_item(item: dict) -> dict:
        action = item.get("action", {})
        name = str(action.get("type", "action"))
        pai = action.get("pai")
        labels = {
            "dahai": "Discard",
            "reach": "Riichi",
            "chi": "Chi",
            "pon": "Pon",
            "daiminkan": "Open kan",
            "ankan": "Closed kan",
            "kakan": "Added kan",
            "hora": "Win",
            "ryukyoku": "Abortive draw",
            "none": "Pass",
        }
        label = labels.get(name, name)
        if pai is not None:
            label = f"{label} {pai}"
        pais = [
            str(value)
            for value in (*action.get("consumed", ()), pai)
            if value is not None
        ]
        notes = []
        if item.get("selected"):
            notes.append("selected")
        prospects = item.get("prospects", {})
        outcomes = prospects.get("hand_outcome_probabilities", {})
        if outcomes:
            notes.append(f"win {100 * float(outcomes.get('win', 0)):.1f}%")
            notes.append(
                f"deal-in {100 * float(outcomes.get('deal_in', 0)):.1f}%"
            )
        if "expected_score_delta" in prospects:
            notes.append(f"projected delta {float(prospects['expected_score_delta']):+.0f}")
        result = {
            "label": f"#{int(item.get('rank', 0))} {label}",
            "value": f"{100 * float(item.get('policy_probability', 0)):.1f}%",
        }
        if pais:
            result["pais"] = pais
        if notes:
            result["note"] = " · ".join(notes)
        if item.get("selected"):
            result["color"] = "#22c55e"
        return result
