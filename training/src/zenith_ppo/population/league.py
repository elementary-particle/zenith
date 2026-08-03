"""Pure, adversarial, and frozen-checkpoint self-play populations."""

from __future__ import annotations

from dataclasses import dataclass


LEAGUE_SIZE = 4
PLACEMENT_UTILITY = (1.0, 1.0 / 3.0, -1.0 / 3.0, -1.0)


class PureSelfPlayLeague:
    """Single live policy owning every seat in every training match."""

    VERSION = 1

    def __init__(self, policy_id="self-play", *, state=None):
        self.policy_ids = (str(policy_id),)
        self.trainable_policy_ids = self.policy_ids
        self.games = 0
        if state is not None:
            self.load_state_dict(state)

    def record(self, outcomes):
        for outcome in outcomes:
            if set(outcome.checkpoint_ids) != set(self.policy_ids):
                raise ValueError("pure self-play outcome contains another policy")
        self.games += len(outcomes)
        return len(outcomes)

    def metrics(self):
        return {
            "arena_size": 1.0,
            "observed_pairs": 0.0,
            "pairwise_games": float(self.games),
            "maximum_payoff_gap": 0.0,
        }

    def state_dict(self):
        return {
            "version": self.VERSION,
            "kind": "pure_self_play",
            "policy_ids": list(self.policy_ids),
            "games": self.games,
        }

    def load_state_dict(self, state):
        state = dict(state or {})
        if state.get("kind") != "pure_self_play" \
                or int(state.get("version", -1)) != self.VERSION:
            raise ValueError("unsupported pure self-play state")
        if tuple(state.get("policy_ids", ())) != self.policy_ids:
            raise ValueError("pure self-play policy identity changed")
        self.games = int(state.get("games", 0))


class EMASelfPlayLeague:
    """One live learner alternating seats with its dynamic EMA actor."""

    VERSION = 1

    def __init__(self, learner_id="learner", opponent_id="ema-opponent", *, state=None):
        if str(learner_id) == str(opponent_id):
            raise ValueError("EMA learner and opponent identities must differ")
        self.policy_ids = (str(learner_id), str(opponent_id))
        self.trainable_policy_ids = (str(learner_id),)
        self.games = 0
        if state is not None:
            self.load_state_dict(state)

    @property
    def learner_id(self):
        return self.policy_ids[0]

    @property
    def opponent_id(self):
        return self.policy_ids[1]

    def select_pair(self, rng):
        del rng
        return self.learner_id, self.opponent_id, {
            "kind": "ema_self_play",
            "learner": self.learner_id,
            "opponent": self.opponent_id,
            "games": self.games,
        }

    def record(self, outcomes):
        for outcome in outcomes:
            if set(outcome.checkpoint_ids) != set(self.policy_ids):
                raise ValueError("EMA self-play outcome has unknown policies")
            if any(outcome.checkpoint_ids.count(identity) != 2
                   for identity in self.policy_ids):
                raise ValueError("EMA self-play policies must own two seats each")
        self.games += len(outcomes)
        return len(outcomes)

    def metrics(self):
        return {
            "arena_size": 2.0,
            "observed_pairs": float(self.games > 0),
            "pairwise_games": float(self.games),
            "maximum_payoff_gap": 0.0,
        }

    def state_dict(self):
        return {
            "version": self.VERSION,
            "kind": "ema_self_play",
            "policy_ids": list(self.policy_ids),
            "games": self.games,
        }

    def load_state_dict(self, state):
        state = dict(state or {})
        if state.get("kind") != "ema_self_play" \
                or int(state.get("version", -1)) != self.VERSION:
            raise ValueError("unsupported EMA self-play state")
        if tuple(state.get("policy_ids", ())) != self.policy_ids:
            raise ValueError("EMA self-play policy identity changed")
        self.games = int(state.get("games", -1))
        if self.games < 0:
            raise ValueError("EMA self-play game count is invalid")


@dataclass(frozen=True, slots=True)
class PairEstimate:
    games: int = 0
    payoff_sum: float = 0.0

    @property
    def mean(self) -> float:
        return self.payoff_sum / self.games if self.games else 0.0


class AdversarialLeague:
    """Fixed four-live-model league with seeded, resume-safe matchmaking."""

    VERSION = 1

    def __init__(self, policy_ids=None, *, minimum_pair_games=1, state=None):
        ids = tuple(policy_ids or (f"league-{index}" for index in range(LEAGUE_SIZE)))
        if len(ids) != LEAGUE_SIZE or len(set(ids)) != LEAGUE_SIZE:
            raise ValueError("the league must contain exactly four unique models")
        if int(minimum_pair_games) < 1:
            raise ValueError("minimum_pair_games must be positive")
        self.policy_ids = ids
        self.trainable_policy_ids = self.policy_ids
        self.minimum_pair_games = int(minimum_pair_games)
        self._focal_cursor = 0
        self._selections = {policy_id: 0 for policy_id in ids}
        self._pairs = {
            self._key(left, right): PairEstimate()
            for index, left in enumerate(ids)
            for right in ids[index + 1:]
        }
        self._scheduled = {key: 0 for key in self._pairs}
        if state is not None:
            self.load_state_dict(state)

    def _key(self, left, right):
        if left == right or left not in self.policy_ids or right not in self.policy_ids:
            raise ValueError("league payoff requires two distinct league policies")
        return tuple(sorted((left, right), key=self.policy_ids.index))

    def estimate(self, policy_id, opponent_id) -> PairEstimate:
        estimate = self._pairs[self._key(policy_id, opponent_id)]
        if self.policy_ids.index(policy_id) < self.policy_ids.index(opponent_id):
            return estimate
        return PairEstimate(estimate.games, -estimate.payoff_sum)

    def select_pair(self, rng):
        focal = self.policy_ids[self._focal_cursor % LEAGUE_SIZE]
        self._focal_cursor += 1
        candidates = [policy_id for policy_id in self.policy_ids if policy_id != focal]

        def effective_games(opponent):
            return (
                self.estimate(focal, opponent).games
                + self._scheduled[self._key(focal, opponent)]
            )

        exploratory = [
            opponent for opponent in candidates
            if effective_games(opponent) < self.minimum_pair_games
        ]
        if exploratory:
            fewest = min(effective_games(opponent) for opponent in exploratory)
            choices = [
                opponent for opponent in exploratory
                if effective_games(opponent) == fewest
            ]
            mode = "pairwise-evaluation"
        else:
            hardest = min(self.estimate(focal, opponent).mean for opponent in candidates)
            choices = [
                opponent for opponent in candidates
                if self.estimate(focal, opponent).mean == hardest
            ]
            mode = "adversarial-perturbation"
        opponent = choices[rng.randrange(len(choices))]
        self._scheduled[self._key(focal, opponent)] += 1
        self._selections[focal] += 1
        self._selections[opponent] += 1
        estimate = self.estimate(focal, opponent)
        return focal, opponent, {
            "kind": "league_matchmaking",
            "mode": mode,
            "focal": focal,
            "opponent": opponent,
            "games": estimate.games,
            "focal_payoff": estimate.mean,
        }

    def record(self, outcomes):
        recorded = 0
        for outcome in outcomes:
            identities = tuple(dict.fromkeys(outcome.checkpoint_ids))
            if len(identities) != 2 or any(
                policy_id not in self.policy_ids for policy_id in identities
            ):
                raise ValueError("league outcomes must contain exactly two league models")
            left, right = identities
            left_seats = [
                seat for seat, policy_id in enumerate(outcome.checkpoint_ids)
                if policy_id == left
            ]
            right_seats = [
                seat for seat, policy_id in enumerate(outcome.checkpoint_ids)
                if policy_id == right
            ]
            if len(left_seats) != 2 or len(right_seats) != 2:
                raise ValueError("each selected league model must own exactly two seats")
            left_utility = sum(
                PLACEMENT_UTILITY[outcome.ranks[seat]] for seat in left_seats
            ) / 2
            right_utility = sum(
                PLACEMENT_UTILITY[outcome.ranks[seat]] for seat in right_seats
            ) / 2
            payoff = left_utility - right_utility
            key = self._key(left, right)
            if self._scheduled[key]:
                self._scheduled[key] -= 1
            if key[0] != left:
                payoff = -payoff
            previous = self._pairs[key]
            self._pairs[key] = PairEstimate(
                previous.games + 1, previous.payoff_sum + payoff
            )
            recorded += 1
        return recorded

    def metrics(self):
        estimates = tuple(self._pairs.values())
        observed = [estimate for estimate in estimates if estimate.games]
        return {
            "arena_size": float(LEAGUE_SIZE),
            "observed_pairs": float(len(observed)),
            "pairwise_games": float(sum(estimate.games for estimate in estimates)),
            "maximum_payoff_gap": float(
                max((abs(estimate.mean) for estimate in observed), default=0.0)
            ),
        }

    def state_dict(self):
        return {
            "version": self.VERSION,
            "policy_ids": list(self.policy_ids),
            "minimum_pair_games": self.minimum_pair_games,
            "focal_cursor": self._focal_cursor,
            "selections": dict(self._selections),
            "pairs": [
                {
                    "left": left,
                    "right": right,
                    "games": estimate.games,
                    "payoff_sum": estimate.payoff_sum,
                }
                for (left, right), estimate in self._pairs.items()
            ],
        }

    def load_state_dict(self, state):
        state = dict(state or {})
        if int(state.get("version", -1)) != self.VERSION:
            raise ValueError("unsupported league state version")
        if tuple(state.get("policy_ids", ())) != self.policy_ids:
            raise ValueError("league policy identities changed across resume")
        if int(state.get("minimum_pair_games", -1)) != self.minimum_pair_games:
            raise ValueError("league minimum_pair_games changed across resume")
        self._focal_cursor = int(state.get("focal_cursor", 0))
        selections = state.get("selections", {})
        if set(selections) != set(self.policy_ids):
            raise ValueError("league selection counters are incomplete")
        self._selections = {
            policy_id: int(selections[policy_id]) for policy_id in self.policy_ids
        }
        restored = {}
        for row in state.get("pairs", ()):
            key = self._key(row["left"], row["right"])
            restored[key] = PairEstimate(int(row["games"]), float(row["payoff_sum"]))
        if set(restored) != set(self._pairs):
            raise ValueError("league payoff table is incomplete")
        self._pairs = restored
        self._scheduled = {key: 0 for key in self._pairs}


class CheckpointLeague:
    """One live learner trained against a fixed, OpenSkill-rated arena."""

    VERSION = 1

    def __init__(self, opponent_ids, *, learner_id="learner",
                 minimum_games=8, uniform_fraction=0.25,
                 rating_parameters=None, state=None):
        opponents = tuple(map(str, opponent_ids))
        if not opponents or len(set(opponents)) != len(opponents):
            raise ValueError("checkpoint opponents must be non-empty and unique")
        if learner_id in opponents:
            raise ValueError("learner identity collides with checkpoint opponent")
        if int(minimum_games) < 1:
            raise ValueError("checkpoint league minimum games must be positive")
        if not 0 <= float(uniform_fraction) <= 1:
            raise ValueError("checkpoint league uniform fraction must be in [0,1]")
        self.policy_ids = (str(learner_id),) + opponents
        self.trainable_policy_ids = (str(learner_id),)
        self.minimum_games = int(minimum_games)
        self.uniform_fraction = float(uniform_fraction)
        self._games = {opponent: 0 for opponent in opponents}
        self._scheduled = {opponent: 0 for opponent in opponents}
        from ..evaluation.ratings import RatingTable
        self.ratings = RatingTable(parameters=rating_parameters)
        if state is not None:
            self.load_state_dict(state)

    @property
    def learner_id(self):
        return self.trainable_policy_ids[0]

    @property
    def opponent_ids(self):
        return self.policy_ids[1:]

    def select_pair(self, rng):
        coverage = [
            opponent for opponent in self.opponent_ids
            if self._games[opponent] + self._scheduled[opponent]
            < self.minimum_games
        ]
        if coverage:
            least = min(
                self._games[opponent] + self._scheduled[opponent]
                for opponent in coverage
            )
            choices = [
                opponent for opponent in coverage
                if self._games[opponent] + self._scheduled[opponent] == least
            ]
            mode = "rating-calibration"
        elif rng.random() < self.uniform_fraction:
            choices = list(self.opponent_ids)
            mode = "uniform-diversity"
        else:
            ratings = self.ratings.ratings
            rated = [opponent for opponent in self.opponent_ids if opponent in ratings]
            if not rated:
                choices = list(self.opponent_ids)
            else:
                best = max(ratings[opponent].ordinal for opponent in rated)
                choices = [
                    opponent for opponent in rated
                    if ratings[opponent].ordinal == best
                ]
            mode = "strongest-opponent"
        opponent = choices[rng.randrange(len(choices))]
        self._scheduled[opponent] += 1
        rating = self.ratings.ratings.get(opponent)
        return self.learner_id, opponent, {
            "kind": "checkpoint_league_matchmaking",
            "mode": mode,
            "learner": self.learner_id,
            "opponent": opponent,
            "opponent_games": 0 if rating is None else rating.games,
            "opponent_ordinal": 0.0 if rating is None else rating.ordinal,
        }

    def record(self, outcomes):
        rows = []
        for sequence, outcome in enumerate(outcomes):
            identities = set(outcome.checkpoint_ids)
            if self.learner_id not in identities or len(identities) != 2:
                raise ValueError(
                    "checkpoint league outcomes require learner and one opponent"
                )
            opponent = next(
                identity for identity in identities if identity != self.learner_id
            )
            if opponent not in self._games:
                raise ValueError("checkpoint league outcome has unknown opponent")
            if outcome.checkpoint_ids.count(self.learner_id) != 2 \
                    or outcome.checkpoint_ids.count(opponent) != 2:
                raise ValueError("checkpoint league policies must own two seats each")
            self._games[opponent] += 1
            if self._scheduled[opponent]:
                self._scheduled[opponent] -= 1
            rows.append({
                "series_id": f"league-{outcome.match_id[0]}",
                "game_id": int(outcome.match_id[1]) * 1_000_000 + sequence,
                "checkpoint_ids": outcome.checkpoint_ids,
                "ranks": outcome.ranks,
                "valid": True,
            })
        if rows:
            self.ratings.update(rows)
        return len(rows)

    def metrics(self):
        ratings = self.ratings.ratings
        learner = ratings.get(self.learner_id)
        gaps = [
            abs(learner.ordinal - ratings[opponent].ordinal)
            for opponent in self.opponent_ids
            if learner is not None and opponent in ratings
        ]
        return {
            "arena_size": float(len(self.policy_ids)),
            "observed_pairs": float(sum(
                games > 0 for games in self._games.values()
            )),
            "pairwise_games": float(sum(self._games.values())),
            "maximum_payoff_gap": float(max(gaps, default=0.0)),
        }

    def state_dict(self):
        return {
            "version": self.VERSION,
            "kind": "checkpoint_league",
            "policy_ids": list(self.policy_ids),
            "minimum_games": self.minimum_games,
            "uniform_fraction": self.uniform_fraction,
            "games": dict(self._games),
            "ratings": self.ratings.state_dict(),
        }

    def load_state_dict(self, state):
        state = dict(state or {})
        if state.get("kind") != "checkpoint_league" \
                or int(state.get("version", -1)) != self.VERSION:
            raise ValueError("unsupported checkpoint league state")
        if tuple(state.get("policy_ids", ())) != self.policy_ids:
            raise ValueError("checkpoint league identities changed across resume")
        if int(state.get("minimum_games", -1)) != self.minimum_games \
                or float(state.get("uniform_fraction", -1)) != self.uniform_fraction:
            raise ValueError("checkpoint league matchmaking changed across resume")
        games = {key: int(value) for key, value in state.get("games", {}).items()}
        if set(games) != set(self.opponent_ids):
            raise ValueError("checkpoint league game counts are incomplete")
        self._games = games
        self._scheduled = {opponent: 0 for opponent in self.opponent_ids}
        from ..evaluation.ratings import RatingTable
        self.ratings = RatingTable.from_state_dict(state.get("ratings", {}))
