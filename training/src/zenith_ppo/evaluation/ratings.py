"""Deterministic official Plackett-Luce checkpoint ratings."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction


@dataclass(frozen=True, slots=True)
class Rating:
    mu: float = 25.0
    sigma: float = 25.0 / 3.0
    games: int = 0
    placements: tuple[int, int, int, int] = (0, 0, 0, 0)
    last_series: str | None = None

    @property
    def ordinal(self): return self.mu - 3.0 * self.sigma


class RatingTable:
    def __init__(self, ratings=None, *, parameters=None):
        self.ratings = dict(ratings or {})
        self.parameters = {
            "mu": 25.0, "sigma": 25.0 / 3.0, "beta": 25.0 / 6.0,
            "kappa": 0.0001, "tau": 0.0, "ordinal_sigma": 3.0,
            **(parameters or {}),
        }
    def update(self, outcomes):
        try:
            from openskill.models import PlackettLuce
        except ImportError as exc:
            raise RuntimeError("OpenSkill 6.2 is required for official ratings") from exc
        model = PlackettLuce(**{
            key: self.parameters[key]
            for key in ("mu", "sigma", "beta", "kappa", "tau")
        })
        next_ratings = dict(self.ratings)
        for outcome in sorted((item for item in outcomes if item.get("valid", True)),
                              key=lambda item: (item["series_id"], item["game_id"])):
            ids, ranks = tuple(outcome["checkpoint_ids"]), list(map(int, outcome["ranks"]))
            if len(ids) != 4 or len(ranks) != 4 or any(rank not in range(4) for rank in ranks):
                raise ValueError("valid rating outcomes require four bound ranks")
            grouped = {}
            for identity, rank in zip(ids, ranks, strict=True):
                grouped.setdefault(identity, []).append(rank)
            identities = sorted(grouped)
            means = {key: Fraction(sum(grouped[key]), len(grouped[key])) for key in identities}
            ordered_means = sorted(set(means.values()))
            grouped_ranks = [ordered_means.index(means[key]) for key in identities]
            teams = [[model.rating(mu=next_ratings.get(cid, Rating()).mu,
                                   sigma=next_ratings.get(cid, Rating()).sigma)] for cid in identities]
            result = model.rate(teams, ranks=grouped_ranks)
            for cid, team in zip(identities, result, strict=True):
                old = next_ratings.get(cid, Rating()); counts = list(old.placements)
                for rank in grouped[cid]:
                    counts[rank] += 1
                next_ratings[cid] = Rating(float(team[0].mu), float(team[0].sigma), old.games + 1,
                                           tuple(counts), outcome["series_id"])
        self.ratings = next_ratings
        return self

    def leaderboard(self):
        scale = float(self.parameters["ordinal_sigma"])
        return sorted(
            self.ratings.items(),
            key=lambda item: (-(item[1].mu - scale * item[1].sigma), item[0]),
        )

    def state_dict(self):
        return {
            "parameters": dict(self.parameters),
            "ratings": {key: asdict(value) for key, value in sorted(self.ratings.items())},
        }

    @classmethod
    def from_state_dict(cls, state):
        ratings = {
            key: Rating(
                mu=float(value["mu"]),
                sigma=float(value["sigma"]),
                games=int(value.get("games", 0)),
                placements=tuple(value.get("placements", (0, 0, 0, 0))),
                last_series=value.get("last_series"),
            )
            for key, value in state.get("ratings", {}).items()
        }
        return cls(ratings, parameters=state.get("parameters"))
