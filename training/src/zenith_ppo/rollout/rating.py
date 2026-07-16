"""Direct rollout strength estimates against the deterministic baseline bot."""

from __future__ import annotations

from dataclasses import dataclass
import math

from ..inference import CONSERVATIVE_BOT_ID


@dataclass(slots=True)
class ConservativeBotRating:
    """Accumulate policy-vs-bot placement comparisons.

    A two-policy match contributes one comparison for every policy/bot seat
    pair.  This uses the actual four-player placements without pretending the
    changing live policy and its checkpoints are stable rating identities.
    """

    bot_matches: int = 0
    wins: int = 0
    comparisons: int = 0

    def update(self, outcomes) -> "ConservativeBotRating":
        for outcome in outcomes:
            identities = tuple(map(str, outcome.checkpoint_ids))
            placements = tuple(map(int, outcome.ranks))
            if len(identities) != 4 or sorted(placements) != list(range(4)):
                raise ValueError("rollout outcomes require four resolved placements")
            bot_seats = [
                seat for seat, identity in enumerate(identities)
                if identity == CONSERVATIVE_BOT_ID
            ]
            if not bot_seats:
                continue
            policy_seats = [
                seat for seat, identity in enumerate(identities)
                if identity != CONSERVATIVE_BOT_ID
            ]
            if not policy_seats:
                raise ValueError("bot rating matches require live-policy seats")
            self.bot_matches += 1
            for policy_seat in policy_seats:
                for bot_seat in bot_seats:
                    self.comparisons += 1
                    self.wins += placements[policy_seat] < placements[bot_seat]
        return self

    def metrics(self, *, confidence_z: float = 1.959963984540054) -> dict[str, float]:
        """Return the empirical win rate and its Wilson score interval."""
        if self.comparisons == 0:
            return {
                "win_rate_vs_conservative_bot": 0.5,
                "lower_win_rate_vs_conservative_bot": 0.0,
                "upper_win_rate_vs_conservative_bot": 1.0,
                "bot_matches": float(self.bot_matches),
                "head_to_head_comparisons": 0.0,
            }
        n = float(self.comparisons)
        probability = self.wins / n
        z = float(confidence_z)
        denominator = 1.0 + z * z / n
        center = (probability + z * z / (2.0 * n)) / denominator
        radius = z * math.sqrt(
            probability * (1.0 - probability) / n + z * z / (4.0 * n * n)
        ) / denominator
        return {
            "win_rate_vs_conservative_bot": probability,
            "lower_win_rate_vs_conservative_bot": max(0.0, center - radius),
            "upper_win_rate_vs_conservative_bot": min(1.0, center + radius),
            "bot_matches": float(self.bot_matches),
            "head_to_head_comparisons": float(self.comparisons),
        }

    def state_dict(self):
        return {
            "bot_matches": self.bot_matches,
            "wins": self.wins,
            "comparisons": self.comparisons,
        }

    @classmethod
    def from_state_dict(cls, state):
        return cls(
            bot_matches=int(state.get("bot_matches", 0)),
            wins=int(state.get("wins", 0)),
            comparisons=int(state.get("comparisons", 0)),
        )
