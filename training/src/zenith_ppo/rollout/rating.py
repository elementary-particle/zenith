"""Recent rollout strength estimates against the deterministic baseline bot."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

from ..inference import CONSERVATIVE_BOT_ID


_STATE_VERSION = 2
_WINDOW_UPDATES = 10


@dataclass(frozen=True, slots=True)
class _UpdateResult:
    bot_matches: int = 0
    wins: int = 0
    comparisons: int = 0

    def state_dict(self) -> dict[str, int]:
        return {
            "bot_matches": self.bot_matches,
            "wins": self.wins,
            "comparisons": self.comparisons,
        }

    @classmethod
    def from_state_dict(cls, state) -> "_UpdateResult":
        if not isinstance(state, dict):
            raise ValueError("rolling bot rating update state must be a mapping")
        unknown = set(state) - {"bot_matches", "wins", "comparisons"}
        if unknown:
            raise ValueError(f"unknown rolling bot rating update fields: {sorted(unknown)}")
        result = cls(
            bot_matches=int(state.get("bot_matches", 0)),
            wins=int(state.get("wins", 0)),
            comparisons=int(state.get("comparisons", 0)),
        )
        if result.bot_matches < 0 or result.comparisons < 0:
            raise ValueError("rolling bot rating counts must be non-negative")
        if result.wins < 0 or result.wins > result.comparisons:
            raise ValueError("rolling bot rating wins must be within comparisons")
        return result


@dataclass(slots=True)
class ConservativeBotRating:
    """Track policy-vs-bot placement comparisons over ten rollout updates.

    A two-policy match contributes one comparison for every policy/bot seat
    pair. Every call to :meth:`update` occupies one slot, including calls with
    no bot probes, so the window always represents rollout updates rather than
    a variable number of policy versions.
    """

    updates: list[_UpdateResult] = field(default_factory=list)

    def update(self, outcomes) -> "ConservativeBotRating":
        bot_matches = wins = comparisons = 0
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
            bot_matches += 1
            for policy_seat in policy_seats:
                for bot_seat in bot_seats:
                    comparisons += 1
                    wins += placements[policy_seat] < placements[bot_seat]
        self.updates.append(_UpdateResult(bot_matches, wins, comparisons))
        del self.updates[:-_WINDOW_UPDATES]
        return self

    def metrics(self, *, confidence_z: float = 1.959963984540054) -> dict[str, float]:
        """Return rolling support counts and, when observed, a Wilson interval."""
        bot_matches = sum(update.bot_matches for update in self.updates)
        wins = sum(update.wins for update in self.updates)
        comparisons = sum(update.comparisons for update in self.updates)
        metrics = {
            "rolling_bot_matches": float(bot_matches),
            "rolling_head_to_head_comparisons": float(comparisons),
            "rolling_window_updates": float(len(self.updates)),
        }
        if comparisons == 0:
            return metrics
        n = float(comparisons)
        probability = wins / n
        z = float(confidence_z)
        denominator = 1.0 + z * z / n
        center = (probability + z * z / (2.0 * n)) / denominator
        radius = z * math.sqrt(
            probability * (1.0 - probability) / n + z * z / (4.0 * n * n)
        ) / denominator
        return {
            **metrics,
            "rolling_win_rate_vs_conservative_bot": probability,
            "rolling_lower_win_rate_vs_conservative_bot": max(0.0, center - radius),
            "rolling_upper_win_rate_vs_conservative_bot": min(1.0, center + radius),
        }

    def state_dict(self):
        return {
            "version": _STATE_VERSION,
            "window_updates": _WINDOW_UPDATES,
            "updates": [update.state_dict() for update in self.updates],
        }

    @classmethod
    def from_state_dict(cls, state):
        if not isinstance(state, dict):
            raise ValueError("rolling bot rating state must be a mapping")
        # Version-one checkpoints stored lifetime totals only. Those totals
        # cannot recover a recent window, so migration deliberately starts fresh.
        if "version" not in state:
            unknown = set(state) - {"bot_matches", "wins", "comparisons"}
            if unknown:
                raise ValueError(f"unknown legacy bot rating state fields: {sorted(unknown)}")
            _UpdateResult.from_state_dict(state)
            return cls()
        if int(state["version"]) != _STATE_VERSION:
            raise ValueError(f"unsupported rolling bot rating state version {state['version']}")
        if int(state.get("window_updates", -1)) != _WINDOW_UPDATES:
            raise ValueError(
                f"rolling bot rating window must be {_WINDOW_UPDATES} updates"
            )
        unknown = set(state) - {"version", "window_updates", "updates"}
        if unknown:
            raise ValueError(f"unknown rolling bot rating state fields: {sorted(unknown)}")
        updates = [_UpdateResult.from_state_dict(row) for row in state.get("updates", ())]
        if len(updates) > _WINDOW_UPDATES:
            raise ValueError(
                f"rolling bot rating state contains more than {_WINDOW_UPDATES} updates"
            )
        return cls(updates=updates)
