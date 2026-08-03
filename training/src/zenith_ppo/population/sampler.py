"""Pure or adversarial self-play lineup construction."""

from __future__ import annotations

from ..types import MatchLineup

class SelfPlaySampler:
    def __init__(self, streams, league=None):
        from .league import AdversarialLeague

        self.streams = streams
        self.league = league or AdversarialLeague()

    def sample(self, *, environment_id, generation, current_id, policy_version):
        if len(self.league.policy_ids) == 1:
            return MatchLineup(
                (environment_id, generation),
                (current_id,) * 4,
                0b1111,
                policy_version,
                "pure-self-play",
                ({"kind": "pure_self_play", "policy": current_id},),
            )
        rng = self.streams.python_rng("opponent")
        first, second, trace = self.league.select_pair(rng)
        rotation = rng.randrange(2)
        seats = tuple(
            first if (seat + rotation) % 2 == 0 else second
            for seat in range(4)
        )
        trainable = set(getattr(
            self.league, "trainable_policy_ids", self.league.policy_ids
        ))
        learner_mask = sum(
            (1 << seat) for seat, policy_id in enumerate(seats)
            if policy_id in trainable
        )
        return MatchLineup(
            (environment_id, generation), seats, learner_mask, policy_version,
            "checkpoint-league" if learner_mask != 0b1111 else "adversarial-league",
            (trace,),
        )
