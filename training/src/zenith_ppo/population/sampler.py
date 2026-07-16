"""Pure self-play lineups with occasional conservative-bot probes."""

from __future__ import annotations

from ..types import MatchLineup

class SelfPlaySampler:
    def __init__(self, streams, conservative_bot_match_fraction=0.05):
        if not 0 <= float(conservative_bot_match_fraction) <= 1:
            raise ValueError("bot match fraction must be in [0,1]")
        self.streams = streams
        self.conservative_bot_match_fraction = float(conservative_bot_match_fraction)

    def sample(self, *, environment_id, generation, current_id, policy_version):
        from ..inference import CONSERVATIVE_BOT_ID

        rng = self.streams.python_rng("opponent")
        draw = rng.random()
        bot_match = draw < self.conservative_bot_match_fraction
        if bot_match:
            rotation = (int(environment_id) + int(generation)) % 4
            learner_positions = {rotation, (rotation + 2) % 4}
            seats = tuple(
                current_id if seat in learner_positions else CONSERVATIVE_BOT_ID
                for seat in range(4)
            )
            learner_mask = sum(1 << seat for seat in learner_positions)
        else:
            seats = (current_id,) * 4
            learner_mask = 0b1111
        trace = ({
            "kind": "conservative_bot_probe",
            "draw": draw,
            "probability": self.conservative_bot_match_fraction,
            "chosen": bot_match,
        },)
        return MatchLineup(
            (environment_id, generation), seats, learner_mask, policy_version,
            "self-play", trace,
        )
