"""Versioned uniform-without-replacement 2+2 lineups."""

from __future__ import annotations

from .registry import PoolSnapshot
from ..types import MatchLineup

SAMPLER_VERSION = 1


class UniformSampler:
    def __init__(self, streams, learner_seats=2, shortage="all_current_bootstrap"):
        if not 1 <= learner_seats <= 3: raise ValueError("learner_seats must be in 1..3")
        self.streams, self.learner_seats, self.shortage = streams, learner_seats, shortage

    def sample(self, snapshot, *, environment_id, generation, current_id, policy_version):
        needed = 4 - self.learner_seats
        candidates = [entry.checkpoint_id for entry in snapshot.eligible if entry.checkpoint_id != current_id]
        trace, chosen = [], []
        rng = self.streams.python_rng("opponent")
        if len(candidates) < needed:
            if self.shortage != "all_current_bootstrap": raise RuntimeError("insufficient historical checkpoints")
            seats = (current_id,) * 4; mask = 0b1111
        else:
            for _ in range(needed):
                probabilities = [1 / len(candidates)] * len(candidates)
                draw = rng.random(); index = min(int(draw * len(candidates)), len(candidates) - 1)
                trace.append({"candidates": tuple(candidates), "probabilities": tuple(probabilities),
                              "draw": draw, "chosen": candidates[index]})
                chosen.append(candidates.pop(index))
            rotation = (environment_id + generation) % 4
            learner_positions = {(rotation + index * (4 // self.learner_seats)) % 4
                                 for index in range(self.learner_seats)}
            history = iter(chosen); seats = tuple(current_id if seat in learner_positions else next(history)
                                                  for seat in range(4))
            mask = sum(1 << seat for seat in learner_positions)
        return MatchLineup((environment_id, generation), seats, mask, policy_version,
                           snapshot.snapshot_id, tuple(trace))

    def select_cohort(self, snapshot, size, *, preferred=()):
        """Select one rollout-wide historical cohort uniformly without replacement."""
        size = int(size)
        if size < 1:
            raise ValueError("checkpoint cohort size must be positive")
        eligible = list(snapshot.eligible)
        by_id = {entry.checkpoint_id: entry for entry in eligible}
        selected = [
            by_id[checkpoint_id]
            for checkpoint_id in dict.fromkeys(map(str, preferred))
            if checkpoint_id in by_id
        ][:size]
        selected_ids = {entry.checkpoint_id for entry in selected}
        candidates = [
            entry for entry in eligible if entry.checkpoint_id not in selected_ids
        ]
        remaining = size - len(selected)
        if remaining <= 0:
            return PoolSnapshot(tuple(selected), snapshot.rating_snapshot_id), ()
        if len(candidates) <= remaining:
            selected.extend(candidates)
            return PoolSnapshot(tuple(selected), snapshot.rating_snapshot_id), ()
        rng = self.streams.python_rng("opponent")
        trace = []
        while len(selected) < size:
            probabilities = tuple(1 / len(candidates) for _ in candidates)
            draw = rng.random()
            index = min(int(draw * len(candidates)), len(candidates) - 1)
            chosen = candidates.pop(index)
            trace.append({
                "candidates": tuple(entry.checkpoint_id for entry in candidates[:index])
                    + (chosen.checkpoint_id,)
                    + tuple(entry.checkpoint_id for entry in candidates[index:]),
                "probabilities": probabilities,
                "draw": draw,
                "chosen": chosen.checkpoint_id,
            })
            selected.append(chosen)
        return PoolSnapshot(tuple(selected), snapshot.rating_snapshot_id), tuple(trace)
