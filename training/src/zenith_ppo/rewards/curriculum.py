"""Completed-match training progress."""

from __future__ import annotations

from ..types import CurriculumSnapshot


class Curriculum:
    def __init__(self, config, state=None):
        self.config = dict(config)
        if state not in (None, {}):
            raise ValueError("training-progress state must be empty")

    def snapshot(self, completed_matches: int, policy_version: int):
        completed_matches = int(completed_matches)
        total = int(self.config["total_matches"])
        return CurriculumSnapshot(
            completed_matches=completed_matches,
            progress=min(1.0, max(0.0, completed_matches / max(1, total))),
            policy_version=int(policy_version),
        )

    @staticmethod
    def state_dict():
        return {}
