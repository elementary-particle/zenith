"""Competence-gated curriculum driven only by completed matches."""

from __future__ import annotations

from ..types import CurriculumSnapshot


class Curriculum:
    def __init__(self, config, state=None):
        self.config = dict(config)
        state = dict(state or {})
        self.guidance_scale = float(state.get("guidance_scale", 1.0))
        self.phase = str(state.get("phase", "full"))
        self.competence_streak = int(state.get("competence_streak", 0))
        self.regression_streak = int(state.get("regression_streak", 0))
        self.taper_matches = int(state.get("taper_matches", 0))
        self.last_valid_rate = state.get("last_valid_rate")
        self.rank_weight = float(state.get("rank_weight", 0.0))
        self.rank_blending_started = bool(state.get("rank_blending_started", False))
        self.rank_frozen = bool(state.get("rank_frozen", False))

    def snapshot(self, completed_matches: int, policy_version: int) -> CurriculumSnapshot:
        completed_matches = int(completed_matches)
        total = int(self.config["total_matches"])
        if self.guidance_scale == 0.0:
            start = int(total * float(self.config.get("rank_start_fraction", 0.60)))
            ramp = max(1, int(total * float(self.config.get("rank_ramp_fraction", 0.15))))
            if completed_matches >= start:
                self.rank_blending_started = True
                self.rank_frozen = False
                self.rank_weight = max(
                    self.rank_weight, min(1.0, (completed_matches - start) / ramp)
                )
        elif self.rank_blending_started:
            self.rank_frozen = True
        target = int(self.config.get("taper_matches", 10_000))
        return CurriculumSnapshot(
            completed_matches=completed_matches,
            progress=min(1.0, max(0.0, completed_matches / max(1, total))),
            weights=(1.0 - self.rank_weight, self.rank_weight),
            policy_version=int(policy_version),
            guidance_phase=self.phase,
            guidance_scale=self.guidance_scale,
            competence_streak=self.competence_streak,
            regression_streak=self.regression_streak,
            last_valid_worse_shanten_rate=self.last_valid_rate,
            taper_matches=self.taper_matches,
            taper_progress=min(1.0, self.taper_matches / max(1, target)),
            bot_fraction=0.05 + 0.45 * (1.0 - self.guidance_scale),
        )

    def observe(self, *, applicable_rows: int, worse_shanten_rate: float,
                completed_matches: int) -> bool:
        if int(applicable_rows) < int(self.config.get("minimum_discard_rows", 4096)):
            return False
        rate = float(worse_shanten_rate)
        self.last_valid_rate = rate
        competent = float(self.config.get("competence_threshold", 0.15))
        pause = float(self.config.get("pause_threshold", 0.18))
        if rate > pause:
            self.competence_streak = 0
            self.regression_streak += 1
            if self.regression_streak >= int(self.config.get("regression_batches", 2)):
                self.phase = "full"
                self.guidance_scale = 1.0
                self.taper_matches = 0
                self.competence_streak = 0
                self.regression_streak = 0
            return True
        self.regression_streak = 0
        if rate > competent:
            return True
        if self.phase == "full":
            self.competence_streak += 1
            if self.competence_streak >= int(self.config.get("competence_batches", 5)):
                self.phase = "taper"
        if self.phase in {"taper", "zero"}:
            target = int(self.config.get("taper_matches", 10_000))
            self.taper_matches = min(target, self.taper_matches + int(completed_matches))
            self.guidance_scale = 1.0 - self.taper_matches / max(1, target)
            if self.taper_matches >= target:
                self.phase = "zero"
                self.guidance_scale = 0.0
        return True

    def state_dict(self):
        return {
            "guidance_scale": self.guidance_scale,
            "phase": self.phase,
            "competence_streak": self.competence_streak,
            "regression_streak": self.regression_streak,
            "taper_matches": self.taper_matches,
            "last_valid_rate": self.last_valid_rate,
            "rank_weight": self.rank_weight,
            "rank_blending_started": self.rank_blending_started,
            "rank_frozen": self.rank_frozen,
        }
