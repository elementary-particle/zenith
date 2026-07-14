"""Strict TOML configuration resolution."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import tomllib
from typing import Any

from .compatibility import CompatibilitySet

GROUPS = {"run", "env", "rollout", "observation", "encoding", "model", "ppo",
          "curriculum", "population", "evaluation", "rating", "checkpoint", "metrics"}
OPERATIONAL_OVERRIDES = {"run.output_root", "checkpoint.cadence_updates",
                         "evaluation.cadence_updates", "metrics.progress_every_updates"}
REDACT_WORDS = ("secret", "password", "token", "credential")
EXPECTED_KEYS = {
    "run": {"name", "output_root", "profile", "seed"},
    "env": {"state_schema", "event_schema", "hand_analysis_schema", "snapshot_schema",
        "rules_profile", "rules_profile_id", "rng_profile_id", "num_envs", "num_threads",
        "privileged"},
    "rollout": {"learner_decisions_per_update", "max_frames_per_call", "context_overflow",
        "complete_kyoku_per_env"},
    "observation": {"critic_mode", "mask_schema"},
    "encoding": {"token_schema", "action_schema", "context_tokens", "packing_max_waste"},
    "model": {"schema", "layers", "d_model", "query_heads", "kv_heads", "head_dim", "ffn_dim",
        "dropout", "norm", "position", "critic_layers"},
    "ppo": {"gamma", "gae_lambda", "ratio_clip", "target_kl", "epochs", "token_budget",
        "learning_rate", "adam_beta1", "adam_beta2", "adam_epsilon", "weight_decay",
        "warmup_fraction", "value_loss", "value_coefficient", "entropy_start", "entropy_end",
        "max_grad_norm", "belief_coefficient", "belief_tenpai_coefficient"},
    "curriculum": {"schema", "total_updates", "discard_only_end", "discard_kyoku_blend_end",
        "kyoku_only_end", "kyoku_rank_blend_end"},
    "population": {"sampler", "learner_seats", "shortage", "admit_every_updates",
        "resident_historical_models", "resident_bytes", "retained_checkpoints_min",
        "retained_checkpoints_max", "checkpoint_cohort_size"},
    "evaluation": {"cadence_updates", "seat_block_games", "held_out_seeds"},
    "rating": {"namespace", "mu", "sigma", "beta", "kappa", "tau", "ordinal_sigma"},
    "checkpoint": {"cadence_updates", "keep"},
    "metrics": {"canonical", "progress_every_updates", "tensorboard"},
}


@dataclass(frozen=True)
class ResolvedConfig:
    source: Path
    values: dict[str, Any]
    canonical_json: str
    digest: str

    def group(self, name: str) -> dict[str, Any]:
        return self.values[name]

    def redacted(self) -> dict[str, Any]:
        def walk(value):
            if isinstance(value, dict):
                return {key: "<redacted>" if any(word in key.lower() for word in REDACT_WORDS)
                        else walk(item) for key, item in value.items()}
            if isinstance(value, list):
                return [walk(item) for item in value]
            return value
        return walk(self.values)

    @property
    def compatibility(self) -> CompatibilitySet:
        env, enc, model = self.values["env"], self.values["encoding"], self.values["model"]
        return CompatibilitySet(state_schema=env["state_schema"], event_schema=env["event_schema"],
            hand_analysis_schema=env["hand_analysis_schema"], snapshot_schema=env["snapshot_schema"],
            rules_profile=env["rules_profile_id"], rng_profile=env["rng_profile_id"],
            token_schema=enc["token_schema"], action_schema=enc["action_schema"],
            model_schema=model["schema"], curriculum_schema=self.values["curriculum"]["schema"])


def load(path: str | Path, *, base: str | Path | None = None) -> ResolvedConfig:
    path = Path(path).resolve()
    values = tomllib.loads(path.read_text(encoding="utf-8"))
    if base is not None:
        values = _merge(tomllib.loads(Path(base).read_text(encoding="utf-8")), values)
    validate(values)
    values = deepcopy(values)
    values["run"]["output_root"] = str((path.parent / values["run"]["output_root"]).resolve()) \
        if not Path(values["run"]["output_root"]).is_absolute() else values["run"]["output_root"]
    canonical = json.dumps(values, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return ResolvedConfig(path, values, canonical, sha256(canonical.encode()).hexdigest())


def validate(values: dict[str, Any]) -> None:
    unknown = set(values) - GROUPS - {"schema_version"}
    if unknown:
        raise ValueError(f"unknown top-level configuration keys: {sorted(unknown)}")
    missing = GROUPS - set(values)
    if missing:
        raise ValueError(f"missing configuration groups: {sorted(missing)}")
    if values.get("schema_version") != 2:
        raise ValueError("configuration schema_version must be 2")
    for group, expected in EXPECTED_KEYS.items():
        unknown_group = set(values[group]) - expected
        if unknown_group:
            raise ValueError(f"unknown {group} keys: {sorted(unknown_group)}")
        missing_group = expected - set(values[group])
        if missing_group:
            raise ValueError(f"missing {group} keys: {sorted(missing_group)}")
    tensorboard = values["metrics"].get("tensorboard", {})
    tensorboard_keys = {"enabled", "scalar_every_updates", "histogram_every_updates",
        "histogram_max_elements", "histogram_max_bytes", "flush_seconds",
        "final_flush_timeout_seconds", "runtime_failure"}
    unknown_tensorboard = set(tensorboard) - tensorboard_keys
    if unknown_tensorboard: raise ValueError(f"unknown metrics.tensorboard keys: {sorted(unknown_tensorboard)}")
    missing_tensorboard = tensorboard_keys - set(tensorboard)
    if missing_tensorboard:
        raise ValueError(f"missing metrics.tensorboard keys: {sorted(missing_tensorboard)}")
    model = values["model"]
    if model["d_model"] != model["query_heads"] * model["head_dim"]:
        raise ValueError("model.d_model must equal query_heads * head_dim")
    if model["query_heads"] % model["kv_heads"]:
        raise ValueError("model.kv_heads must divide query_heads")
    if int(model["critic_layers"]) < 1:
        raise ValueError("model.critic_layers must be positive")
    if not 0 <= float(values["encoding"]["packing_max_waste"]) < 1:
        raise ValueError("encoding.packing_max_waste must be in [0, 1)")
    curriculum = values["curriculum"]
    if int(curriculum["total_updates"]) < 1:
        raise ValueError("curriculum.total_updates must be positive")
    points = [curriculum[key] for key in ("discard_only_end", "discard_kyoku_blend_end",
        "kyoku_only_end", "kyoku_rank_blend_end")]
    if not all(0 <= point <= 1 for point in points) or points != sorted(points):
        raise ValueError("curriculum boundaries must be ordered values in [0,1]")
    if values["observation"]["critic_mode"] not in {"ordinary", "privileged"}:
        raise ValueError("observation.critic_mode is unsupported")
    ppo = values["ppo"]
    if float(ppo["belief_coefficient"]) < 0 or float(ppo["belief_tenpai_coefficient"]) < 0:
        raise ValueError("belief coefficients must be non-negative")
    if (values["observation"]["critic_mode"] == "privileged" or
            float(ppo["belief_coefficient"]) > 0) and not values["env"]["privileged"]:
        raise ValueError("privileged native state is required for belief training or critic")
    if not 1 <= values["population"]["learner_seats"] <= 3:
        raise ValueError("population.learner_seats must be in 1..3")
    population = values["population"]
    if population["sampler"] != "uniform_without_replacement_v1":
        raise ValueError("population.sampler is unsupported")
    if population["shortage"] not in {"all_current_bootstrap", "error"}:
        raise ValueError("population.shortage is unsupported")
    if population["retained_checkpoints_min"] < 0 or (
        population["retained_checkpoints_max"] < population["retained_checkpoints_min"]
    ):
        raise ValueError("population retention bounds are invalid")
    if population["resident_historical_models"] < 1 or population["resident_bytes"] < 1:
        raise ValueError("population residency limits must be positive")
    needed_opponents = 4 - int(population["learner_seats"])
    if int(population["checkpoint_cohort_size"]) < needed_opponents:
        raise ValueError(
            "population.checkpoint_cohort_size must cover distinct opponent seats"
        )
    if not isinstance(values["rollout"]["complete_kyoku_per_env"], bool):
        raise ValueError("rollout.complete_kyoku_per_env must be boolean")
    for path, value in (
        ("population.admit_every_updates", population["admit_every_updates"]),
        ("evaluation.cadence_updates", values["evaluation"]["cadence_updates"]),
        ("checkpoint.cadence_updates", values["checkpoint"]["cadence_updates"]),
        ("checkpoint.keep", values["checkpoint"]["keep"]),
        ("metrics.progress_every_updates", values["metrics"]["progress_every_updates"]),
        ("env.num_envs", values["env"]["num_envs"]),
        ("env.num_threads", values["env"]["num_threads"]),
        (
            "rollout.learner_decisions_per_update",
            values["rollout"]["learner_decisions_per_update"],
        ),
    ):
        if int(value) < 1:
            raise ValueError(f"{path} must be positive")
    if values["run"]["profile"] not in {"cpu-smoke", "cuda-strict", "cuda-production"}:
        raise ValueError("run.profile is invalid")
    env = values["env"]
    expected = CompatibilitySet()
    actual = CompatibilitySet(state_schema=env["state_schema"], event_schema=env["event_schema"],
        hand_analysis_schema=env["hand_analysis_schema"], snapshot_schema=env["snapshot_schema"],
        rules_profile=env["rules_profile_id"], rng_profile=env["rng_profile_id"],
        token_schema=values["encoding"]["token_schema"], action_schema=values["encoding"]["action_schema"],
        model_schema=model["schema"], curriculum_schema=curriculum["schema"])
    expected.require(actual, context="resolved configuration")


def _merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result
