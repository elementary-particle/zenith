"""Strict TOML configuration resolution."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import tomllib
from typing import Any

GROUPS = {"run", "env", "rollout", "observation", "encoding", "model", "ppo",
          "curriculum", "teacher", "population", "evaluation", "rating", "checkpoint", "metrics"}
OPERATIONAL_OVERRIDES = {"run.output_root", "evaluation.cadence_matches",
                         "metrics.progress_every_matches"}
REDACT_WORDS = ("secret", "password", "token", "credential")
EXPECTED_KEYS = {
    "run": {"name", "output_root", "profile", "seed"},
    "env": {"rules_profile", "num_envs", "num_threads", "privileged"},
    "rollout": {"matches_per_update", "max_frames_per_match", "context_overflow"},
    "observation": {"critic_mode"},
    "encoding": {"context_tokens", "packing_max_waste"},
    "model": {"layers", "d_model", "query_heads", "kv_heads", "head_dim", "ffn_dim",
        "dropout", "norm", "position", "critic_layers"},
    "ppo": {"gamma", "score_gae_lambda", "rank_gae_lambda", "ratio_clip", "target_kl",
        "epochs", "minibatches", "token_budget",
        "learning_rate", "adam_beta1", "adam_beta2", "adam_epsilon", "weight_decay",
        "warmup_fraction", "score_value_scale", "value_clip", "value_coefficient",
        "entropy_start", "entropy_end",
        "max_grad_norm", "belief_coefficient", "belief_tenpai_coefficient",
        },
    "curriculum": {"total_matches", "rank_start_fraction", "rank_ramp_fraction",
        "minimum_discard_rows", "competence_threshold", "pause_threshold",
        "competence_batches", "regression_batches", "taper_matches"},
    "teacher": {"discard_coefficient", "reaction_coefficient", "riichi_coefficient",
        "reaction_entropy_coefficient", "discard_temperature",
        "reaction_pass_target", "reaction_call_target", "reaction_call_pass_target",
        "riichi_target", "dama_target", "supported_yaku"},
    "population": {"retained_checkpoints_max"},
    "evaluation": {"cadence_matches", "checkpoint_matches", "held_out_seeds",
        "diagnostic_seed_start", "diagnostic_seed_count", "seat_rotations"},
    "rating": {"mu", "sigma", "beta", "kappa", "tau", "ordinal_sigma"},
    "checkpoint": {"cadence_matches", "keep"},
    "metrics": {"canonical", "progress_every_matches", "tensorboard"},
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


def load(path: str | Path, *, base: str | Path | None = None) -> ResolvedConfig:
    path = Path(path).resolve()
    values = tomllib.loads(path.read_text(encoding="utf-8"))
    inherited = values.pop("extends", None)
    if inherited is not None:
        if base is not None:
            raise ValueError("configuration cannot specify both extends and base")
        base = path.parent / str(inherited)
    if base is not None:
        values = _merge(tomllib.loads(Path(base).read_text(encoding="utf-8")), values)
    validate(values)
    values = deepcopy(values)
    values["run"]["output_root"] = str((path.parent / values["run"]["output_root"]).resolve()) \
        if not Path(values["run"]["output_root"]).is_absolute() else values["run"]["output_root"]
    canonical = json.dumps(values, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return ResolvedConfig(path, values, canonical, sha256(canonical.encode()).hexdigest())


def validate(values: dict[str, Any]) -> None:
    unknown = set(values) - GROUPS
    if unknown:
        raise ValueError(f"unknown top-level configuration keys: {sorted(unknown)}")
    missing = GROUPS - set(values)
    if missing:
        raise ValueError(f"missing configuration groups: {sorted(missing)}")
    for group, expected in EXPECTED_KEYS.items():
        unknown_group = set(values[group]) - expected
        if unknown_group:
            raise ValueError(f"unknown {group} keys: {sorted(unknown_group)}")
        missing_group = expected - set(values[group])
        if missing_group:
            raise ValueError(f"missing {group} keys: {sorted(missing_group)}")
    tensorboard = values["metrics"].get("tensorboard", {})
    tensorboard_keys = {"enabled", "scalar_every_matches", "histogram_every_matches",
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
    if int(curriculum["total_matches"]) < 1 or int(curriculum["taper_matches"]) < 1:
        raise ValueError("curriculum match budgets must be positive")
    if not 0 <= float(curriculum["rank_start_fraction"]) <= 1:
        raise ValueError("curriculum.rank_start_fraction must be in [0,1]")
    if not 0 < float(curriculum["rank_ramp_fraction"]) <= 1:
        raise ValueError("curriculum.rank_ramp_fraction must be in (0,1]")
    if not 0 <= float(curriculum["competence_threshold"]) <= float(curriculum["pause_threshold"]) <= 1:
        raise ValueError("curriculum competence thresholds are invalid")
    if any(int(curriculum[key]) < 1 for key in
           ("minimum_discard_rows", "competence_batches", "regression_batches")):
        raise ValueError("curriculum gate counts must be positive")
    teacher = values["teacher"]
    for key in ("discard_coefficient", "reaction_coefficient", "riichi_coefficient",
                "reaction_entropy_coefficient"):
        if float(teacher[key]) < 0:
            raise ValueError(f"teacher.{key} must be non-negative")
    if float(teacher["discard_temperature"]) <= 0:
        raise ValueError("teacher.discard_temperature must be positive")
    for key in ("reaction_pass_target", "reaction_call_target",
                "reaction_call_pass_target", "riichi_target", "dama_target"):
        if not 0 <= float(teacher[key]) <= 1:
            raise ValueError(f"teacher.{key} must be in [0,1]")
    if abs(float(teacher["reaction_call_target"]) +
           float(teacher["reaction_call_pass_target"]) - 1.0) > 1e-9:
        raise ValueError("accepted reaction targets must sum to one")
    if abs(float(teacher["riichi_target"]) + float(teacher["dama_target"]) - 1.0) > 1e-9:
        raise ValueError("riichi targets must sum to one")
    supported = tuple(teacher["supported_yaku"])
    if not supported or set(supported) - {"yakuhai", "open_tanyao"}:
        raise ValueError("teacher.supported_yaku must contain only yakuhai/open_tanyao")
    if values["observation"]["critic_mode"] != "privileged":
        raise ValueError("observation.critic_mode must be privileged for oracle training")
    ppo = values["ppo"]
    if float(ppo["gamma"]) != 1.0:
        raise ValueError("ppo.gamma must be 1 for complete choice-only trajectories")
    if not 0 <= float(ppo["score_gae_lambda"]) <= 1:
        raise ValueError("ppo.score_gae_lambda must be in [0,1]")
    if float(ppo["rank_gae_lambda"]) != 1.0:
        raise ValueError("ppo.rank_gae_lambda must equal 1.0")
    if not 0 < float(ppo["ratio_clip"]) < 1:
        raise ValueError("ppo.ratio_clip must be in (0,1)")
    if float(ppo["target_kl"]) <= 0:
        raise ValueError("ppo.target_kl must be positive")
    if any(int(ppo[key]) < 1 for key in ("epochs", "minibatches", "token_budget")):
        raise ValueError("PPO epochs, minibatches, and token budget must be positive")
    if float(ppo["learning_rate"]) <= 0 or float(ppo["adam_epsilon"]) <= 0:
        raise ValueError("PPO learning rate and Adam epsilon must be positive")
    if not all(0 <= float(ppo[key]) < 1 for key in ("adam_beta1", "adam_beta2")):
        raise ValueError("Adam beta values must be in [0,1)")
    if float(ppo["weight_decay"]) < 0:
        raise ValueError("ppo.weight_decay must be non-negative")
    if not 0 <= float(ppo["warmup_fraction"]) <= 1:
        raise ValueError("ppo.warmup_fraction must be in [0,1]")
    if float(ppo["score_value_scale"]) <= 0:
        raise ValueError("ppo.score_value_scale must be positive")
    if float(ppo["value_clip"]) < 0:
        raise ValueError("PPO value clip must be non-negative; zero disables clipping")
    if float(ppo["value_coefficient"]) < 0:
        raise ValueError("PPO value coefficient must be non-negative")
    if float(ppo["max_grad_norm"]) <= 0:
        raise ValueError("ppo.max_grad_norm must be positive")
    if float(ppo["belief_coefficient"]) < 0 or float(ppo["belief_tenpai_coefficient"]) < 0:
        raise ValueError("belief coefficients must be non-negative")
    if float(model["dropout"]) != 0:
        raise ValueError("model.dropout must be zero for exact PPO behavior accounting")
    if not values["env"]["privileged"]:
        raise ValueError("privileged native state is required for belief training or critic")
    population = values["population"]
    if int(population["retained_checkpoints_max"]) < 4:
        raise ValueError("population.retained_checkpoints_max must retain four evaluations")
    for path, value in (
        ("checkpoint.cadence_matches", values["checkpoint"]["cadence_matches"]),
        ("checkpoint.keep", values["checkpoint"]["keep"]),
        ("evaluation.cadence_matches", values["evaluation"]["cadence_matches"]),
        ("metrics.progress_every_matches", values["metrics"]["progress_every_matches"]),
        ("env.num_envs", values["env"]["num_envs"]),
        ("env.num_threads", values["env"]["num_threads"]),
        (
            "rollout.matches_per_update",
            values["rollout"]["matches_per_update"],
        ),
        ("rollout.max_frames_per_match", values["rollout"]["max_frames_per_match"]),
    ):
        if int(value) < 1:
            raise ValueError(f"{path} must be positive")
    if values["run"]["profile"] not in {"cpu-smoke", "cuda-strict", "cuda-production"}:
        raise ValueError("run.profile is invalid")
    evaluation = values["evaluation"]
    if not evaluation["held_out_seeds"]:
        raise ValueError("evaluation.held_out_seeds must not be empty")
    if int(evaluation["diagnostic_seed_start"]) < 0 or int(evaluation["diagnostic_seed_count"]) < 1:
        raise ValueError("diagnostic evaluation seed range is invalid")
    if int(evaluation["seat_rotations"]) != 4:
        raise ValueError("evaluation.seat_rotations must be four")
    checkpoints = [int(matches) for matches in evaluation["checkpoint_matches"]]
    if checkpoints != sorted(set(checkpoints)) or any(matches < 1 for matches in checkpoints):
        raise ValueError("evaluation.checkpoint_matches must be sorted unique positive matches")
def _merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result
