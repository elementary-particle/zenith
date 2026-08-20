"""Strict TOML configuration resolution."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import tomllib
from typing import Any

GROUPS = {"run", "env", "rollout", "encoding", "model", "ppo",
          "curriculum", "population",
          "evaluation", "rating", "checkpoint", "metrics", "behavior_cloning"}
OPTIONAL_GROUPS = {"behavior_cloning"}
REDACT_WORDS = ("secret", "password", "token", "credential")
EXPECTED_KEYS = {
    "run": {"output_root", "profile", "seed"},
    "env": {"rules_profile", "num_envs", "num_threads"},
    "rollout": {"matches_per_update"},
    "encoding": {"context_tokens", "packing_max_waste"},
    "model": {
        "layers", "d_model", "query_heads", "kv_heads", "head_dim",
        "ffn_dim", "action_memory_layers", "action_memory_ffn_dim",
        "concealed_shape_channels", "concealed_shape_blocks",
        "boundary_critic_width",
        "ground_board_layers", "structured_boundary_layers",
    },
    "ppo": {"ratio_clip", "target_kl",
        "kl_coefficient_initial", "kl_coefficient_minimum",
        "kl_coefficient_maximum", "kl_adaptation_factor",
        "epochs", "minibatches", "token_budget",
        "actor_learning_rate", "critic_learning_rate",
        "adam_beta1", "adam_beta2", "adam_epsilon", "weight_decay",
        "actor_learning_starts_matches", "actor_warmup_matches",
        "critic_warmup_matches",
        "boundary_rank_coefficient", "critic_epochs",
        "gae_lambda", "value_coefficient",
        "max_grad_norm", "magnet_kl_coefficient",
        "magnet_half_life_matches", "entropy_coefficient",
        },
    "curriculum": {"total_matches"},
    "population": {"retained_checkpoints_max"},
    "evaluation": {"cadence_matches", "checkpoint_matches",
        "held_out_seed_start", "held_out_seed_count",
        "diagnostic_seed_start", "diagnostic_seed_count"},
    "rating": {"mu", "sigma", "beta", "kappa", "tau", "ordinal_sigma"},
    "checkpoint": {"cadence_matches", "keep"},
    "metrics": {"progress_every_matches", "tensorboard"},
    "behavior_cloning": {
        "train_archives", "validation_archives", "train_decisions",
        "validation_decisions", "epochs", "token_budget", "learning_rate",
        "adam_beta1", "adam_beta2", "adam_epsilon", "weight_decay",
        "max_grad_norm", "boundary_rank_learning_rate",
        "boundary_rank_coefficient",
    },
}
OPTIONAL_KEYS = {
    "rollout": {
        "training_mode", "league_checkpoints", "league_uniform_fraction",
        "league_minimum_games", "league_learner_seats",
        "ema_opponent_half_life_matches",
    },
    "evaluation": {"batch_size", "token_budget", "held_out_seeds"},
    "model": {"architecture", "policy_temperature"},
    "curriculum": {"schedule_matches"},
    "behavior_cloning": {
        "label_smoothing", "confidence_penalty_coefficient", "family_weights",
        "outcome_coefficient", "score_delta_coefficient",
        "placement_coefficient",
    },
}


@dataclass(frozen=True)
class ResolvedConfig:
    source: Path
    values: dict[str, Any]
    canonical_json: str
    digest: str

    def redacted(self) -> dict[str, Any]:
        def walk(value):
            if isinstance(value, dict):
                return {key: "<redacted>" if any(word in key.lower() for word in REDACT_WORDS)
                        else walk(item) for key, item in value.items()}
            if isinstance(value, list):
                return [walk(item) for item in value]
            return value
        return walk(self.values)


def evaluation_seeds(evaluation: dict[str, Any]) -> tuple[int, ...]:
    """Resolve explicit smoke seeds or the production held-out seed range."""
    explicit = evaluation.get("held_out_seeds")
    if explicit is not None:
        return tuple(map(int, explicit))
    start = int(evaluation["held_out_seed_start"])
    count = int(evaluation["held_out_seed_count"])
    return tuple(range(start, start + count))


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
    if "behavior_cloning" in values:
        for key in ("train_archives", "validation_archives"):
            values["behavior_cloning"][key] = [
                str((path.parent / archive).resolve())
                if not Path(archive).is_absolute() else str(Path(archive))
                for archive in values["behavior_cloning"][key]
            ]
    if "league_checkpoints" in values["rollout"]:
        values["rollout"]["league_checkpoints"] = [
            str((path.parent / checkpoint).resolve())
            if not Path(checkpoint).is_absolute() else str(Path(checkpoint))
            for checkpoint in values["rollout"]["league_checkpoints"]
        ]
    canonical = json.dumps(values, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return ResolvedConfig(path, values, canonical, sha256(canonical.encode()).hexdigest())


def validate(values: dict[str, Any]) -> None:
    unknown = set(values) - GROUPS
    if unknown:
        raise ValueError(f"unknown top-level configuration keys: {sorted(unknown)}")
    missing = GROUPS - OPTIONAL_GROUPS - set(values)
    if missing:
        raise ValueError(f"missing configuration groups: {sorted(missing)}")
    for group, expected in EXPECTED_KEYS.items():
        if group not in values:
            continue
        optional = OPTIONAL_KEYS.get(group, set())
        unknown_group = set(values[group]) - expected - optional
        if unknown_group:
            raise ValueError(f"unknown {group} keys: {sorted(unknown_group)}")
        missing_group = expected - set(values[group])
        if missing_group:
            raise ValueError(f"missing {group} keys: {sorted(missing_group)}")
    tensorboard = values["metrics"].get("tensorboard", {})
    tensorboard_keys = {"enabled", "histogram_every_matches",
        "histogram_max_elements", "histogram_max_bytes", "flush_seconds",
        "runtime_failure"}
    unknown_tensorboard = set(tensorboard) - tensorboard_keys
    if unknown_tensorboard:
        raise ValueError(
            f"unknown metrics.tensorboard keys: {sorted(unknown_tensorboard)}"
        )
    missing_tensorboard = tensorboard_keys - set(tensorboard)
    if missing_tensorboard:
        raise ValueError(f"missing metrics.tensorboard keys: {sorted(missing_tensorboard)}")
    model = values["model"]
    from .model.factory import actor_critic_architecture_supported
    if not actor_critic_architecture_supported(model.get(
        "architecture", "verified-public-state-value-ppo-v1",
    )):
        raise ValueError("model.architecture is invalid")
    training_mode = values["rollout"].get("training_mode", "pure_self_play")
    if training_mode not in {
        "pure_self_play", "ema_self_play", "adversarial_league",
        "checkpoint_league",
    }:
        raise ValueError("rollout.training_mode is invalid")
    if training_mode == "adversarial_league":
        raise ValueError(
            "PPO requires one trainable policy; use pure_self_play or "
            "checkpoint_league"
        )
    league_checkpoints = values["rollout"].get("league_checkpoints", ())
    if training_mode == "checkpoint_league":
        if not 1 <= len(league_checkpoints) <= 7:
            raise ValueError(
                "checkpoint league requires between one and seven checkpoints"
            )
        if len(set(map(str, league_checkpoints))) != len(league_checkpoints):
            raise ValueError("checkpoint league paths must be unique")
    elif league_checkpoints:
        raise ValueError(
            "rollout.league_checkpoints requires checkpoint_league mode"
        )
    ema_opponent_half_life = values["rollout"].get(
        "ema_opponent_half_life_matches"
    )
    if training_mode == "ema_self_play":
        if ema_opponent_half_life is None \
                or float(ema_opponent_half_life) <= 0:
            raise ValueError(
                "EMA self-play requires a positive "
                "rollout.ema_opponent_half_life_matches"
            )
    elif ema_opponent_half_life is not None:
        raise ValueError(
            "rollout.ema_opponent_half_life_matches requires ema_self_play mode"
        )
    if not 0 <= float(values["rollout"].get(
        "league_uniform_fraction", 0.25
    )) <= 1:
        raise ValueError("rollout.league_uniform_fraction must be in [0,1]")
    if int(values["rollout"].get("league_minimum_games", 8)) < 1:
        raise ValueError("rollout.league_minimum_games must be positive")
    learner_seats = int(values["rollout"].get("league_learner_seats", 2))
    if learner_seats not in (1, 2):
        raise ValueError("rollout.league_learner_seats must be one or two")
    if training_mode != "checkpoint_league" and "league_learner_seats" in values[
        "rollout"
    ]:
        raise ValueError(
            "rollout.league_learner_seats requires checkpoint_league mode"
        )
    if model["d_model"] != model["query_heads"] * model["head_dim"]:
        raise ValueError("model.d_model must equal query_heads * head_dim")
    if model["query_heads"] % model["kv_heads"]:
        raise ValueError("model.kv_heads must divide query_heads")
    if float(model.get("policy_temperature", 1.0)) <= 0:
        raise ValueError("model.policy_temperature must be positive")
    width_keys = {
        "boundary_critic_width",
        "ground_board_layers", "structured_boundary_layers",
        "action_memory_layers", "action_memory_ffn_dim",
        "concealed_shape_channels", "concealed_shape_blocks",
    }
    if any(int(model[key]) < 1 for key in width_keys):
        raise ValueError("model widths and layer counts must be positive")
    if not 0 <= float(values["encoding"]["packing_max_waste"]) < 1:
        raise ValueError("encoding.packing_max_waste must be in [0, 1)")
    curriculum = values["curriculum"]
    if int(curriculum["total_matches"]) < 1:
        raise ValueError("curriculum total_matches must be positive")
    schedule_matches = int(curriculum.get(
        "schedule_matches", curriculum["total_matches"]
    ))
    if schedule_matches < int(curriculum["total_matches"]):
        raise ValueError(
            "curriculum schedule_matches must be at least total_matches"
        )
    ppo = values["ppo"]
    if not 0 < float(ppo["ratio_clip"]) < 1:
        raise ValueError("ppo.ratio_clip must be in (0,1)")
    if float(ppo["target_kl"]) <= 0:
        raise ValueError("ppo.target_kl must be positive")
    if not 0 < float(ppo["kl_coefficient_minimum"]) <= float(
        ppo["kl_coefficient_initial"]
    ) <= float(ppo["kl_coefficient_maximum"]):
        raise ValueError("PPO KL coefficients must satisfy 0 < minimum <= initial <= maximum")
    if float(ppo["kl_adaptation_factor"]) <= 1:
        raise ValueError("ppo.kl_adaptation_factor must be greater than one")
    if any(int(ppo[key]) < 1 for key in ("epochs", "minibatches", "token_budget")):
        raise ValueError("PPO epochs, minibatches, and token budget must be positive")
    if int(ppo["minibatches"]) != 1:
        raise ValueError(
            "PPO requires exactly one full logical-batch optimizer group"
        )
    if (
        int(ppo["epochs"]) != 1
        and int(values["env"]["num_envs"])
        < int(values["rollout"]["matches_per_update"])
    ):
        raise ValueError(
            "multiple PPO actor epochs require a retained logical batch "
            "(env.num_envs >= rollout.matches_per_update)"
        )
    if int(ppo["critic_epochs"]) < 1:
        raise ValueError("ppo.critic_epochs must be positive")
    if (
        int(ppo["critic_epochs"]) != 1
        and int(values["env"]["num_envs"])
        < int(values["rollout"]["matches_per_update"])
    ):
        raise ValueError(
            "multiple PPO critic epochs require a retained logical batch "
            "(env.num_envs >= rollout.matches_per_update)"
        )
    if any(float(ppo[key]) <= 0 for key in (
        "actor_learning_rate", "critic_learning_rate", "adam_epsilon",
    )):
        raise ValueError("PPO actor/critic learning rates and Adam epsilon must be positive")
    if not all(0 <= float(ppo[key]) < 1 for key in ("adam_beta1", "adam_beta2")):
        raise ValueError("Adam beta values must be in [0,1)")
    if float(ppo["weight_decay"]) < 0:
        raise ValueError("ppo.weight_decay must be non-negative")
    if any(int(ppo[key]) < 0 for key in (
        "actor_learning_starts_matches", "actor_warmup_matches",
        "critic_warmup_matches",
    )):
        raise ValueError("PPO learning-start and warmup matches must be non-negative")
    if float(ppo["boundary_rank_coefficient"]) <= 0:
        raise ValueError("PPO boundary-rank coefficient must be positive")
    if not 0 <= float(ppo["gae_lambda"]) <= 1:
        raise ValueError("ppo.gae_lambda must be in [0,1]")
    if float(ppo["max_grad_norm"]) <= 0:
        raise ValueError("ppo.max_grad_norm must be positive")
    if float(ppo["magnet_kl_coefficient"]) <= 0:
        raise ValueError("ppo.magnet_kl_coefficient must be positive")
    if float(ppo["magnet_half_life_matches"]) <= 0:
        raise ValueError("ppo.magnet_half_life_matches must be positive")
    if not 0 <= float(ppo["entropy_coefficient"]) < 1:
        raise ValueError("ppo.entropy_coefficient must be in [0,1)")
    if float(ppo["value_coefficient"]) <= 0:
        raise ValueError("ppo.value_coefficient must be positive")
    if "behavior_cloning" in values:
        bc = values["behavior_cloning"]
        if not bc["train_archives"] or not bc["validation_archives"]:
            raise ValueError("behavior-cloning archive lists must not be empty")
        if int(bc["train_decisions"]) < 0:
            raise ValueError(
                "behavior-cloning train_decisions must be zero (all) or positive"
            )
        if any(int(bc[key]) < 1 for key in (
            "validation_decisions", "epochs", "token_budget",
        )):
            raise ValueError("behavior-cloning counts and budgets must be positive")
        if any(float(bc[key]) <= 0 for key in (
            "learning_rate", "boundary_rank_learning_rate", "adam_epsilon",
        )):
            raise ValueError("behavior-cloning learning rate and epsilon must be positive")
        if not all(0 <= float(bc[key]) < 1 for key in ("adam_beta1", "adam_beta2")):
            raise ValueError("behavior-cloning Adam betas must be in [0,1)")
        if float(bc["weight_decay"]) < 0 or float(bc["max_grad_norm"]) <= 0:
            raise ValueError("behavior-cloning weight decay/norm are invalid")
        if float(bc["boundary_rank_coefficient"]) <= 0:
            raise ValueError(
            "behavior_cloning.boundary_rank_coefficient must be positive"
            )
        if not 0 <= float(bc.get("label_smoothing", 0.0)) < 1:
            raise ValueError(
                "behavior_cloning.label_smoothing must be in [0,1)"
            )
        if float(bc.get("confidence_penalty_coefficient", 0.0)) < 0:
            raise ValueError(
                "behavior_cloning.confidence_penalty_coefficient must be "
                "non-negative"
            )
        if any(float(bc.get(key, 0.0)) < 0 for key in (
            "outcome_coefficient", "score_delta_coefficient",
            "placement_coefficient",
        )):
            raise ValueError(
                "behavior-cloning auxiliary coefficients must be non-negative"
            )
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
    ):
        if int(value) < 1:
            raise ValueError(f"{path} must be positive")
    if values["run"]["profile"] not in {"cpu-smoke", "cuda-strict", "cuda-production"}:
        raise ValueError("run.profile is invalid")
    evaluation = values["evaluation"]
    for prefix in ("held_out", "diagnostic"):
        if (
            int(evaluation[f"{prefix}_seed_start"]) < 0
            or int(evaluation[f"{prefix}_seed_count"]) < 1
        ):
            raise ValueError(f"{prefix} evaluation seed range is invalid")
    explicit_seeds = evaluation.get("held_out_seeds")
    if explicit_seeds is not None and (
        not explicit_seeds
        or len(set(map(int, explicit_seeds))) != len(explicit_seeds)
        or any(int(seed) < 0 for seed in explicit_seeds)
    ):
        raise ValueError(
            "evaluation.held_out_seeds must be unique non-negative seeds"
        )
    if any(int(evaluation.get(key, 1)) < 1 for key in (
        "batch_size", "token_budget",
    )):
        raise ValueError("evaluation batch size and token budget must be positive")
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
