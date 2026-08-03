"""Summarize preserved Zenith Q-boosting and current-kyoku gradient audits."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _gradient(row, estimator, kind="raw_gradient"):
    value = row["estimators"][estimator][kind]
    estimate = value["estimate"]
    return {
        "gradient_noise_trace": estimate["gradient_noise_trace"],
        "debiased_gradient_signal_squared": estimate[
            "debiased_gradient_signal_squared"
        ],
        "critical_batch_matches": estimate["critical_batch_matches"],
        "split_half_gradient_cosine": estimate["split_half_gradient_cosine"],
        "terminal_gradient_cosine": value["terminal_gradient_cosine"],
        "terminal_gradient_relative_error_norm": value[
            "terminal_gradient_relative_error_norm"
        ],
        "systematic_relative_error_estimate": value[
            "systematic_relative_error_estimate"
        ],
    }


def _snr_at_batch(critical_batch, matches=4096):
    return math.sqrt(matches / critical_batch) if critical_batch else None


def _only_pair(evaluation):
    pairs = evaluation["paired_bootstrap"]
    if len(pairs) != 1:
        raise ValueError("expected exactly one paired evaluation comparison")
    return next(iter(pairs.values()))


def extract(repo_root: Path):
    evaluation_root = repo_root / "runs" / "evaluations"
    q0 = _load(evaluation_root / "vrpo-v-vgae-update0-4096" / "report.json")
    q16 = _load(evaluation_root / "vrpo-v-vgae-update16-4096" / "report.json")
    current_report = _load(
        evaluation_root
        / "current-kyoku-vrpo-lambda-one-update0-4096"
        / "report.json"
    )
    current_last = current_report["history"][-1]
    current_raw = current_last["current_kyoku"]["raw_gradient"]
    current_standardized = current_last["current_kyoku"]["standardized_gradient"]

    current = {
        "matches": current_report["status"]["matches"],
        "status": current_report["status"]["phase"],
        "target": current_last["current_kyoku"]["target_comparison"],
        "raw_gradient": {
            "gradient_noise_trace": current_raw["estimate"]["gradient_noise_trace"],
            "debiased_gradient_signal_squared": current_raw["estimate"][
                "debiased_gradient_signal_squared"
            ],
            "critical_batch_matches": current_raw["estimate"][
                "critical_batch_matches"
            ],
            "split_half_gradient_cosine": current_raw["estimate"][
                "split_half_gradient_cosine"
            ],
            "terminal_gradient_cosine": current_raw["terminal_gradient_cosine"],
            "terminal_gradient_relative_error_norm": current_raw[
                "terminal_gradient_relative_error_norm"
            ],
            "paired_error_debiased_signal_squared": current_raw["gradient_error"][
                "debiased_gradient_signal_squared"
            ],
        },
        "standardized_gradient": {
            "gradient_noise_trace": current_standardized["estimate"][
                "gradient_noise_trace"
            ],
            "debiased_gradient_signal_squared": current_standardized["estimate"][
                "debiased_gradient_signal_squared"
            ],
            "critical_batch_matches": current_standardized["estimate"][
                "critical_batch_matches"
            ],
            "split_half_gradient_cosine": current_standardized["estimate"][
                "split_half_gradient_cosine"
            ],
            "terminal_gradient_cosine": current_standardized[
                "terminal_gradient_cosine"
            ],
        },
    }

    qboost = {}
    for label, report in (("update_0", q0), ("update_16", q16)):
        last = report["history"][-1]
        qboost[label] = {
            "matches": report["status"]["matches"],
            "target": last["estimators"]["vrpo_q"]["target_comparison"],
            "raw_gradient": _gradient(last, "vrpo_q", "raw_gradient"),
            "standardized_gradient": _gradient(
                last, "vrpo_q", "standardized_gradient"
            ),
        }
    historical_lambda_one = current_report["historical_vrpo_lambda_one"]
    qboost["lambda_one_update_0"] = {
        "matches": historical_lambda_one["matches"],
        "raw_gradient": historical_lambda_one["raw_gradient"],
        "standardized_gradient": historical_lambda_one["standardized_gradient"],
    }

    production_metrics = []
    metrics_path = (
        repo_root
        / "runs"
        / "ppo-vrpo-emagnet-streaming-bc2024"
        / "metrics"
        / "canonical.jsonl"
    )
    wanted = {
        "critic/q_mse",
        "rollout/q_boost_advantage_variance",
        "ppo/actor_gradient_norm",
    }
    by_step = {}
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["name"] in wanted:
            by_step.setdefault(int(row["step"]), {})[row["name"]] = row["value"]
    for step, values in sorted(by_step.items()):
        production_metrics.append({"matches": step, **values})

    qboost_strength_path = (
        evaluation_root
        / "vrpo-update16-vs-bc-sampled-256"
        / "evaluation.json"
    )
    qboost_strength = _load(qboost_strength_path)
    current_strength_root = (
        repo_root / "runs" / "current-kyoku-strength-reproduction" / "evaluations"
    )
    current_update_1_path = current_strength_root / "update1-vs-parent" / "evaluation.json"
    current_update_2_path = (
        current_strength_root
        / "update2-vs-update1-power-replication"
        / "evaluation.json"
    )
    current_update_1 = _load(current_update_1_path)
    current_update_2 = _load(current_update_2_path)

    current_critical = current["raw_gradient"]["critical_batch_matches"]
    q0_critical = qboost["update_0"]["raw_gradient"]["critical_batch_matches"]
    q16_critical = qboost["update_16"]["raw_gradient"]["critical_batch_matches"]
    q1_critical = qboost["lambda_one_update_0"]["raw_gradient"][
        "critical_batch_matches"
    ]
    return {
        "sources": {
            "current_kyoku": str(
                evaluation_root
                / "current-kyoku-vrpo-lambda-one-update0-4096"
                / "report.json"
            ),
            "q_boost_update_0": str(
                evaluation_root / "vrpo-v-vgae-update0-4096" / "report.json"
            ),
            "q_boost_update_16": str(
                evaluation_root / "vrpo-v-vgae-update16-4096" / "report.json"
            ),
            "production_metrics": str(metrics_path),
            "q_boost_strength": str(qboost_strength_path),
            "current_kyoku_update_1_strength": str(current_update_1_path),
            "current_kyoku_update_2_strength": str(current_update_2_path),
        },
        "protocol_note": (
            "Full 3,675,096-parameter actor gradients; 256-match blocks; "
            "unclipped sampled-action importance-corrected objective. Raw and "
            "per-block standardized advantages were both audited."
        ),
        "current_kyoku": current,
        "q_boost_lambda_0_95": qboost,
        "comparisons": {
            "current_vs_q_boost_update_0_critical_batch_ratio": (
                current_critical / q0_critical
            ),
            "current_vs_q_boost_update_16_critical_batch_ratio": (
                current_critical / q16_critical
            ),
            "current_vs_q_boost_lambda_1_critical_batch_ratio": (
                current_critical / q1_critical
            ),
            "snr_at_4096_matches": {
                "current_kyoku": _snr_at_batch(current_critical),
                "q_boost_lambda_0_95_update_0": _snr_at_batch(q0_critical),
                "q_boost_lambda_0_95_update_16": _snr_at_batch(q16_critical),
                "q_boost_lambda_1_update_0": _snr_at_batch(q1_critical),
            },
        },
        "q_boost_training_metrics": production_metrics,
        "held_out_strength_context": {
            "warning": (
                "These strength runs use different update procedures and budgets; "
                "they are corroborating context, not a causal estimator ablation."
            ),
            "q_boost_update_16_vs_bc": {
                "valid_games": qboost_strength["valid_games"],
                **_only_pair(qboost_strength),
            },
            "current_kyoku_update_1_vs_parent": {
                "valid_games": current_update_1["valid_games"],
                **_only_pair(current_update_1),
            },
            "current_kyoku_update_2_vs_update_1": {
                "valid_games": current_update_2["valid_games"],
                **_only_pair(current_update_2),
            },
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = extract(args.repo_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
