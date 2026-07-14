import pytest
from zenith_ppo.metric_registry import REGISTRY, validate


def test_registry_rejects_wrong_axis_and_dynamic_tags():
    validate("ppo/policy_loss", "update", "scalar", "update", "mean")
    with pytest.raises(ValueError): validate("ppo/policy_loss", "step", "scalar", "update", "mean")
    with pytest.raises(KeyError): validate("unknown/x", "update", "count", "instant", "last")


def test_encoding_metrics_name_mean_length_and_bounded_padding():
    assert "encoding/tokens" not in REGISTRY
    validate("encoding/mean_token_length", "update", "tokens/decision", "update", "mean")
    validate("encoding/padding_fraction", "update", "ratio", "update", "last")


def test_boundary_alignment_and_cohort_metrics_are_registered():
    validate("rollout/kyoku_environment_coverage", "update", "ratio", "update", "last")
    validate("rollout/boundary_aligned", "update", "ratio", "update", "last")
    validate("population/checkpoint_cohort_size", "update", "count", "instant", "last")
    validate("population/inference_model_count", "update", "count", "update", "last")
