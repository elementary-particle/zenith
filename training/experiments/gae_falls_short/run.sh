#!/usr/bin/env bash
set -euo pipefail

experiment_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${experiment_dir}/../../.." && pwd)"
final_dir="${experiment_dir}/results/final"
staging_dir="${experiment_dir}/results/.staging"

# Each invocation replaces the one obsolete staging attempt. Completed evidence
# remains immutable in final/ until both analyses and tests succeed.
rm -rf "${staging_dir}"
mkdir -p "${staging_dir}"

cd "${repo_root}"
.venv/bin/python "${experiment_dir}/exact_bias_variance.py" \
  --samples 500000 \
  --seed 20260801 \
  --output "${staging_dir}/exact_bias_variance.json"
.venv/bin/python "${experiment_dir}/extract_pipeline_evidence.py" \
  --repo-root "${repo_root}" \
  --output "${staging_dir}/pipeline_evidence.json"
.venv/bin/pytest -q "${experiment_dir}/test_exact_bias_variance.py"

rm -rf "${final_dir}"
mv "${staging_dir}" "${final_dir}"
