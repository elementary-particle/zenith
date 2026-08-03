#!/usr/bin/env bash
set -euo pipefail

experiment_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${experiment_dir}/../../.." && pwd)"
final_dir="${experiment_dir}/results/privileged_current_kyoku_qboost"
staging_dir="${experiment_dir}/results/.privileged_current_kyoku_qboost-staging"

rm -rf "${staging_dir}"
mkdir -p "${staging_dir}"

cd "${repo_root}"
.venv/bin/pytest -q \
  "${experiment_dir}/test_current_kyoku_qboost.py" \
  "${experiment_dir}/test_privileged_critic.py"
.venv/bin/python \
  "${experiment_dir}/audit_privileged_current_kyoku_qboost.py" \
  --output "${staging_dir}/report.json" \
  "$@"

rm -rf "${final_dir}"
mv "${staging_dir}" "${final_dir}"
