#!/usr/bin/env bash
set -euo pipefail

experiment_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${experiment_dir}/../../.." && pwd)"
final_dir="${experiment_dir}/results/current_kyoku_qboost"
staging_dir="${experiment_dir}/results/.current_kyoku_qboost-staging"

# A partial report is diagnostic only. Each invocation removes it, while the
# last complete report remains available until its replacement succeeds.
rm -rf "${staging_dir}"
mkdir -p "${staging_dir}"

cd "${repo_root}"
.venv/bin/pytest -q \
  "${experiment_dir}/test_current_kyoku_qboost.py"
.venv/bin/python "${experiment_dir}/audit_current_kyoku_qboost.py" \
  --output "${staging_dir}/report.json" \
  "$@"

rm -rf "${final_dir}"
mv "${staging_dir}" "${final_dir}"
