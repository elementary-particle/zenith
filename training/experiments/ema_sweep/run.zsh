#!/usr/bin/env zsh
set -euo pipefail

experiment_dir=${0:A:h}
repository_root=${experiment_dir:h:h:h}
cd "$repository_root"

arms=(
  w010-h032768
  w003-h032768
  w030-h032768
  w010-h008192
  w010-h131072
  w003-h008192
  w030-h008192
  w003-h131072
  w030-h131072
)

for arm in $arms; do
  output="runs/ema-sweep/$arm"
  if [[ -e "$output" ]]; then
    print -u2 "refusing to overwrite existing sweep arm: $output"
    exit 1
  fi
  env PYTHONPATH=training/src .venv/bin/python -m zenith_ppo.cli.train \
    --config "training/configs/ema-sweep-$arm.toml" \
    --output "$output" \
    --initial-checkpoint runs/behavior-cloning-rank-v/checkpoints
done
