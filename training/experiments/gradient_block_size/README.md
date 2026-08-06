# Gradient block-size and actor-epoch audit

This experiment distinguishes gradient-noise limits from PPO replay/update
limits without changing production training.

The block-size phase collects independent complete-match chunks at every
requested size and normalizes raw current-kyoku advantages over the complete
block. This matches production's per-policy logical-update normalization.

For each block size it reports the full-actor gradient noise trace, debiased
signal, critical batch size, expected SNR, split-half cosine, and direction
relative to the largest-block mean. The gradient contains the policy term only;
EMAgnet, entropy, and adaptive-KL gradients are excluded so the result measures
the sampling quality of the policy-gradient estimator.

The optional epoch phase then replays one fixed rollout through 1, 2, and 4
actor epochs using the real transactional PPO trainer. Each arm starts from the
same model and optimizer checkpoint, but resets the actor parameter-group LR to
the configured base LR (`6e-5` by default). This avoids making the audit vacuous
when a completed checkpoint contains a schedule-decayed near-zero LR. A second
untouched rollout measures the held-out clipped importance objective, KL, and
clipping after the update.

## Production audit

Run from the repository root on CUDA:

```bash
PYTHONPATH=training/src .venv/bin/python \
  training/experiments/gradient_block_size/audit.py \
  --checkpoint runs/behavior-cloning-rank-v/checkpoints \
  --block-sizes 64,128,256,512 \
  --blocks-per-size 8 \
  --actor-epochs 1,2,4 \
  --epoch-audit-matches 512 \
  --output runs/audits/gradient-block-size-final.json
```

This collects 7,680 matches for gradient estimation plus 1,024 matches for the
epoch train/validation pair. The JSON is updated after every completed block,
so an interrupted run retains usable partial evidence.

For a cheap pipeline smoke test, use the smoke configuration and tiny blocks:

```bash
PYTHONPATH=training/src .venv/bin/python \
  training/experiments/gradient_block_size/audit.py \
  --config training/configs/gradient-audit-cpu.toml \
  --checkpoint runs/behavior-cloning-rank-v/checkpoints \
  --block-sizes 1,2 \
  --blocks-per-size 2 \
  --actor-epochs '' \
  --output /tmp/zenith-gradient-block-smoke.json
```

## Interpretation

- If critical batch and mean direction stabilize below 512 matches, batch size
  is unlikely to be the main bottleneck.
- If 512 remains below SNR 1 or its mean direction disagrees with larger
  blocks, increase the logical update size before tuning actor epochs.
- More epochs are supported only when they commit within the KL guardrail and
  improve the untouched rollout objective. Better training-rollout loss alone
  is replay overfitting, not evidence for a production change.
- Repeat the audit at an early, middle, and late PPO checkpoint; gradient noise
  can change as the policy becomes more concentrated.

## Final-checkpoint result (2026-08-05)

The production audit completed on final checkpoint
`c0d6c7d540e044b9636ec0b516597a243ae7c8a08c243b7ae2c7e121c2675323`
with eight independent blocks at each size and all 3,675,096 actor parameters.
The machine-readable report is
`runs/audits/gradient-block-size-final.json`.

| Matches/block | Critical batch | SNR at block | Split-half cosine | Cosine to 512 mean |
|---:|---:|---:|---:|---:|
| 64 | 1,238 | 0.227 | 0.159 | 0.502 |
| 128 | 1,120 | 0.338 | 0.400 | 0.604 |
| 256 | 1,233 | 0.456 | 0.449 | 0.687 |
| 512 | 1,347 | 0.616 | 0.622 | 1.000 |

Noise trace per match was stable across sizes (about 108k--114k), and the
debiased signal estimate was also reasonably stable (about 81--102). The
consistent critical-batch estimates indicate that the current 512-match
logical update is below SNR one at this late policy. Using a representative
critical batch of 1,235 matches predicts SNRs of about 0.91, 1.29, and 1.82 at
1,024, 2,048, and 4,096 matches respectively. This differs materially from the
near-BC audit and demonstrates that batch requirements changed during training.

The retained 512-match replay comparison reset actor LR to `6e-5` while
preserving checkpoint Adam moments:

| Actor epochs | Post-update KL | Held-out clipped-objective change | Held-out KL | Held-out clip fraction |
|---:|---:|---:|---:|---:|
| 1 | 2.30e-5 | +4.28e-5 | 2.30e-5 | 0.000003 |
| 2 | 5.44e-5 | +6.14e-5 | 5.34e-5 | 0.000339 |
| 4 | 1.43e-4 | +5.28e-5 | 1.37e-4 | 0.003038 |

Two epochs are the best-supported local replay setting: they improve the
untouched-rollout surrogate over one epoch while staying below the configured
target at that checkpoint. Four epochs still pass the hard guardrail, but give
back part of the held-out surrogate gain.

The follow-up replay audit at the compatible BC checkpoint
`2581f2db68262d10f020a309c5f5f66064c70fa9e24cb7f513a844c7823d501c`
used one retained 2,048-match train rollout and an independent 2,048-match
validation rollout. At actor LR `6e-5`:

| Actor epochs | Post-update KL | Held-out clipped-objective change | Held-out clip fraction |
|---:|---:|---:|---:|
| 1 | 1.45e-4 | +2.59e-4 | 0.24% |
| 2 | 2.07e-4 | +3.58e-4 | 0.84% |
| 4 | 3.06e-4 | +4.03e-4 | 1.33% |

The second epoch improves the held-out objective by 38% over one epoch at the
BC start and by 43% in the late-policy 512-match audit. Four epochs cost twice
as much as two for only 13% more early held-out gain, and regress the late
held-out objective. Production therefore uses two actor epochs, LR `6e-5`, and
a `2e-4` KL controller target. This is an optimization diagnostic rather than
arena evidence; online rating remains the promotion criterion.

## Physical-batch throughput result (RTX 5090)

Physical rollout size was swept independently of logical batch size using the
production 3x192 model, BF16/SDPA, eager inference, eight environment threads,
seed 700, and the required optimized Rust release build:

| Physical matches | Decisions/s | Matches/hour | Rollout wall time |
|---:|---:|---:|---:|
| 256 | 12,022 | 40,369 | 22.8 s |
| 512 | 13,634 | 46,279 | 39.8 s |
| 1,024 | 15,879 | 53,460 | 69.0 s |
| 2,048 | 21,733 | 73,282 | 100.6 s |

The original 2,048 arm exposed a quadratic completion scan: each terminal and
kyoku event searched every row accumulated by every environment. Per-match row
indices reduce its native time from 45.3 to 25.7 seconds and total rollout time
from 120.6 to 100.6 seconds.

With one physical rollout equal to the logical batch, raw advantages can be
normalized before autograd. The materialized path then uses one ordinary
backward per packed tensor batch instead of the streaming path's sufficient
VJPs. At 256 matches this reduces optimization from 73.6 to 42.2 seconds and
the complete update from 97.0 to 65.0 seconds. A full 2,048-match transaction
also passed: 443.1 seconds total, 342.5 seconds in PPO, 34.1 GiB peak host
memory, and 4.91 GiB peak CUDA-reserved memory. Production therefore uses one
physical 2,048-match batch; 256 remains the quick performance-test size.
Earlier debug-native sweep numbers are not production-comparable.
