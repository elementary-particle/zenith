# Zenith PPO

`zenith-ppo` is the independently packaged Python/PyTorch training layer for the native `riichi`
environment. The Rust crate remains unaware of models, PPO, rewards, curriculum, populations, and metrics.

## Install

Build the environment first, then choose one pinned training profile:

```bash
python -m pip install -r requirements.txt
maturin develop --manifest-path riichi/Cargo.toml --release
python -m pip install -r training/requirements-cpu.lock
python -m pip install -e training
```

For CUDA, install `requirements-cuda.lock` from the compatible PyTorch CUDA index. The `cuda-strict`
and `cuda-production` profiles fail when CUDA is unavailable; they never silently use CPU. PyTorch
(BSD-style) supplies tensors/autograd/CUDA, NumPy (BSD-3-Clause) supplies host arrays, OpenSkill (MIT)
supplies multiplayer Plackett-Luce ratings, and TensorBoard (Apache-2.0) supplies the derived dashboard.

## Contracts

State, event, hand-analysis, snapshot, token, action, model, reward, metric, and checkpoint schemas are
versioned independently. Each native call returns immutable `State`, `Event`, `Decision`, `Action`, and
`Transition` values. `Transition.as_numpy()` provides an optional read-only bulk projection without
caller-owned buffers. State contains current gameplay and legal actions; events are a minimal
MJAI-compatible chronological delta; `riichi.analyze_hands` supplies shanten families and improving
tile-type masks on demand. Actor observations are always projected to the ordinary public view.

The actor sequence is a complete player-safe event prefix, ordinary actor state, and actor query. A
separate private critic consumes the detached actor state, canonical sparse opponent-hand and aggregate
live-wall factors, and a learned value query. Numeric magnitudes use declared FP32 Fourier features.
The default actor has six width-256 GQA layers and the critic has two; both use RMSNorm, RoPE, and
SwiGLU. PPO uses explicit temporal links, GAE, clipped policy loss, Huber value loss, belief supervision,
and finite gates.

The curriculum blends discard regret, per-kyoku point deltas, and terminal rank reward. Historical checkpoints are immutable and sampled uniformly
without replacement by default; only bound current-policy rows enter PPO. Official evaluation uses
ordinary inputs, rank-only games, cyclic seat rotation, and `official_plackett_luce_v1`.

Discard reward is negative regret against the best unique legal post-discard hand: shanten is compared
first, then structural ukeire measured in remaining public copies. Kyoku reward is the rules-settled
score delta in thousands of points. Final placement reward is `(1, 1/3, -1/3, -1)` and is emitted only
at normal match completion. Adjacent reward phases blend convexly. The active weights and GAE boundary
are frozen into each rollout sample. Hidden targets remain ephemeral rollout data and are never emitted
to checkpoints, metrics, logs, or histograms.

Official evaluation is isolated from self-play. It uses held-out seeds in four-game cyclic seat blocks,
forces ordinary actor visibility and the final rank objective, and quarantines games without a valid
four-seat terminal placement. Valid games update `official_plackett_luce_v1` in canonical
`(series_id, game_id)` order with the configured OpenSkill Plackett-Luce prior. The immutable outcome
ledger is sufficient to recompute ratings; leaderboard rows report checkpoint ID, model schema, `mu`,
`sigma`, `mu - 3*sigma`, placements, games, last series, and provisional status. Exploratory self-play
ratings never enter this namespace.

## Run layout and durability

Each run owns a manifest, resolved configuration, dependency/host metadata, append-only canonical JSONL
metrics, evaluation outcomes, independently checksummed atomic checkpoints, and one TensorBoard
subdirectory per writer session. Canonical metrics commit before TensorBoard projection. TensorBoard
failure degrades to canonical-only operation at a safe boundary. Histograms are opt-in, deterministically
sampled, and may never contain concealed tiles, wall data, RNG state, or hidden-target payloads.

Metric names are registered with exactly one logical axis, unit, window, and reduction. Canonical
sorted JSONL is committed and synced first; TensorBoard receives the same double-precision scalar
values asynchronously in a writer-session-specific directory. Resume creates a distinct session while
retaining the run identity and monotonic canonical steps. Histogram cadence, element count, and bytes
are bounded and sampled from post-update aggregate tensors only. A runtime TensorBoard failure is
reported at the safe update boundary and may degrade to canonical-only logging according to config;
canonical history remains authoritative.

## Commands

See [`specs/002-ppo-training-framework/quickstart.md`](../specs/002-ppo-training-framework/quickstart.md)
for build, smoke, curriculum, resume, CUDA benchmark, TensorBoard verification, evaluation, and
convergence commands. Start with:

```bash
python -m zenith_ppo.cli.smoke capabilities --profile cpu-smoke
python -m zenith_ppo.cli.smoke run --config training/configs/smoke.toml \
  --profile cpu-smoke --output runs/smoke-cpu
python -m zenith_ppo.cli.train --config training/configs/default.toml \
  --output runs/experiment
python -m zenith_ppo.cli.train --config training/configs/default.toml \
  --resume runs/experiment/checkpoints --output runs/experiment
```

`zenith_ppo.cli.train` is the production driver: without a bound it runs through
`curriculum.total_updates`, preserving native matches across PPO updates. It admits immutable policy
checkpoints, samples fixed self-play lineups, batches resident historical inference, evaluates/rates
eligible quartets at the configured cadence, prunes unreferenced recovery artifacts, and finishes with
a durable checkpoint. `SIGINT`/`SIGTERM` request a safe-boundary checkpoint and controlled exit.

For a bounded allocation or scheduler job, add `--max-updates N`. This records status `stopped` rather
than `completed`; continue the same run directory with `--resume runs/experiment/checkpoints`. The
one-update behavior remains available only through `zenith_ppo.cli.smoke run` for acceptance testing.

To attribute a small CUDA update before optimizing, use the synchronized stage profiler in a fresh run
directory:

```bash
python -m zenith_ppo.cli.profile \
  --config training/configs/default.toml \
  --output runs/profile-update \
  --updates 1
```

The command writes `profile.json`. CUDA synchronization deliberately perturbs normal overlap, so this
mode is for stage attribution only and must not be used for training throughput claims.

The validated CPU smoke uses 64 environments and 2,048 decisions. The validated CUDA-production
profile now uses 128 environments, a six-layer width-256 model, BF16 token-budgeted SDPA inference,
8,192 eligible decisions per update, and 65,536-token PPO minibatches. The earlier 4,096-environment
stress run on the current RTX 5090 completed one update at 314.5 decisions/s with 6.59 GB peak allocated
and 6.95 GB peak reserved CUDA memory. Exact results and reproducibility tolerances are recorded in the
feature validation report.
