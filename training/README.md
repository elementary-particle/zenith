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

CUDA profiles reject a debug-mode native extension because native environment and hand-analysis
work is on the training critical path. Rebuild with the release command above after Rust changes.

For CUDA, install `requirements-cuda.lock` from the compatible PyTorch CUDA index. The `cuda-strict`
and `cuda-production` profiles fail when CUDA is unavailable; they never silently use CPU. PyTorch
(BSD-style) supplies tensors/autograd/CUDA, NumPy (BSD-3-Clause) supplies host arrays, OpenSkill (MIT)
supplies multiplayer Plackett-Luce ratings, and TensorBoard (Apache-2.0) supplies the derived dashboard.

## Interfaces

Each native call returns immutable `State`, `Event`, `Decision`, `Action`, and
`Transition` values. `Transition.as_numpy()` provides an optional read-only bulk projection without
caller-owned buffers. State contains current gameplay and legal actions; events are a minimal
MJAI-compatible chronological delta; `riichi.analyze_hands` supplies shanten families and improving
tile-type masks on demand. Actor observations are always projected to the ordinary public view.

The actor sequence is ordered as match state and summary, current-kyoku history and tactical state,
kyoku summary, and actor query. Legal actions cross-attend to that public sequence and then interact
through a permutation-equivariant candidate block. A fully separate dealer-canonical oracle consumes
public state, all four concealed-hand counts, and aggregate live-wall counts; it never consumes wall
order or the selected action. Eight task/seat queries feed a scaled-MSE score head and categorical
placement head. Numeric magnitudes use declared FP32 Fourier features.
The default actor has six width-256 causal GQA layers and the position-free bidirectional oracle has
four width-256 layers. PPO uses explicit temporal links, GAE, a clipped policy objective, reproducibly shuffled
packed microbatches, belief supervision, gradient clipping, and finite gates. Each epoch accumulates a
single row-weighted empirical-objective gradient across the variable-size token microbatches, so long
histories do not give a decision extra optimizer influence. Actor and oracle use disjoint AdamW
optimizers. Actor KL stopping never truncates the oracle's four critic epochs.

Training rewards contain outcomes only. Kyoku reward is the rules-settled score delta in thousands of
points. Final placement reward is `(1, 1/3, -1/3, -1)`. Rank blending cannot begin until both 60% of
the 46,000-match budget has elapsed and discard guidance has reached zero; it then ramps over 15% of
the budget. Score and rank GAE are computed
independently over the same match-boundary trajectory, combined with the current curriculum weights,
and normalized once for the policy objective. Both value heads remain supervised throughout training.
Every rollout batch contains complete matches under frozen seat policies. GAE follows each learner seat
across kyoku and ends only at the true match terminal; production has no truncation or bootstrap samples.

Temporary public-information teachers guide discard tile choice, legal wins, reaction restraint, and riichi at
full coefficients 0.50, 0.15, and 0.10, with 0.025 reaction entropy. Five valid batches at or below
15% worse-shanten begin a 10,000-qualified-match taper; 15–18% pauses it and two batches above 18%
restore full guidance. Discard targets use
post-discard shanten and visible ukeire, with ordinary and
riichi logits for the same tile aggregated by `logsumexp`. Calls require strict shanten improvement plus
guaranteed yakuhai or open tanyao; uncertain calls default to pass. A reaction-only entropy term follows
the same competence scale. Legal Ron and Tsumo actions receive unconditional win targets. Targets are
transient encoded rollout data and add no checkpoint tensors.
Hidden belief targets likewise remain ephemeral and never enter checkpoints, logs, or histograms.

Two learner seats face either two instances of the deterministic conservative bot or two instances of
the newest admitted checkpoint, selected once per match from the opponent RNG stream. Bot exposure is
`0.05 + 0.45 * (1 - guidance_scale)`. Bot rows bypass neural residency and are never PPO-eligible. The conservative bot takes wins,
uses multi-riichi genbutsu when available, otherwise follows the discard teacher, declares legal riichi,
and calls only under the same supported-yaku rule.

Official evaluation is isolated from self-play. It uses held-out seeds in four-game cyclic seat blocks,
forces ordinary actor visibility and the final rank objective, and quarantines games without a valid
four-seat terminal placement. Valid games update the official Plackett-Luce ratings in canonical
`(series_id, game_id)` order with the configured OpenSkill Plackett-Luce prior. The immutable outcome
ledger is sufficient to recompute ratings; leaderboard rows report checkpoint ID, `mu`,
`sigma`, `mu - 3*sigma`, placements, games, last series, and provisional status. Exploratory self-play
ratings remain separate from official evaluation.

Training rollouts use the live policy in all four seats. A small curriculum-controlled fraction instead
uses two rotating live-policy seats against two deterministic conservative-bot seats. Those probe games
produce a direct pairwise placement win rate and Wilson confidence interval; rollout progress does not
create checkpoint identities or use OpenSkill. TensorBoard reports that cumulative bot-relative rate,
its interval, and the supporting match/comparison counts. Match telemetry reports the update-local
mean first- and fourth-place final scores in thousands of points and mean completed kyoku per match.
Learner open wins, closed wins, post-riichi deal-ins, exhaustive ryukyoku, calls, improving calls,
riichi opportunities, declarations, and ryukyoku tenpai score contribution are normalized per completed
kyoku; per-kyoku series are omitted on updates with no completed kyoku rather than emitting a fake zero.
Low-value boundary, packing, and duplicate optimizer-row metrics are not part of the
public metric set.

## Run layout and durability

Each run owns a manifest, resolved configuration, dependency/host metadata, append-only canonical JSONL
metrics, evaluation outcomes, independently checksummed atomic checkpoints, and one stable TensorBoard
run directory. Resumes add event files to that logical run and purge abandoned tail steps. Canonical
metrics commit before TensorBoard projection. TensorBoard
failure degrades to canonical-only operation at a safe boundary. Histograms are opt-in, deterministically
sampled, and may never contain concealed tiles, wall data, RNG state, or hidden-target payloads.

A native control-flow failure writes `diagnostics/native-env/stall-*.json` before aborting. The artifact
contains the master seed, schema versions, stalled state metadata, last submitted actions, recent events,
and hex-encoded native snapshots. Restore `bytes.fromhex(payload["snapshots_hex"][env_id])` with
`Env.restore` to replay the exact state; diagnostic snapshots are stabilized before they are exposed.

Metric names are registered with exactly one logical axis, unit, window, and reduction. Canonical
sorted JSONL is committed and synced first; TensorBoard receives the same double-precision scalar
values asynchronously in the stable run directory. Resume creates a distinct writer session while
retaining the logical run identity and monotonic canonical steps. Histogram cadence, element count, and bytes
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
`curriculum.total_matches`, collecting 32 complete matches per update (and a smaller final batch). It admits immutable policy
checkpoints, samples fixed self-play lineups, batches resident historical inference, evaluates/rates
eligible quartets at the configured cadence, prunes unreferenced recovery artifacts, and finishes with
a durable checkpoint. `SIGINT`/`SIGTERM` request a safe-boundary checkpoint and controlled exit.

For a bounded allocation or scheduler job, add `--max-updates N`. This records status `stopped` rather
than `completed`; continue the same run directory with `--resume runs/experiment/checkpoints`. The
one-update behavior remains available only through `zenith_ppo.cli.smoke run` for acceptance testing.
It uses the same completed-match learning-rate and entropy schedules as production while intentionally
omitting production population and evaluation lifecycle work.

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

The CPU smoke uses one complete match. The CUDA-production
profile uses 32 environments, a six-layer width-256 model, BF16 token-budgeted SDPA inference,
32 complete matches per update, and 65,536-token PPO microbatches. Each update still computes an
exact row-weighted full-rollout gradient, but the live policy advances four times as often per match.
