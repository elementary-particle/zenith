# Zenith training

`zenith-ppo` trains one public-information riichi policy on the native `riichi`
environment. Production has two stages:

1. behavior cloning initializes the policy and match-boundary rank critic;
2. pure self-play PPO improves the complete policy with current-kyoku
   rank-potential advantages from a match-boundary rank critic.

There is no production DQN, Q-greedy, replay, or CQL path.

## Installation

Build the native environment, then install one pinned Python profile:

```bash
maturin develop --manifest-path riichi/Cargo.toml --release
.venv/bin/python -m pip install -r training/requirements-cpu.lock
.venv/bin/python -m pip install -e training
```

For CUDA, install `training/requirements-cuda.lock` from the matching PyTorch
CUDA index. The CUDA production profile fails rather than silently falling back
to CPU.

## Model

The actor is the width-192 shared-shape action-memory model:

- a three-layer causal transformer encodes public match state and history;
- a small suit-local CNN encodes concealed tile geometry without shanten;
- every legal action attends to strategy, all tile slots, and public history;
- four action-memory blocks apply cross-attention, candidate self-attention,
  and a SwiGLU MLP;
- one canonical tile embedding is shared by events, tile slots, discards,
  calls, and kans.

The critic is disjoint from the actor. Its BC-trained kyoku-boundary tower
predicts one of the 24 possible final seat orders from scores, dealer, round,
honba, riichi sticks, and remaining match structure. The resulting expected
rank utility is an action-independent control variate. No privileged action-Q
network or hidden-wall input is present in the production model.

Both checkpoint formats are explicit and strict:

- `shared-shape-rank-v-bc-v1`
- `shared-shape-emagnet-current-kyoku-ppo-v1`

Older experimental checkpoints are intentionally incompatible.

## Behavior cloning

The BC reader streams original Tenhou-to-MJAI ZIP members, reconstructs the
physical wall, and replays each game through the native Tenhou rules. It does
not write an encoded-decision cache. The actor receives legal-action negative
log likelihood; one sparse row per kyoku also trains the 24-way final-order
critic. Actor and critic use separate learning rates in one AdamW optimizer.

The default profile streams the full 2024 archive once and holds out 200,000
decisions from 2025. Set `behavior_cloning.train_decisions` to a positive value
for a bounded trial; zero means the entire archive.

```bash
.venv/bin/zenith-ppo-train-bc \
  --config training/configs/default.toml \
  --output runs/behavior-cloning
```

Resume without deleting or rebuilding any data:

```bash
.venv/bin/zenith-ppo-train-bc \
  --config training/configs/default.toml \
  --output runs/behavior-cloning \
  --resume runs/behavior-cloning/checkpoints
```

The loader can be tuned with `ZENITH_BC_DATA_WORKERS`,
`ZENITH_BC_REPLAY_BATCH`, and `ZENITH_BC_REPLAY_THREADS`. Defaults leave CPU
headroom while CUDA consumes staged batches.

## PPO

Production PPO is pure self-play current-kyoku rank-V PPO. One live actor
controls all four seats and every genuine decision is eligible. Every action in
a kyoku receives the same predicted change in final-rank utility from the
start of that kyoku to the next kyoku boundary. The exact terminal rank utility
closes the final kyoku. Values are frozen rollout-policy predictions, so the
actor does not backpropagate through the target. This is a kyoku-level policy
gradient rather than an action-boundary GAE trace.

The actor update is fully end to end: the history transformer, tile encoder,
action-memory blocks, embeddings, and policy head all share one optimizer step.
Production retains one complete 2,048-match logical batch, normalizes its raw
advantages globally, and replays it for two actor epochs with one full
logical-batch optimizer group per epoch. It clips the ratio at 0.10 and centers
the adaptive KL controller at `2e-4`. Smaller physical batches use the exact
sufficient-gradient streaming fallback, which supports one actor epoch. The
critic accumulates sparse final-order supervision from kyoku boundaries before
its single optimizer step. An update is transactional and rolls back both
optimizers if replay KL, post-update KL, gradients, or parameters are invalid.

Policy regularization uses EMAgnet: a detached exponential moving average of
the actor defines an adaptive forward-KL target over the complete legal-action
distribution. Its decay is configured as a match-count half-life, so rollout
batch size does not change the time scale. This preserves support on strategies
the policy has found viable without pulling uniformly toward dangerous or
otherwise dominated discards. A fixed `entropy_floor = 1e-4` retains minimal
within-family support recovery; the former feedback-controlled entropy target
is not part of production. PPO clipping and rollout-policy KL checks remain as
separate update-safety mechanisms. Exact checkpoints include the EMA actor.

```bash
.venv/bin/zenith-ppo-train \
  --config training/configs/default.toml \
  --output runs/ppo \
  --initial-checkpoint runs/behavior-cloning-rank-v/checkpoints
```

Exact resume:

```bash
.venv/bin/zenith-ppo-train \
  --config training/configs/default.toml \
  --output runs/ppo \
  --resume runs/ppo/checkpoints
```

Evaluation compares PPO with the immutable BC initialization on held-out cyclic
seat rotations. Production evaluations use 256 held-out seed blocks and four
rotations (1,024 games) in one 1,024-environment inference batch. At 16k, 32k,
64k, 128k, and 262k matches, the evaluator also measures restricted unilateral
exploitability witnesses: one greedy-current, BC, or log-spaced earlier-policy
seat against three sampled-current seats. These are lower bounds from a fixed
challenger set, not approximate best responses or NashConv. A longer run only
reduces the measured exploitability when the maximum challenger advantage
trends toward a 50% pairwise rate and zero score/placement difference.
TensorBoard contains policy/magnet-KL/entropy/gradient signals,
boundary-rank calibration, current-kyoku advantage scale, and gameplay outcomes
such as win, deal-in, riichi, call, tsumo, dama, exhaustive draw, bankruptcy,
point value, and win timing.

## MJAI inference

Both current BC and PPO checkpoints can be served through the same public-state
encoder:

```bash
.venv/bin/zenith-mjai-bot \
  --config training/configs/default.toml \
  --checkpoint runs/ppo/checkpoints
```

The MJAI bridge maintains match/kyoku context, concealed hand, rivers, melds,
pending riichi state, and action phase. It loads model state strictly; it has no
legacy architecture aliases.

## Validation

```bash
.venv/bin/ruff check training/src training/tests
.venv/bin/pytest -q training/tests
```
