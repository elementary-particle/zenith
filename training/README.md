# Zenith training

`zenith-ppo` trains one public-information riichi policy on the native `riichi`
environment. Production has two stages:

1. behavior cloning initializes the policy and match-boundary rank critic;
2. pure self-play PPO improves it with a public per-decision state critic and
   next-boundary rank-potential returns.

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

The sole production actor is the verified width-256 referenced workspace:

- a three-layer causal transformer encodes public kyoku state and history after
  removing the fixed 11-token match prefix;
- a suit-equivariant 34-tile workspace combines exact public counts,
  concealed-hand geometry, and object-referenced causal history;
- legal actions reference affected tile objects and reason as a
  permutation-equivariant set;
- four shared candidate blocks condition tactical candidates on four
  actor-relative score/player tokens and one match-horizon token;
- selected-action hand-outcome, score-delta, and placement prospects remain
  BC auxiliary targets, but they are not treated as Q values.

The critics are disjoint from the actor. A structured dealer-relative boundary
tower predicts all 24 final seat orders from scores and match progress. A
public per-decision scalar critic reads the detached tactical state and
policy-averaged legal-action set. The resulting baseline is
action-independent. It is a zero-initialized residual
over the boundary value, so verified BC checkpoints retain exactly the same
policy and initially recover the old boundary baseline. There is no privileged
critic, hidden-wall input, action-Q head, or architecture switch.

The production checkpoint format is explicit and strict:

- `verified-public-state-value-ppo-v1`

The preceding verified BC checkpoint ID is accepted only as an initialization
source. Its missing zero residual is migrated explicitly. It cannot be resumed
as an exact PPO checkpoint with the old criticless optimizer state.

## Behavior cloning

The BC reader streams original Tenhou-to-MJAI ZIP members, reconstructs the
physical wall, and replays each game through the native Tenhou rules. It does
not write an encoded-decision cache. The actor receives legal-action negative
log likelihood. Selected actions also ground hand-outcome, bucketed
score-delta, and terminal-placement prospects with coefficients 0.05, 0.05,
and 0.02. One sparse row per kyoku trains the separate 24-way final-order
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

Production PPO is pure self-play public-state actor-critic PPO. One live actor
controls all four seats and every genuine decision is eligible. For decision
state (s_t), the rollout stores the frozen public-state prediction
(V(s_t)). The return is the next kyoku-boundary rank potential, or exact
terminal rank utility for the final kyoku. Within each acting-player kyoku
trace, production applies decision-level GAE with `lambda = 0.90`: successive
TD residuals use the next public decision value and the final residual uses the
boundary return. At `lambda = 1` this telescopes exactly to the former
`R_boundary - V(s_t)` estimator. The critic fits the undistorted boundary
return with direct MSE; its detached features cannot move the verified actor.

The actor update is fully end to end: the history transformer, tile encoder,
candidate blocks, embeddings, and policy head all share one optimizer step.
Production retains one complete 2,048-match logical batch, normalizes its raw
advantages globally, and replays it for two actor epochs with one full
logical-batch optimizer group per epoch. It clips the ratio at 0.10 and centers
the adaptive KL controller at `2e-4`. The actor rate is `2e-4`, selected by
achieved KL and an independent terminal-MC update surrogate rather than an LR
ratio. The first 2,048-match update is critic-only; the actor then warms through
`1e-4` to `2e-4` and stays there instead of following the former arbitrary
linear decay. Smaller physical batches use the exact sufficient-gradient
streaming fallback, which requires one actor epoch and one critic pass. The
critic combines dense per-decision value supervision with sparse final-order
supervision from kyoku boundaries. Its `5e-4` rate and four dense MSE passes
were selected independently on held-out value error; sparse boundary-order
supervision is applied once. An update is transactional and rolls back both
optimizers if replay KL, post-update KL, gradients, or parameters are invalid.

Policy regularization uses EMAgnet: a detached exponential moving average of
the actor defines an adaptive forward-KL target over the complete legal-action
distribution. Its decay is configured as a match-count half-life, so rollout
batch size does not change the time scale. This preserves support on strategies
the policy has found viable without pulling uniformly toward dangerous or
otherwise dominated discards. A fixed `entropy_coefficient = 1e-4` retains minimal
support recovery over the complete legal-action distribution, including mass
between strategic choices such as call/pass and riichi/dama; the former
feedback-controlled entropy target is not part of production. PPO clipping and
rollout-policy KL checks remain separate update-safety mechanisms. Exact
checkpoints include the EMA actor.

```bash
.venv/bin/zenith-ppo-train \
  --config training/configs/default.toml \
  --output runs/ppo \
  --initial-checkpoint runs/verified-bc/checkpoints
```

Exact resume:

```bash
.venv/bin/zenith-ppo-train \
  --config training/configs/default.toml \
  --output runs/ppo \
  --resume runs/ppo/checkpoints
```

For a controlled hyperparameter branch, retain the model, optimizer, EMA,
random streams, and environment boundary while writing into a new run:

```bash
.venv/bin/zenith-ppo-train \
  --config training/configs/branch.toml \
  --output runs/ppo-branch \
  --resume runs/ppo/checkpoints \
  --branch-resume
```

Branch checkpoints record `numerical_compatible` reproducibility and the exact
source checkpoint. This differs from `--weights-only`, which deliberately
resets optimizer, EMA, counters, random streams, and environments.

Evaluation compares PPO with the immutable BC initialization on held-out cyclic
seat rotations. Production evaluations use 256 held-out seed blocks and four
rotations (1,024 games) in one 1,024-environment inference batch. At 16k, 32k,
64k, 128k, and 262k matches, the evaluator also measures restricted unilateral
exploitability witnesses: one greedy-current, BC, or log-spaced earlier-policy
seat against three sampled-current seats. These are lower bounds from a fixed
challenger set, not approximate best responses or NashConv. The primary statistic
is the challenger's average rank advantage (reference placement minus challenger
placement); pairwise win rate and score difference remain secondary diagnostics.
A longer run only reduces the measured exploitability when the maximum challenger
rank advantage trends toward zero.
Checkpoint-league runs additionally evaluate the live learner in one rotating
seat against three seats of every configured frozen checkpoint.  The resulting
`frozen-response-*.json` reports call this an achieved held-out response gain:
it is the right curve for selecting a challenger checkpoint, but it is not a
certified best response because optimization can still stop in a local basin.
TensorBoard contains policy/magnet-KL/entropy/gradient signals,
boundary/state-value calibration, advantage scale, and gameplay outcomes
such as win, deal-in, riichi, call, tsumo, dama, exhaustive draw, bankruptcy,
point value, and win timing.

## MJAI inference

Both current BC and PPO checkpoints can be served through the same public-state
encoder:

```bash
.venv/bin/zenith-mjai-bot \
  --config training/configs/default.toml \
  --checkpoint runs/ppo/checkpoints \
  --temperature 0.7
```

The MJAI inference temperature is runtime-only: `0` is greedy (and remains the
default), `1` samples the model distribution, and values between them sharpen
that distribution; values above `1` flatten it. Forced riichi follow-up
discards are never resampled.

The MJAI bridge maintains match/kyoku context, concealed hand, rivers, melds,
pending riichi state, and action phase. It serves the same sole verified actor;
the critic is ignored for action selection.

### Remote Akagi client

Akagi runs external bots as local JSONL subprocesses. When Akagi and Zenith are
on different hosts, run the model server here and install the lightweight relay
from `integrations/akagi/zenith-remote` on the Akagi machine.

On the Zenith server, set a bearer token and start one long-lived checkpoint
process:

```bash
export ZENITH_AKAGI_TOKEN='replace-with-a-long-random-token'
.venv/bin/python -m zenith_ppo.cli.akagi_server \
  --config training/configs/default.toml \
  --checkpoint runs/ppo/checkpoints \
  --host 0.0.0.0 \
  --port 8765 \
  --device cuda \
  --bf16
```

Copy the relay directory to `<akagi>/mjai_bot/zenith-remote`. In Akagi's Bots
page, install its environment, set `server_url` and the same `api_token`, then
activate it for 4-player games. Use a `wss://` URL with `--tls-cert` and
`--tls-key` on the public internet. Plain `ws://` should be limited to a trusted
private VPN or SSH tunnel.

The server shares immutable model weights across connections but creates an
independent live MJAI state per game. A relay that loses its connection returns
`none` and stays failed closed until the next `start_game`; it never reconnects
an in-progress suffix as a fresh game.

Each policy response also includes Akagi's optional `meta.show` payload with
the model's top three legal actions and policy probabilities. The raw
`meta.policy` diagnostics include entropy, confidence margin, selected rank,
inference latency, the action-independent state baseline, and auxiliary
hand-outcome, score-delta, and placement projections. `meta.state` reports the
remaining live wall, tenpai/waits, and scores. The auxiliary projections are BC
prospects for inspection, not counterfactual action Q-values.
`meta.policy.selection_mode` is `greedy` at the default `--temperature 0` and
`sampled` only when the server is explicitly started with a positive
temperature.

## Validation

```bash
.venv/bin/ruff check training/src training/tests
.venv/bin/pytest -q training/tests
```
