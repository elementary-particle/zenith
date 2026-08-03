# GAE Falls Short: Zenith validation

This folder tests whether the Q-boosting estimator from Fan and Farina's
*GAE Falls Short in Imperfect-Information Self-Play Reinforcement Learning*
can reduce policy-gradient variance without adding bias in Zenith's Mahjong
pipeline.

The short answer for the historical whole-match implementation is **no**.
Q-boosting substantially compressed the raw scalar advantage, but it
did not improve the standardized policy-gradient signal/noise ratio used by
Zenith. The current-kyoku estimator produced a much more coherent full-model
gradient. The paper's no-added-bias guarantee applies at `lambda = 1`; Zenith's
historical Q-boosting run used `lambda = 0.95`, where the guarantee does not
apply.

## Scope and reproducibility

All new code and generated outputs live in this directory. The extractor reads
preserved historical reports under `runs/evaluations/` without modifying them.
`run.sh` uses a staging directory and promotes it only after every analysis and
test succeeds; failed or superseded staging output is deleted on the next run.

Run from the repository root:

```bash
bash training/experiments/gae_falls_short/run.sh
```

Final machine-readable outputs:

- `results/final/exact_bias_variance.json`
- `results/final/pipeline_evidence.json`

The separately gated current-kyoku Q-boosting audit runs with:

```bash
bash training/experiments/gae_falls_short/run_current_kyoku_qboost.sh
```

It cross-fits an action-Q head on independent matches, targets the next kyoku
boundary, resets every trace at `end_kyoku`, and treats `lambda = 1` as the
primary and only estimator. Its promoted output is
`results/current_kyoku_qboost/report.json`; a failed or superseded staging run
is removed automatically.

The paper-style independent privileged-critic audit runs with:

```bash
bash training/experiments/gae_falls_short/run_privileged_current_kyoku_qboost.sh
```

Its critic has separate parameters and consumes dealer-relative concealed
hands, the exact wall/order, public state, acting seat, and legal actions. It
does not consume actor hidden tensors. The promoted output is
`results/privileged_current_kyoku_qboost/report.json`.

The detached actor-hidden ablation runs with:

```bash
bash training/experiments/gae_falls_short/run_privileged_actor_hidden_current_kyoku_qboost.sh
```

It adds the frozen actor's per-decision state and per-legal-action hidden
states to the privileged critic. Critic loss cannot backpropagate through
those tensors or update actor parameters. This is a secondary engineering
ablation, not the paper's separate-encoder architecture. Its promoted output
is `results/privileged_actor_hidden_current_kyoku_qboost/report.json`.

## What is tested

1. `exact_bias_variance.py` uses a finite, four-seat stochastic rank-utility
   game with exact dynamic-programming Q, V, and policy gradient. It validates
   the paper's theorem and sweeps controlled Q-critic error.
2. `extract_pipeline_evidence.py` summarizes Zenith's preserved full-actor
   gradient audits at Q-boosting updates 0 and 16 and the current-kyoku audit.
   The audits use 3,675,096 actor parameters, 256-match blocks, and both raw and
   per-block standardized advantages.
3. `test_exact_bias_variance.py` checks Bellman consistency, the exact-Q
   pathwise result, lambda-one gradient unbiasedness under critic error, and a
   controlled variance reduction case.
4. `audit_current_kyoku_qboost.py` is the previously missing Mahjong test. It
   uses a held-out, frozen public action-Q critic and compares paired full-actor
   gradients against current-kyoku on both raw and per-block standardized
   advantages.
5. `audit_privileged_current_kyoku_qboost.py` runs both centralized-critic
   variants. The `--actor-hidden-states` switch adds detached actor state and
   action representations while preserving critic/actor gradient isolation.

## Interpretation

The controlled experiment validates the paper's narrow theoretical claim:
with exact Q, Q-boosting removes future-action noise pathwise; with an
inaccurate Q critic and `lambda = 1`, conditional advantages can be shifted but
the score-function gradient remains unbiased. At `lambda = 0.95`, critic error
can introduce gradient bias.

In the 500,000-trajectory exact experiment, exact-Q Q-boosting retained only
3.04% of current-kyoku's gradient variance with no detected gradient bias. With
Q sup-norm error 0.15, `lambda = 1` retained 6.31% with no detected bias, while
`lambda = 0.95` retained 3.68% but introduced gradient bias of -0.00335 (33.9
Monte Carlo standard errors, about 1.5% of the exact gradient). This confirms
that the variance reduction is real in the regime covered by the theorem and
that the practical shorter trace changes the bias conclusion.

The real Mahjong evidence does not validate deployment:

- Current-kyoku raw-gradient critical batch size: about 95 matches.
- Q-boosting `lambda = 0.95`: about 3,695 matches at update 0 and 3,000 at
  update 16.
- Q-boosting `lambda = 1`: about 212,511 matches at update 0.
- At a 4,096-match batch, the corresponding raw-gradient SNRs are about 6.57,
  1.05, 1.17, and 0.14.
- Current-kyoku's split-half gradient cosine was 0.951 versus 0.371/0.381 for
  Q-boosting at updates 0/16.
- Q-boosting reduced raw gradient-noise trace, but after the same per-block
  advantage standardization its noise trace was essentially unchanged from the
  terminal reference. Its useful gradient signal fell much more than its noise.
- The final historical Q-boosting actor did not show a statistically resolved
  improvement over BC in 1,024 held-out games (paired win rate 0.504, 95% CI
  0.483--0.526). Separate current-kyoku strength reproductions were positive,
  but used different update procedures and budgets, so they are corroborating
  context rather than a causal head-to-head estimator comparison.

The current-kyoku audit stopped after 3,584 of 4,096 planned matches once the
estimate was stable, so comparisons should retain that qualification. It still
contains fourteen independent 256-match blocks and is decisive on the observed
SNR gap.

### Current-kyoku lambda-one result

The new cross-fitted Mahjong audit used 128 critic-training matches (80,895
policy rows) and eight untouched 64-match gradient blocks (512 matches and
338,289 policy rows). The Q critic predicted the absolute next-boundary rank
potential from frozen public action states. On held-out data it reached 0.293
explained variance and 0.545 target correlation.

That critic did not make Q-boosting a variance reduction:

- Raw advantage variance increased to 4.61 times current-kyoku.
- Raw full-actor gradient noise increased from 4,948 to 28,671 per match; the
  critical batch increased from 171 to 445 matches.
- After the production per-block standardization, noise was essentially equal
  (104,092 versus 105,176), but Q-boosting retained less useful signal. Its
  critical batch was 343 versus 170 matches.
- At the observed 512-match batch, raw gradient SNR was 1.07 for Q-boosting
  versus 1.73 for current-kyoku. Split-half cosine was 0.323 versus 0.612.

The paired gradient-direction difference itself was unresolved at this budget,
which is compatible with the lambda-one unbiasedness result: its inferred
critical batch was about 3,327 matches in raw scale. Thus this run rejects a
variance benefit; it does not find evidence that lambda one added gradient
bias.

This is specifically a **public-Q** result. We therefore tested a new
centralized hidden-state critic retargeted and trained for next-kyoku outcomes,
as reported below. The whole-match centralized critic was not reused because
its horizon and supervision are different.

### Independent privileged-critic result

The centralized experiment trained a separate 643,393-parameter critic on 512
matches in eight memory-bounded chunks, with two critic passes per chunk. It
then used eight disjoint 64-match audit blocks (512 matches and 343,895 policy
rows). The critic saw all concealed hands and the exact wall but no actor
activations or actor-owned parameters.

Privileged information worked as a prediction aid: held-out Q explained
variance rose to 0.688, target correlation to 0.839, and MSE fell to 0.0749.
It also reduced the public-Q failure substantially, but still did not produce
a gradient variance win:

- Raw advantage variance was 1.63 times current-kyoku, down from 4.61 times for
  public Q.
- Raw critical batch was 183 matches for privileged Q-boosting versus 125 for
  current-kyoku; observed-batch SNR was 1.67 versus 2.02.
- Standardized noise traces were nearly equal (101,775 versus 99,282), but the
  Q-boosting signal estimate was lower (556 versus 795). Standardized critical
  batch was therefore 183 versus 125.
- Split-half gradient cosine was 0.555 for Q-boosting versus 0.659 for
  current-kyoku.
- The raw paired gradient-direction difference remained unresolved (SNR 0.25,
  inferred critical batch about 8,136 matches), so the audit found no evidence
  that `lambda = 1` added gradient bias.

The critic's mean absolute action-centering term was only 0.0078 despite its
strong endpoint prediction. This suggests that, over the shortened kyoku
horizon, future sampled own-action value differences are small relative to
the remaining settlement/opponent noise. Better state prediction alone is not
enough; variance reduction requires accurate action-conditioned differences.

### Detached actor-hidden result

The directly matched ablation used the same checkpoint, seed, 512 critic
matches, and eight held-out 64-match blocks. It added both the actor's frozen
decision state and its per-legal-action hidden state to a new 680,737-parameter
privileged critic. Unit tests verify that critic backpropagation creates no
gradient on either actor tensor.

The actor features made the Q predictor stronger but did not improve the
policy-gradient estimator:

- Held-out explained variance improved from 0.688 to 0.748, target correlation
  from 0.839 to 0.867, and MSE from 0.0749 to 0.0615.
- Mean absolute action centering increased from 0.0078 to 0.0127, while the
  Q-boost advantage-variance ratio improved from 1.63 to 1.38 times
  current-kyoku.
- Raw Q-boost gradient noise fell from 7,763 to 6,744 per match, but the
  debiased signal estimate also fell from 42.5 to 36.5. Critical batch was
  therefore 185 matches, slightly worse than 183 without actor states and 125
  for current-kyoku.
- After production standardization, critical batch was 184 matches versus 183
  without actor states and 125 for current-kyoku. Observed-batch SNR was 1.67
  versus 2.02 for current-kyoku, and split-half cosine was 0.566 versus 0.659.
- The paired gradient-direction shift was unresolved in both scales. The raw
  debiased shift estimate was below zero after noise subtraction; standardized
  paired SNR was 0.44 with an inferred critical batch around 2,593 matches.

Thus actor hidden states carry useful predictive information, but at this
sample size that information mostly improves endpoint fit rather than the
full-gradient signal/noise tradeoff. The remaining variance is more consistent
with opponent/chance outcomes and the seat-local semi-Markov trace than with a
missing actor representation.

## Recommendation

Keep current-kyoku as the production estimator. Do not restore the historical
whole-match Q-boosting critic or label it a variance reduction based only on
advantage variance. Do not deploy the tested public current-kyoku Q head: at
`lambda = 1` it needs about twice the standardized critical batch. The
independent privileged critic tested here also misses the gate, requiring about
46% more standardized matches than current-kyoku. The detached actor-hidden
critic also misses it, requiring about 47% more standardized matches. A future
paper-faithful, separately parameterized history-mirroring critic remains gated
on these criteria:

1. report full-actor gradient noise trace, debiased signal, critical batch size,
   split-half cosine, and paired gradient error;
2. evaluate raw and production-standardized advantages;
3. use `lambda = 1` for the unbiased claim, with `0.95` reported only as an
   explicitly biased ablation;
4. demonstrate critic accuracy out of sample and across policy updates;
5. beat current-kyoku on critical batch size without a detectable paired
   gradient-direction shift, then confirm strength with held-out seat-rotated
   games.

## Sources

- [Fan & Farina, arXiv:2605.19235](https://arxiv.org/abs/2605.19235)
- [Full HTML paper, including Theorem 3.1](https://arxiv.org/html/2605.19235)
- `runs/evaluations/vrpo-v-vgae-update0-4096/report.json`
- `runs/evaluations/vrpo-v-vgae-update16-4096/report.json`
- `runs/evaluations/current-kyoku-vrpo-lambda-one-update0-4096/report.json`
