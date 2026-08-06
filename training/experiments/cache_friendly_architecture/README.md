# Cache-friendly policy architecture audit

This audit asks whether Zenith can make the append-only public event history a
stable causal prefix without losing the production BC policy's strength.  It
also tests whether the action head can replace raw-history attention with a
small fixed set of summary tokens.

## Findings

### Raw history cannot be collapsed into the existing summaries

At a matched one-million-decision BC budget, replacing the full action memory
with the actor, strategy, 34 tile, match-summary, and kyoku-summary slots raised
held-out NLL from `0.88760` to `0.94640`.  The paired game-clustered difference
was `+0.05877`, 95% CI `[+0.05575, +0.06179]`.  Most of the loss was discard
prediction (`1.04668 -> 1.12590`).

This is evidence that the 192-dimensional actor query and two summaries do not
retain all action-specific information from the public history.  A fast action
head must preserve richer history access or learn a substantially better
compression objective.

The fixed summary memory would remove about 70% of action-to-memory attention
interactions at the observed mean sequence length, but its quality loss rejects
it as a production candidate.

### Event-first causal layout is viable

The viable variant moves only `TokenKind.EVENT` rows before mutable match and
tactical state.  It retains the full generalized transformer and full-history
action attention, has exactly the same 3,682,736 parameters and state-dict
keys, and makes the within-kyoku event sequence an append-only cacheable prefix.

From a fresh initialization and one million identical BC decisions, event-first
improved held-out NLL from `0.88749` to `0.88364`.  The paired game-clustered
difference was `-0.00387`, 95% CI `[-0.00504, -0.00271]`.

A production BC checkpoint cannot be reordered without adaptation: its NLL
immediately changes from `0.52595` to `1.20391`.  After one million matched BC
adaptation decisions, event-first reaches `0.52330`, better than the unchanged
production checkpoint but slightly behind a current-layout continuation at
`0.52236`.  Their paired difference is `+0.00094`, 95% CI
`[+0.00056, +0.00132]`.

### Greedy game strength is tied with production BC

One event-first policy against three unchanged production BC policies was run
for 4,096 native games over 1,024 paired seeds:

| Metric | Event-first minus production BC | 95% interval |
| --- | ---: | ---: |
| Pairwise win rate | 49.935% | 48.771–51.082% |
| Score | -0.006k | -0.596k–+0.603k |
| Placement (lower is better) | +0.0026 | -0.0433–+0.0492 |

The result is non-inferior under a two-percentage-point pairwise margin, but
does not establish a one-point margin.  The event-first policy calls more
(`28.83%` versus `27.25%`), declares riichi less (`16.97%` versus `19.34%`),
uses dama more (`17.77%` versus `13.98%`), and deals in slightly more (`13.40%`
versus `12.85%`).  Win rate is nearly unchanged (`21.58%` versus `21.73%`).

### Stable boundary prefix is the lower-risk recommendation

Keeping the original causal order but replacing leading current-score/counter
numerics with their start-of-kyoku values makes `boundary + events` permanently
cacheable.  The existing checkpoint is nearly zero-shot compatible: held-out
NLL changes only from `0.525950` to `0.526093`.  After identical one-million-row
continuations, stable-boundary slightly beats the current layout (`0.522157`
versus `0.522347`); paired difference `-0.000186`, 95% CI
`[-0.000263, -0.000108]`.

Its 4,096-game greedy comparison is also tied with unchanged production BC:
49.935% pairwise (95% CI `48.804–51.042%`), score `-0.202k` (95% CI
`-0.809k–+0.385k`), and placement difference `+0.0026` (95% CI
`-0.0417–+0.0479`).  This is the recommended layout for implementing the
first real K/V cache because it preserves match context before the event stream
and needs no checkpoint conversion phase.

## Cache boundary

This experiment establishes cache-ready layouts, not a completed inference
cache.  The current `Decoder.forward` API recomputes the complete sequence and
the four action-memory blocks recompute their memory K/V projections.  The
recommended stable-boundary layout permits a follow-up implementation to retain
decoder and action-memory K/V for the boundary plus append-only event prefix,
while recomputing only the current-state suffix and legal actions.  A new hand
resets the prefix.

The production layout can reuse its prefix only while the leading match-state
tokens remain unchanged; a riichi score/deposit change invalidates all later
event K/V.  Both tested layouts remove that invalidation dependency;
stable-boundary does so while retaining match context before events.

## Artifacts

- Correct fresh 1M comparison:
  `runs/audits/cache-friendly-event-prefix-1m/report.json`
- Production-checkpoint adaptation:
  `runs/audits/cache-friendly-event-prefix-production-bc-1m/report.json`
- 4,096-game greedy evaluation:
  `runs/audits/cache-friendly-event-prefix-vs-production-bc-greedy-4096/evaluation.json`
- Candidate weights:
  `runs/audits/cache-friendly-event-prefix-production-bc-1m/event_prefix.model.pt`
- Stable-boundary BC audit:
  `runs/audits/cache-friendly-stable-boundary-production-bc-1m/report.json`
- Stable-boundary 4,096-game evaluation:
  `runs/audits/cache-friendly-stable-boundary-vs-production-bc-greedy-4096/evaluation.json`
- Recommended stable-boundary weights:
  `runs/audits/cache-friendly-stable-boundary-production-bc-1m/stable_boundary.model.pt`

The earlier `cache-friendly-architecture-screen-1m-parallel` event-prefix rows
are superseded: that screen used the `Segment.EVENT` alias and inadvertently
moved all kyoku-state rows.  Its `summary_memory` control remains valid, but all
event-prefix conclusions above come from the corrected token-kind transform.

## Reproduction

```bash
PYTHONPATH=training/src .venv/bin/python \
  training/experiments/cache_friendly_architecture/audit.py \
  --config training/configs/default.toml \
  --output runs/audits/cache-friendly-stable-boundary-production-bc-1m \
  --initial-checkpoint runs/behavior-cloning-rank-v/checkpoints \
  --train-decisions 1000000 --validation-decisions 200000 \
  --variants baseline stable_boundary \
  --data-workers 8 --replay-batch 1 --replay-threads 1

PYTHONPATH=training/src .venv/bin/python \
  training/experiments/cache_friendly_architecture/evaluate.py \
  --config training/configs/default.toml \
  --candidate-layout stable_boundary \
  --candidate-model \
    runs/audits/cache-friendly-stable-boundary-production-bc-1m/stable_boundary.model.pt \
  --reference-checkpoint runs/behavior-cloning-rank-v/checkpoints \
  --seed-start 10001 --seed-count 1024 --batch-size 1024 \
  --device cuda \
  --output \
    runs/audits/cache-friendly-stable-boundary-vs-production-bc-greedy-4096
```
