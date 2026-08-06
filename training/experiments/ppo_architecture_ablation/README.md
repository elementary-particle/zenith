# PPO-driven architecture ablations

This experiment uses the final 262,144-match PPO checkpoint as the diagnostic
target. It asks whether the online-arena weakness and aggressive greedy policy
come from missing strategic computations, action-head capacity, or an
inadequate representation of multi-tile action consequences.

## PPO diagnosis

Matched held-out counterfactuals do not support a missing-information account.
Compared with the production BC checkpoint, PPO is more sensitive to every
tested strategic relation:

| Counterfactual effect | BC logits | PPO logits | PPO/BC |
| --- | ---: | ---: | ---: |
| Dora indicator successor | 2.357 | 4.020 | 1.71x |
| Suji seat attribution | 1.406 | 2.160 | 1.54x |
| Four-visible kabe | 2.875 | 4.859 | 1.69x |
| Post-riichi temporal safety | 5.489 | 8.573 | 1.56x |
| Score-dependent riichi/dama, ordinary rounds | 2.870 | 3.924 | 1.37x |

The score probe exposes over-aggression rather than blindness. For an early
leader, the mean riichi-minus-dama margin moves from `-1.543` in BC to only
`-0.520` in PPO. For a trailer it moves from `+1.327` to `+3.405`. PPO also
relies more heavily on RoPE for post-riichi safety: disabling RoPE reduces the
rate at which both swapped counterfactuals are correct from `72.22%` to
`58.33%`; the same BC ablation changes `72.22%` to `71.30%`.

This matches the game behavior. Relative to BC, greedy PPO calls more
(`26.6% -> 30.4%`), declares riichi more (`19.5% -> 23.4%`), uses much less
dama (`13.7% -> 5.8%`), and deals in more (`12.6% -> 13.6%`). During training,
policy entropy falls from `0.2670` to `0.1835`, while the sampled riichi rate at
riichi opportunities rises from `40.7%` to `62.4%`.

Artifacts:

- `runs/audits/ppo-u128-strategic-information-probe/report.json`
- `runs/audits/ppo-u128-positional-riichi-probe/report.json`
- BC controls: `runs/strategic-information-probe/report.json` and
  `runs/positional-riichi-probe/report.json`

## Matched architecture screen

Each candidate starts behavior-identically from the PPO checkpoint and adapts
on the same 250,000 human decisions. Paired validation uses 50,000 decisions
from 80 held-out games.

| Variant | Added parameters | Delta NLL vs baseline | 95% CI | Decision |
| --- | ---: | ---: | ---: | --- |
| Role-aware shared action tiles | 768 | +0.000033 | [-0.000023, +0.000089] | tied; no evidence of a tile-role bottleneck |
| Two identity-initialized action blocks | 1,033,344 | +0.000693 | [+0.000400, +0.000987] | reject; capacity is not the bottleneck |
| Explicit post-action hand shape | 36,864 | +0.002182 | [+0.001721, +0.002642] | reject overall; 16% slower offline inference |

The post-action encoder slightly improves call NLL (`0.64800 -> 0.64745`) and
riichi NLL (`0.68813 -> 0.68325`), but harms discard and pass prediction. A
larger game gate was run because this trade could conceivably be strategic.

Over 4,096 greedy games (1,024 paired seeds), one post-action candidate versus
three identically adapted baselines scores:

| Metric | Candidate minus baseline | 95% interval |
| --- | ---: | ---: |
| Pairwise win rate | 50.553% | 49.406-51.685% |
| Score | +0.232k | -0.346k to +0.797k |
| Placement (lower is better) | -0.0221 | -0.0674 to +0.0238 |

The candidate is tied, not improved. It calls slightly more (`26.71%` versus
`26.47%`), deals in slightly less (`12.15%` versus `12.32%`), and wins slightly
more (`21.20%` versus `20.99%`), but every strength interval crosses zero.
Together with worse imitation and extra compute, this rejects promotion.

Artifacts:

- `runs/audits/ppo-architecture-ablation-ppo-init-250k/report.json`
- `runs/audits/ppo-post-action-shape-ppo-init-250k/report.json`
- `runs/audits/ppo-post-action-shape-vs-baseline-greedy-4096/evaluation.json`

## Conclusion

The current evidence does not identify model capacity or missing tactical
relations as the main PPO limitation. The actor can represent the tested
strategy, and PPO amplifies it too strongly while losing entropy. Keep the
tsumogiri action distinction, but do not promote any architecture candidate
from this screen.

The next high-information ablation should be on PPO calibration: preserve a
BC-logit or BC-policy residual anchor, constrain family-level movement for
riichi/call decisions, or add an entropy/greedy-stability target. This directly
tests the observed failure mode and is preferable to making the actor larger.

## Reproduction

```bash
PYTHONPATH=training/src .venv/bin/python \
  training/experiments/ppo_architecture_ablation/audit.py \
  --initial-checkpoint runs/ppo-bc2024-brave4/checkpoints \
  --output runs/audits/ppo-architecture-ablation-ppo-init-250k \
  --train-decisions 250000 --validation-decisions 50000

PYTHONPATH=training/src .venv/bin/python \
  training/experiments/ppo_architecture_ablation/evaluate.py \
  --candidate-variant post_action_shape \
  --candidate-model \
    runs/audits/ppo-post-action-shape-ppo-init-250k/post_action_shape.model.pt \
  --baseline-model \
    runs/audits/ppo-post-action-shape-ppo-init-250k/baseline.model.pt \
  --matches 1024 --batch-size 1024 --device cuda \
  --output runs/audits/ppo-post-action-shape-vs-baseline-greedy-1024
```
