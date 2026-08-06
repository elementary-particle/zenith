# EMA magnet sweep

This experiment isolates the two EMAgnet hyperparameters after the production
`0.3 / 65,536` configuration produced a small exploitability improvement at
16,384 matches and then plateaued at 32,768 matches.

The coarse grid crosses magnet KL coefficients `0.03`, `0.1`, and `0.3` with
match-count half-lives `8,192`, `32,768`, and `131,072`. Every arm uses the
same BC initialization, training seed, 2,048-match logical/physical batch,
two actor epochs, BF16 production profile, and 16,384-match training budget.
The short run budget retains the 262,144-match production learning-rate
schedule through `curriculum.schedule_matches`, so warmup and decay are
identical to the first eight production updates.
Each final policy is evaluated with the fixed 256-seed, four-rotation,
1,024-game exploitability protocol at evaluation batch size 1,024.

The original long run at coefficient `0.3` and half-life `65,536` is an
additional control. Rank arms primarily by the current-greedy pairwise witness,
then use BC and earlier-checkpoint witnesses to reject regressions. Because the
grid shares one training seed, replicate the leading settings before promoting
one to production.

The production default was subsequently set to coefficient `0.03` and
half-life `8,192`. It was statistically tied with the best observed greedy
witness while keeping the magnet and policy losses balanced, making it the
preferred candidate for avoiding the prior long-run anchoring plateau.

Run the complete sequential CUDA sweep from the repository root with
`zsh training/experiments/ema_sweep/run.zsh`.
