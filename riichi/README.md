# Riichi env

`riichi` provides deterministic, multithreaded four-player riichi games for training and evaluation.
The local `riichi-core` crate owns one-game rules, state transitions, bound actions, MJAI-compatible
events, snapshots, scoring, RNG, shanten, and hand efficiency. It has no Python, NumPy, batching, threading,
PyTorch, PPO, reward, or training dependency. The root crate adds the thin Rust `BatchEnv`, PyO3
values, and an optional bulk NumPy projection.

The baseline is `riichilab-mjsoul-yonma-v1`: RiichiLab's Mahjong Soul four-player red-five preset and
East–South ranked progression. All eligible reactions are submitted simultaneously in one frame.
State, event, decision, hand-efficiency, and snapshot versions are independent and currently
`5/2/1/2/3`; rules and RNG profile IDs are `2/1`. There is no umbrella env API schema.

## Build

```bash
python -m pip install -r requirements.txt
maturin develop --manifest-path riichi/Cargo.toml --release
cargo run --release --manifest-path riichi/Cargo.toml --bin build_shanten_cache
```

The shanten table is repository-owned and cached under
`$XDG_CACHE_HOME/zenith-riichi/shanten-v1.bin` (or `$HOME/.cache/...`). Set
`ZENITH_SHANTEN_CACHE` to use another local disk. `riichi.ensure_shanten_cache()` performs the same
prewarm from a wheel installation.

## Native API

```python
import numpy as np
import riichi

env = riichi.Env(
    4096,
    master_seed=1,
    num_threads=8,
    rules_profile=riichi.RULES_PROFILE,
    privileged=True,
)

transition = env.reset(range(4096))
selections = tuple(
    space.candidates[0].select()
    for state in transition.states
    for space in state.action_spaces
)
transition = env.step(selections)

# One optional bulk conversion per native call for training hot paths.
arrays = transition.as_numpy()
assert arrays["state_scores"].shape == (4096, 4)
assert not arrays["candidate_kind"].flags.writeable

# Exact continuation; values are ordinary bytes owned by Python.
snapshots = env.snapshot(range(4096))
restored = env.restore(snapshots)

# Env-independent hand-efficiency evaluation returns new read-only arrays.
counts = arrays["action_space_concealed_counts"]
open_melds = np.zeros(len(counts), dtype=np.uint8)
efficiency = riichi.evaluate_hand_efficiency(counts, open_melds)
assert efficiency.shanten.shape == (len(counts), 4)
```

`Env.reset`, `step`, `advance`, `restore`, and `inspect` return immutable `Transition` values containing
native `State`, `Event`, `ActionSpace`, `ActionCandidate`, and `ActionSelection` objects. Returned values and NumPy projections
remain valid after later env calls or after the env is dropped. `step` validates the complete
simultaneous selection set before any game changes; `advance` resolves one decision-free rules
transition for the selected environments. Rust canonicalizes physical-copy-equivalent actions and
omits seats with no semantic choice. Python supplies no output arrays, capacities, buffer rings, or
generation counters.

The root extension contains no duplicate rules engine: `riichi-core` is the only rules/state owner.
Batching and worker partitioning live in the thin root `BatchEnv`; Python owns rollout/model batching.

Events are the minimal chronological authority for realized gameplay and use the RiichiLab/MJAI game
types (`start_game`, `start_kyoku`, `tsumo`, `dahai`, calls, `dora`, `reach`, `hora`, `ryukyoku`, and
end events). State is the current maintenance/oracle view and contains decisions and valid actions but
no events or derived shanten/ukeire fields. Public state exposes `live_wall_remaining`; privileged
`HiddenState` additionally exposes the canonical 34-type `live_wall_counts` vector. Python owns
ordinary/oracle/critic masking; canonical native state and events are never rewritten for training.

Run the deterministic native rollout and snapshot replay example with:

```bash
python ppo.py --num-envs 4096 --num-threads 8 --seed 1 --smoke-steps 1000
```

For the measured executor comparison, run
`python riichi/benchmarks/interface_migration.py --num-envs 256 --threads 4 --steps 32 --trials 5`.
The current Ryzen 7 9800X3D host retained `BatchEnv` after a 1.688x median advantage over the
Python-thread prototype.
