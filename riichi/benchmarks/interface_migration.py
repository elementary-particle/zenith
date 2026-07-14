"""Current-host native bulk versus Python-thread executor benchmark."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
import os
import platform
import resource
import random
import statistics
import time

import riichi


def _first_actions(transition):
    return tuple(decision.actions[0] for state in transition.states for decision in state.decisions)


def _native(num_envs, threads, seed, steps):
    env = riichi.Env(num_envs, master_seed=seed, num_threads=threads)
    transition = env.reset(range(num_envs))
    started = time.perf_counter_ns()
    for _ in range(steps):
        transition = env.step(_first_actions(transition))
    elapsed = time.perf_counter_ns() - started
    arrays = transition.as_numpy()
    digest = sha256(
        arrays["state_scores"].tobytes() + arrays["event_kind"].tobytes()
    ).hexdigest()
    metrics = env.metrics()
    env.close()
    return {"elapsed_ns": elapsed, "digest": digest, "metrics": metrics}


def _single(seed, steps):
    env = riichi.Env(1, master_seed=seed, num_threads=1)
    transition = env.reset([0])
    for _ in range(steps):
        transition = env.step(_first_actions(transition))
    arrays = transition.as_numpy()
    digest = sha256(
        arrays["state_scores"].tobytes() + arrays["event_kind"].tobytes()
    ).hexdigest()
    env.close()
    return digest


def _python_threads(num_envs, workers, seed, steps):
    started = time.perf_counter_ns()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        digests = tuple(pool.map(
            lambda environment_id: _single(seed ^ environment_id, steps),
            range(num_envs),
        ))
    return {
        "elapsed_ns": time.perf_counter_ns() - started,
        "digest": sha256("".join(digests).encode()).hexdigest(),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    native = [_native(args.num_envs, args.threads, args.seed, args.steps) for _ in range(args.trials)]
    prototype = [
        _python_threads(args.num_envs, args.threads, args.seed, args.steps)
        for _ in range(args.trials)
    ]
    native_median = statistics.median(row["elapsed_ns"] for row in native)
    python_median = statistics.median(row["elapsed_ns"] for row in prototype)
    ratios = [
        prototype[index]["elapsed_ns"] / native[index]["elapsed_ns"]
        for index in range(args.trials)
    ]
    bootstrap = random.Random(args.seed)
    estimates = sorted(
        statistics.median(bootstrap.choices(ratios, k=len(ratios)))
        for _ in range(2_000)
    )
    lower = estimates[int(0.025 * len(estimates))]
    upper = estimates[int(0.975 * len(estimates))]
    transitions = args.num_envs * args.steps
    report = {
        "host": platform.platform(),
        "python": platform.python_version(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "num_envs": args.num_envs,
        "threads": args.threads,
        "steps": args.steps,
        "trials": args.trials,
        "seed": args.seed,
        "native": native,
        "python_thread_prototype": prototype,
        "median_speedup": python_median / native_median,
        "speedup_95_percent_ci": [lower, upper],
        "native_transitions_per_second": transitions * 1e9 / native_median,
        "python_thread_transitions_per_second": transitions * 1e9 / python_median,
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "decision": "keep_rust_batch_env" if native_median <= python_median else "architecture_review",
    }
    rendered = json.dumps(report, indent=2)
    if args.output:
        from pathlib import Path
        Path(args.output).write_text(rendered, encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
