use std::collections::BTreeSet;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use riichi::{BatchEnv, GameState};

fn first_actions(states: &[GameState]) -> Vec<riichi::Action> {
    let mut selected = Vec::new();
    for state in states {
        let mut seats = BTreeSet::new();
        for action in state.legal_actions() {
            if seats.insert(action.seat) {
                selected.push(action);
            }
        }
    }
    selected
}

fn native_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("interface_migration/native_batch_env");
    for threads in [1_usize, 2, 4, 8] {
        let environments = 4_096_usize;
        group.throughput(Throughput::Elements(environments as u64));
        group.bench_function(BenchmarkId::new("threads", threads), |b| {
            b.iter_batched(
                || {
                    let mut env = BatchEnv::new(environments, 1, threads).unwrap();
                    let transition = env
                        .reset(&(0..environments as u32).collect::<Vec<_>>())
                        .unwrap();
                    (env, transition.states)
                },
                |(mut env, states)| {
                    let selected = first_actions(&states);
                    env.step(&selected).unwrap()
                },
                criterion::BatchSize::LargeInput,
            )
        });
    }
    group.finish();
}

fn sequential_state_only(c: &mut Criterion) {
    let environments = 4_096_usize;
    c.bench_function("interface_migration/sequential_state_only", |b| {
        b.iter_batched(
            || {
                (0..environments as u32)
                    .map(|environment_id| {
                        let mut state = GameState::new(environment_id);
                        state.reset_from_seed(1);
                        state
                    })
                    .collect::<Vec<_>>()
            },
            |mut states| {
                for state in &mut states {
                    let selected = first_actions(std::slice::from_ref(state));
                    state.step(&selected).unwrap();
                }
                states
            },
            criterion::BatchSize::LargeInput,
        )
    });
}

criterion_group!(benches, native_batch, sequential_state_only);
criterion_main!(benches);
