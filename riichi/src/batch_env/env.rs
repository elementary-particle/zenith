use std::{
    collections::{BTreeMap, HashSet},
    time::{Duration, Instant},
};

use rayon::{prelude::*, ThreadPool, ThreadPoolBuilder};
use riichi_core::{
    game::phase::{EnvironmentLifecycle, HandPhase},
    snapshot, Action, EventKind, GameState,
};

use super::metrics::EnvMetrics;

#[derive(Debug, thiserror::Error)]
pub enum EnvError {
    #[error("env is closed")]
    Closed,
    #[error("invalid argument: {0}")]
    InvalidArgument(String),
    #[error("environment id {0} is out of range")]
    EnvironmentOutOfRange(u32),
    #[error("duplicate environment id {0}")]
    DuplicateEnvironment(u32),
    #[error("snapshot environment {snapshot_id} does not match target {target_id}")]
    SnapshotEnvironmentMismatch { target_id: u32, snapshot_id: u32 },
    #[error(
        "environment {environment_id} generation {episode_generation} stopped in lifecycle \
         {lifecycle:?}, phase {phase:?}, without a queryable decision"
    )]
    UnqueryableState {
        environment_id: u32,
        episode_generation: u64,
        lifecycle: EnvironmentLifecycle,
        phase: Option<HandPhase>,
    },
    #[error(transparent)]
    Core(#[from] riichi_core::error::CoreError),
}

#[derive(Clone, Debug)]
pub struct BatchTransition {
    pub transition_id: u64,
    pub states: Vec<GameState>,
    pub events: Vec<riichi_core::game::event::EventRecord>,
    pub validation_ns: u64,
    pub env_step_ns: u64,
    pub materialization_ns: u64,
}

pub struct BatchEnv {
    master_seed: u64,
    num_threads: usize,
    pool: ThreadPool,
    states: Vec<GameState>,
    transition_id: u64,
    metrics: EnvMetrics,
    closed: bool,
}

impl BatchEnv {
    pub fn new(num_envs: usize, master_seed: u64, num_threads: usize) -> Result<Self, EnvError> {
        if num_envs == 0 {
            return Err(EnvError::InvalidArgument(
                "num_envs must be at least one".into(),
            ));
        }
        if num_threads == 0 {
            return Err(EnvError::InvalidArgument(
                "num_threads must be at least one".into(),
            ));
        }
        let pool = ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .thread_name(|index| format!("riichi-env-{index}"))
            .build()
            .map_err(|error| EnvError::InvalidArgument(error.to_string()))?;
        Ok(Self {
            master_seed,
            num_threads,
            pool,
            states: (0..num_envs as u32).map(GameState::new).collect(),
            transition_id: 0,
            metrics: EnvMetrics::default(),
            closed: false,
        })
    }

    pub fn num_envs(&self) -> usize {
        self.states.len()
    }

    pub fn num_threads(&self) -> usize {
        self.num_threads
    }

    pub fn master_seed(&self) -> u64 {
        self.master_seed
    }

    pub fn reset(&mut self, environment_ids: &[u32]) -> Result<BatchTransition, EnvError> {
        self.ensure_open()?;
        let validation_started = Instant::now();
        validate_ids(environment_ids, self.states.len())?;
        let validation = validation_started.elapsed();
        let selected: HashSet<u32> = environment_ids.iter().copied().collect();
        let master_seed = self.master_seed;
        let env_started = Instant::now();
        let automatic = self.pool.install(|| {
            self.states
                .par_iter_mut()
                .filter(|state| selected.contains(&state.environment_id))
                .map(|state| state.reset_from_seed_and_count(master_seed))
                .sum::<u64>()
        });
        let env_step = env_started.elapsed();
        // reset_from_seed stabilizes automatic control flow before materialization.
        self.finish(environment_ids, validation, env_step, 0, automatic)
    }

    pub fn step(&mut self, actions: &[Action]) -> Result<BatchTransition, EnvError> {
        self.ensure_open()?;
        let validation_started = Instant::now();
        if actions.is_empty() {
            return Err(EnvError::InvalidArgument(
                "actions must contain at least one complete decision frame".into(),
            ));
        }
        let mut grouped: BTreeMap<u32, Vec<Action>> = BTreeMap::new();
        for action in actions {
            if action.environment_id as usize >= self.states.len() {
                return Err(EnvError::EnvironmentOutOfRange(action.environment_id));
            }
            grouped
                .entry(action.environment_id)
                .or_default()
                .push(action.clone());
        }
        for (&environment_id, values) in &grouped {
            self.states[environment_id as usize].validate_actions(values)?;
        }
        let validation = validation_started.elapsed();
        let selected: HashSet<u32> = grouped.keys().copied().collect();
        let env_started = Instant::now();
        let automatic = self.pool.install(|| {
            self.states
                .par_iter_mut()
                .filter(|state| selected.contains(&state.environment_id))
                .map(|state| {
                    state
                        .step_and_count(
                            grouped.get(&state.environment_id).expect("validated group"),
                        )
                        .expect("prevalidated actions remain valid")
                })
                .sum::<u64>()
        });
        let env_step = env_started.elapsed();
        let ids = grouped.keys().copied().collect::<Vec<_>>();
        self.finish(&ids, validation, env_step, actions.len() as u64, automatic)
    }

    pub fn inspect(&mut self, environment_ids: &[u32]) -> Result<BatchTransition, EnvError> {
        self.ensure_open()?;
        let started = Instant::now();
        validate_ids(environment_ids, self.states.len())?;
        let validation = started.elapsed();
        self.materialize(environment_ids, validation, Duration::ZERO, false, 0, 0)
    }

    pub fn snapshots(&self, environment_ids: &[u32]) -> Result<BTreeMap<u32, Vec<u8>>, EnvError> {
        self.ensure_open()?;
        validate_ids(environment_ids, self.states.len())?;
        environment_ids
            .iter()
            .map(|&id| snapshot::encode(&self.states[id as usize]).map(|bytes| (id, bytes)))
            .collect::<Result<_, _>>()
            .map_err(EnvError::from)
    }

    pub fn restore(
        &mut self,
        snapshots: &BTreeMap<u32, Vec<u8>>,
    ) -> Result<BatchTransition, EnvError> {
        self.ensure_open()?;
        let validation_started = Instant::now();
        let ids = snapshots.keys().copied().collect::<Vec<_>>();
        validate_ids(&ids, self.states.len())?;
        let mut decoded = Vec::with_capacity(ids.len());
        for (&target_id, bytes) in snapshots {
            let state = snapshot::decode(bytes)?;
            if state.environment_id != target_id {
                return Err(EnvError::SnapshotEnvironmentMismatch {
                    target_id,
                    snapshot_id: state.environment_id,
                });
            }
            decoded.push((target_id, state));
        }
        let validation = validation_started.elapsed();
        let env_started = Instant::now();
        let mut automatic = 0;
        for (id, mut state) in decoded {
            let before = state.automatic_decisions;
            state.stabilize_automatic_decisions();
            automatic += state.automatic_decisions.saturating_sub(before);
            self.states[id as usize] = state;
        }
        let env_step = env_started.elapsed();
        self.finish(&ids, validation, env_step, 0, automatic)
    }

    pub fn metrics(&mut self, reset: bool) -> EnvMetrics {
        let value = self.metrics;
        if reset {
            self.metrics = EnvMetrics::default();
        }
        value
    }

    pub fn record_exchange(&mut self, elapsed: Duration) {
        self.metrics.exchange += elapsed;
    }

    pub fn close(&mut self) {
        self.closed = true;
    }

    fn finish(
        &mut self,
        ids: &[u32],
        validation: Duration,
        env_step: Duration,
        action_count: u64,
        automatic_count: u64,
    ) -> Result<BatchTransition, EnvError> {
        self.ensure_queryable_or_terminal(ids)?;
        self.materialize(
            ids,
            validation,
            env_step,
            true,
            action_count,
            automatic_count,
        )
    }

    fn ensure_queryable_or_terminal(&self, ids: &[u32]) -> Result<(), EnvError> {
        for &id in ids {
            let state = &self.states[id as usize];
            if matches!(
                state.lifecycle,
                EnvironmentLifecycle::Uninitialized
                    | EnvironmentLifecycle::Complete
                    | EnvironmentLifecycle::Failed
            ) {
                continue;
            }
            let queryable = state
                .hanchan
                .as_ref()
                .and_then(|game| game.hand.decision_frame.as_ref())
                .is_some_and(|frame| !frame.decisions.is_empty());
            if !queryable {
                return Err(EnvError::UnqueryableState {
                    environment_id: state.environment_id,
                    episode_generation: state.episode_generation,
                    lifecycle: state.lifecycle,
                    phase: state.hanchan.as_ref().map(|game| game.hand.phase),
                });
            }
        }
        Ok(())
    }

    fn materialize(
        &mut self,
        ids: &[u32],
        validation: Duration,
        env_step: Duration,
        drain_events: bool,
        action_count: u64,
        automatic_count: u64,
    ) -> Result<BatchTransition, EnvError> {
        let materialization_started = Instant::now();
        let mut ordered = ids.to_vec();
        ordered.sort_unstable();
        let mut events = Vec::new();
        let mut states = Vec::with_capacity(ordered.len());
        for id in ordered {
            let state = &mut self.states[id as usize];
            if drain_events {
                events.extend(state.take_events());
            }
            states.push(state.clone());
        }
        events.sort_by_key(|event| {
            (
                event.environment_id,
                event.episode_generation,
                event.sequence,
                EventKind::mjai_name(event.kind),
            )
        });
        let materialization = materialization_started.elapsed();
        self.transition_id = self.transition_id.wrapping_add(1);
        self.metrics.calls += 1;
        self.metrics.states += states.len() as u64;
        self.metrics.model_queries += action_count;
        self.metrics.rust_resolved_decisions += automatic_count;
        self.metrics.validation += validation;
        self.metrics.env_step += env_step;
        self.metrics.materialization += materialization;
        Ok(BatchTransition {
            transition_id: self.transition_id,
            states,
            events,
            validation_ns: validation.as_nanos() as u64,
            env_step_ns: env_step.as_nanos() as u64,
            materialization_ns: materialization.as_nanos() as u64,
        })
    }

    fn ensure_open(&self) -> Result<(), EnvError> {
        if self.closed {
            Err(EnvError::Closed)
        } else {
            Ok(())
        }
    }
}

fn validate_ids(ids: &[u32], len: usize) -> Result<(), EnvError> {
    let mut seen = HashSet::new();
    for &id in ids {
        if id as usize >= len {
            return Err(EnvError::EnvironmentOutOfRange(id));
        }
        if !seen.insert(id) {
            return Err(EnvError::DuplicateEnvironment(id));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn restore_stabilizes_a_frame_free_settlement_before_materializing() {
        let mut state = GameState::new(0);
        state.reset_from_seed(7);
        state.take_events();
        let hand = &mut state.hanchan.as_mut().unwrap().hand;
        hand.phase = HandPhase::Settlement;
        hand.decision_frame = None;
        let encoded = snapshot::encode(&state).unwrap();

        let mut env = BatchEnv::new(1, 7, 1).unwrap();
        let transition = env.restore(&BTreeMap::from([(0, encoded)])).unwrap();
        let restored = &transition.states[0];

        assert!(
            restored.lifecycle == EnvironmentLifecycle::Complete
                || restored
                    .hanchan
                    .as_ref()
                    .unwrap()
                    .hand
                    .decision_frame
                    .is_some(),
            "restore returned an active state without a queryable decision"
        );
        assert!(transition
            .events
            .iter()
            .any(|event| event.kind == EventKind::EndKyoku));
    }
}
