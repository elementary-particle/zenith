use std::{
    collections::{BTreeMap, HashSet},
    time::{Duration, Instant},
};

use rayon::{prelude::*, ThreadPool, ThreadPoolBuilder};
use riichi_core::{
    game::{
        phase::EnvironmentLifecycle,
        search::{fork_with_privileged_wall, fork_with_public_information, rebind_environment},
    },
    snapshot, ActionSelection, EventKind, GameState, ReplayEvent, ReplayHanchan,
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
    #[error(transparent)]
    Core(#[from] riichi_core::error::CoreError),
}

#[derive(Clone, Debug)]
pub struct BatchTransition {
    pub transition_id: u64,
    pub states: Vec<GameState>,
    pub events: Vec<riichi_core::game::event::EventRecord>,
    pub applied_selections: Vec<ActionSelection>,
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
        Self::new_with_rules_profile(
            num_envs,
            master_seed,
            num_threads,
            riichi_core::RULES_PROFILE_ID,
        )
    }

    pub fn new_with_rules_profile(
        num_envs: usize,
        master_seed: u64,
        num_threads: usize,
        rules_profile_id: u32,
    ) -> Result<Self, EnvError> {
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
        if riichi_core::game::rules::profile::by_id(rules_profile_id).is_none() {
            return Err(EnvError::InvalidArgument(format!(
                "unsupported rules profile id: {rules_profile_id}"
            )));
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
            states: (0..num_envs as u32)
                .map(|environment_id| {
                    GameState::new_with_rules_profile(environment_id, rules_profile_id)
                })
                .collect(),
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
        self.finish(
            environment_ids,
            validation,
            env_step,
            0,
            automatic,
            Vec::new(),
        )
    }

    pub fn step(&mut self, selections: &[ActionSelection]) -> Result<BatchTransition, EnvError> {
        self.ensure_open()?;
        let validation_started = Instant::now();
        if selections.is_empty() {
            return Err(EnvError::InvalidArgument(
                "selections must resolve at least one complete decision".into(),
            ));
        }
        let mut grouped: BTreeMap<u32, Vec<ActionSelection>> = BTreeMap::new();
        for selection in selections {
            if selection.environment_id as usize >= self.states.len() {
                return Err(EnvError::EnvironmentOutOfRange(selection.environment_id));
            }
            grouped
                .entry(selection.environment_id)
                .or_default()
                .push(selection.clone());
        }
        for (&environment_id, values) in &grouped {
            self.states[environment_id as usize].validate_selections(values)?;
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
        self.finish(
            &ids,
            validation,
            env_step,
            selections.len() as u64,
            automatic,
            selections.to_vec(),
        )
    }

    /// Resolve one frame-free transition for each selected environment and
    /// materialize immediately so Python can evaluate the resulting frame.
    pub fn advance(&mut self, environment_ids: &[u32]) -> Result<BatchTransition, EnvError> {
        self.ensure_open()?;
        let validation_started = Instant::now();
        validate_ids(environment_ids, self.states.len())?;
        for &environment_id in environment_ids {
            let state = &self.states[environment_id as usize];
            let has_decision = state
                .hanchan
                .as_ref()
                .and_then(|game| game.hand.decision.as_ref())
                .is_some();
            if has_decision || state.lifecycle == EnvironmentLifecycle::Complete {
                return Err(EnvError::InvalidArgument(format!(
                    "environment {environment_id} has no automatic transition to advance"
                )));
            }
        }
        let validation = validation_started.elapsed();
        let selected: HashSet<u32> = environment_ids.iter().copied().collect();
        let env_started = Instant::now();
        let automatic = self.pool.install(|| {
            self.states
                .par_iter_mut()
                .filter(|state| selected.contains(&state.environment_id))
                .map(|state| u64::from(state.advance_automatic_once()))
                .sum::<u64>()
        });
        let env_step = env_started.elapsed();
        self.materialize(
            environment_ids,
            validation,
            env_step,
            true,
            0,
            automatic,
            Vec::new(),
        )
    }

    pub fn inspect(&mut self, environment_ids: &[u32]) -> Result<BatchTransition, EnvError> {
        self.ensure_open()?;
        let started = Instant::now();
        validate_ids(environment_ids, self.states.len())?;
        let validation = started.elapsed();
        self.materialize(
            environment_ids,
            validation,
            Duration::ZERO,
            false,
            0,
            0,
            Vec::new(),
        )
    }

    /// Clone one search state into target slots without changing its hidden
    /// state or advancing it. This is used to expand a later decision after a
    /// root wall particle has already been sampled.
    pub fn fork_search_state(
        &mut self,
        source_environment_id: u32,
        target_environment_ids: &[u32],
    ) -> Result<BatchTransition, EnvError> {
        self.ensure_open()?;
        let validation_started = Instant::now();
        if source_environment_id as usize >= self.states.len() {
            return Err(EnvError::EnvironmentOutOfRange(source_environment_id));
        }
        if target_environment_ids.is_empty() {
            return Err(EnvError::InvalidArgument(
                "search state fork requires at least one target branch".into(),
            ));
        }
        validate_ids(target_environment_ids, self.states.len())?;
        if target_environment_ids.contains(&source_environment_id) {
            return Err(EnvError::InvalidArgument(
                "search state fork cannot overwrite its source".into(),
            ));
        }
        let root = self.states[source_environment_id as usize].clone();
        if root
            .hanchan
            .as_ref()
            .and_then(|hanchan| hanchan.hand.decision.as_ref())
            .is_none()
        {
            return Err(EnvError::InvalidArgument(
                "search state fork source must be at a decision".into(),
            ));
        }
        let validation = validation_started.elapsed();
        let env_started = Instant::now();
        let replacements = self.pool.install(|| {
            target_environment_ids
                .par_iter()
                .map(|&target| {
                    let mut state = root.clone();
                    rebind_environment(&mut state, target);
                    (target, state)
                })
                .collect::<Vec<_>>()
        });
        for (target, state) in replacements {
            self.states[target as usize] = state;
        }
        let env_step = env_started.elapsed();
        self.materialize(
            target_environment_ids,
            validation,
            env_step,
            false,
            0,
            0,
            Vec::new(),
        )
    }

    /// Clone one decision state into target slots with search-only hidden-wall
    /// particles. Branches sharing a particle key receive the same physical
    /// position-to-tile chance table. No automatic transition is advanced.
    pub fn fork_privileged_wall(
        &mut self,
        source_environment_id: u32,
        branches: &[(u32, u64)],
    ) -> Result<BatchTransition, EnvError> {
        self.ensure_open()?;
        let validation_started = Instant::now();
        if source_environment_id as usize >= self.states.len() {
            return Err(EnvError::EnvironmentOutOfRange(source_environment_id));
        }
        if branches.is_empty() {
            return Err(EnvError::InvalidArgument(
                "search fork requires at least one target branch".into(),
            ));
        }
        let target_ids = branches
            .iter()
            .map(|&(target, _)| target)
            .collect::<Vec<_>>();
        validate_ids(&target_ids, self.states.len())?;
        let root = self.states[source_environment_id as usize].clone();
        if root
            .hanchan
            .as_ref()
            .and_then(|hanchan| hanchan.hand.decision.as_ref())
            .is_none()
        {
            return Err(EnvError::InvalidArgument(
                "search fork source must be at a decision".into(),
            ));
        }
        let validation = validation_started.elapsed();
        let env_started = Instant::now();
        let replacements = self.pool.install(|| {
            branches
                .par_iter()
                .map(|&(target, particle_key)| {
                    fork_with_privileged_wall(&root, target, particle_key)
                        .map(|state| (target, state))
                        .map_err(|error| EnvError::InvalidArgument(error.to_string()))
                })
                .collect::<Result<Vec<_>, _>>()
        })?;
        for (target, state) in replacements {
            self.states[target as usize] = state;
        }
        let env_step = env_started.elapsed();
        self.materialize(&target_ids, validation, env_step, false, 0, 0, Vec::new())
    }

    /// Clone an isolated observer self-turn into target slots while jointly
    /// resampling opponent concealed tiles and the unresolved wall.
    pub fn fork_public_information(
        &mut self,
        source_environment_id: u32,
        observer_seat: u8,
        branches: &[(u32, u64)],
    ) -> Result<BatchTransition, EnvError> {
        self.ensure_open()?;
        let validation_started = Instant::now();
        if source_environment_id as usize >= self.states.len() {
            return Err(EnvError::EnvironmentOutOfRange(source_environment_id));
        }
        if branches.is_empty() {
            return Err(EnvError::InvalidArgument(
                "public search fork requires at least one target branch".into(),
            ));
        }
        let target_ids = branches
            .iter()
            .map(|&(target, _)| target)
            .collect::<Vec<_>>();
        validate_ids(&target_ids, self.states.len())?;
        let root = self.states[source_environment_id as usize].clone();
        let validation = validation_started.elapsed();
        let env_started = Instant::now();
        let replacements = self.pool.install(|| {
            branches
                .par_iter()
                .map(|&(target, particle_key)| {
                    fork_with_public_information(&root, target, observer_seat, particle_key)
                        .map(|state| (target, state))
                        .map_err(|error| EnvError::InvalidArgument(error.to_string()))
                })
                .collect::<Result<Vec<_>, _>>()
        })?;
        for (target, state) in replacements {
            self.states[target as usize] = state;
        }
        let env_step = env_started.elapsed();
        self.materialize(&target_ids, validation, env_step, false, 0, 0, Vec::new())
    }

    pub fn load_hanchan(&mut self, values: &[ReplayHanchan]) -> Result<BatchTransition, EnvError> {
        self.ensure_open()?;
        let validation_started = Instant::now();
        let ids = values
            .iter()
            .map(|value| value.environment_id)
            .collect::<Vec<_>>();
        validate_ids(&ids, self.states.len())?;
        let validation = validation_started.elapsed();
        let env_started = Instant::now();
        let replacements = self.pool.install(|| {
            values
                .par_iter()
                .map(|value| {
                    let mut state = self.states[value.environment_id as usize].clone();
                    state.load_replay_hanchan(value)?;
                    Ok::<_, riichi_core::error::CoreError>((value.environment_id, state))
                })
                .collect::<Result<Vec<_>, _>>()
        })?;
        for (id, state) in replacements {
            self.states[id as usize] = state;
        }
        let env_step = env_started.elapsed();
        self.materialize(&ids, validation, env_step, true, 0, 0, Vec::new())
    }

    pub fn apply_events(&mut self, events: &[ReplayEvent]) -> Result<BatchTransition, EnvError> {
        self.ensure_open()?;
        let validation_started = Instant::now();
        if events.is_empty() {
            return Err(EnvError::InvalidArgument(
                "events must contain at least one replay event".into(),
            ));
        }
        let mut grouped: BTreeMap<u32, Vec<ReplayEvent>> = BTreeMap::new();
        for event in events {
            if event.environment_id as usize >= self.states.len() {
                return Err(EnvError::EnvironmentOutOfRange(event.environment_id));
            }
            grouped
                .entry(event.environment_id)
                .or_default()
                .push(event.clone());
        }
        let validation = validation_started.elapsed();
        let env_started = Instant::now();
        let grouped = grouped.into_iter().collect::<Vec<_>>();
        let replacements = self.pool.install(|| {
            grouped
                .par_iter()
                .map(|(id, rows)| {
                    let mut state = self.states[*id as usize].clone();
                    let actions = state.apply_replay_events(rows)?;
                    Ok::<_, riichi_core::error::CoreError>((*id, state, actions))
                })
                .collect::<Result<Vec<_>, _>>()
        })?;
        let mut applied_selections = Vec::new();
        for (id, state, actions) in replacements {
            self.states[id as usize] = state;
            applied_selections.extend(actions);
        }
        let env_step = env_started.elapsed();
        let ids = grouped.iter().map(|(id, _)| *id).collect::<Vec<_>>();
        self.materialize(
            &ids,
            validation,
            env_step,
            true,
            applied_selections.len() as u64,
            0,
            applied_selections,
        )
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
            if state.externally_loaded {
                state.stabilize_replay_decisions();
            } else {
                state.stabilize_automatic_decisions();
            }
            automatic += state.automatic_decisions.saturating_sub(before);
            self.states[id as usize] = state;
        }
        let env_step = env_started.elapsed();
        self.finish(&ids, validation, env_step, 0, automatic, Vec::new())
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
        applied_selections: Vec<ActionSelection>,
    ) -> Result<BatchTransition, EnvError> {
        self.materialize(
            ids,
            validation,
            env_step,
            true,
            action_count,
            automatic_count,
            applied_selections,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn materialize(
        &mut self,
        ids: &[u32],
        validation: Duration,
        env_step: Duration,
        drain_events: bool,
        action_count: u64,
        automatic_count: u64,
        applied_selections: Vec<ActionSelection>,
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
            applied_selections,
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
    use riichi_core::game::phase::HandPhase;

    #[test]
    fn restore_stabilizes_a_frame_free_settlement_before_materializing() {
        let mut state = GameState::new(0);
        state.reset_from_seed(7);
        state.take_events();
        let hand = &mut state.hanchan.as_mut().unwrap().hand;
        hand.phase = HandPhase::Settlement;
        hand.decision = None;
        let encoded = snapshot::encode(&state).unwrap();

        let mut env = BatchEnv::new(1, 7, 1).unwrap();
        let transition = env.restore(&BTreeMap::from([(0, encoded)])).unwrap();
        let restored = &transition.states[0];

        assert!(
            restored.lifecycle == EnvironmentLifecycle::Complete
                || restored.hanchan.as_ref().unwrap().hand.decision.is_some(),
            "restore returned an active state without a queryable decision"
        );
        assert!(transition
            .events
            .iter()
            .any(|event| event.kind == EventKind::EndKyoku));
    }

    #[test]
    fn restore_preserves_an_external_settlement_boundary() {
        let mut state = GameState::new(0);
        state.reset_from_seed(17);
        state.take_events();
        state.externally_loaded = true;
        let hand = &mut state.hanchan.as_mut().unwrap().hand;
        hand.phase = HandPhase::Settlement;
        hand.decision = None;
        let encoded = snapshot::encode(&state).unwrap();

        let mut env = BatchEnv::new(1, 7, 1).unwrap();
        let transition = env.restore(&BTreeMap::from([(0, encoded)])).unwrap();
        let restored = &transition.states[0];

        assert!(restored.externally_loaded);
        assert_eq!(
            restored.hanchan.as_ref().unwrap().hand.phase,
            HandPhase::Settlement,
        );
        assert!(transition.events.is_empty());
        assert_eq!(
            env.snapshots(&[0]).unwrap()[&0],
            snapshot::encode(restored).unwrap(),
        );
    }

    #[test]
    fn step_materializes_before_a_frame_free_reaction_is_advanced() {
        let mut env = BatchEnv::new(1, 7, 1).unwrap();
        let mut transition = env.reset(&[0]).unwrap();
        for _ in 0..64 {
            let state = &transition.states[0];
            if state
                .hanchan
                .as_ref()
                .and_then(|game| game.hand.decision.as_ref())
                .is_none()
            {
                assert!(!transition.events.is_empty());
                let before = snapshot::encode(state).unwrap();
                transition = env.advance(&[0]).unwrap();
                assert!(!transition.events.is_empty());
                assert_ne!(before, snapshot::encode(&transition.states[0]).unwrap());
                return;
            }
            let frame = state
                .hanchan
                .as_ref()
                .unwrap()
                .hand
                .decision
                .as_ref()
                .unwrap();
            let actions = frame
                .action_spaces
                .iter()
                .map(|decision| ActionSelection::bind(frame, decision.seat, 0))
                .collect::<Vec<_>>();
            transition = env.step(&actions).unwrap();
        }
        panic!("seed did not produce a frame-free reaction within the bound")
    }

    #[test]
    fn privileged_wall_fork_pairs_particle_across_target_slots() {
        let mut env = BatchEnv::new(4, 7, 1).unwrap();
        env.reset(&[0]).unwrap();
        let transition = env
            .fork_privileged_wall(0, &[(1, 11), (2, 11), (3, 12)])
            .unwrap();
        assert_eq!(transition.states.len(), 3);
        let first = &transition.states[0];
        let second = &transition.states[1];
        let third = &transition.states[2];
        let first_wall = first.hanchan.as_ref().unwrap().hand.wall.tiles;
        let second_wall = second.hanchan.as_ref().unwrap().hand.wall.tiles;
        let third_wall = third.hanchan.as_ref().unwrap().hand.wall.tiles;
        assert_eq!(first_wall, second_wall);
        assert_ne!(first_wall, third_wall);
        assert_eq!(
            first
                .hanchan
                .as_ref()
                .unwrap()
                .hand
                .decision
                .as_ref()
                .unwrap()
                .environment_id,
            1,
        );
        assert!(transition.events.is_empty());
    }

    #[test]
    fn search_state_fork_preserves_sampled_hidden_state() {
        let mut env = BatchEnv::new(4, 7, 1).unwrap();
        env.reset(&[0]).unwrap();
        env.fork_privileged_wall(0, &[(1, 11)]).unwrap();
        let source = env.inspect(&[1]).unwrap().states[0].clone();
        let transition = env.fork_search_state(1, &[2, 3]).unwrap();
        assert_eq!(transition.states.len(), 2);
        for state in &transition.states {
            assert_eq!(
                state.hanchan.as_ref().unwrap().hand.wall,
                source.hanchan.as_ref().unwrap().hand.wall,
            );
            assert_eq!(state.rng, source.rng);
            assert_eq!(
                state
                    .hanchan
                    .as_ref()
                    .unwrap()
                    .hand
                    .decision
                    .as_ref()
                    .unwrap()
                    .environment_id,
                state.environment_id,
            );
        }
        assert!(transition.events.is_empty());
    }

    #[test]
    fn public_information_fork_pairs_complete_hidden_samples() {
        let mut env = BatchEnv::new(4, 7, 1).unwrap();
        let root = env.reset(&[0]).unwrap().states[0].clone();
        let observer = root.hanchan.as_ref().unwrap().hand.current_seat;
        let observer_hand = root.hanchan.as_ref().unwrap().players[usize::from(observer)]
            .concealed_tiles
            .clone();
        let transition = env
            .fork_public_information(0, observer, &[(1, 31), (2, 31), (3, 32)])
            .unwrap();
        let first = &transition.states[0];
        let second = &transition.states[1];
        let third = &transition.states[2];
        assert_eq!(
            first.hanchan.as_ref().unwrap().hand.wall,
            second.hanchan.as_ref().unwrap().hand.wall,
        );
        assert_eq!(
            first.hanchan.as_ref().unwrap().players,
            second.hanchan.as_ref().unwrap().players,
        );
        assert_ne!(
            first.hanchan.as_ref().unwrap().hand.wall,
            third.hanchan.as_ref().unwrap().hand.wall,
        );
        assert_eq!(
            first.hanchan.as_ref().unwrap().players[usize::from(observer)].concealed_tiles,
            observer_hand,
        );
        assert!(transition.events.is_empty());
    }
}
