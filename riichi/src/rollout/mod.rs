mod encoding;

use std::collections::{BTreeMap, BTreeSet};

use riichi_core::{
    game::{event::EventRecord, phase::EnvironmentLifecycle},
    ActionCandidate, ActionSelection, EventKind, GameState,
};

use crate::batch_env::{BatchEnv, BatchTransition, EnvError};

use encoding::{boundary_from_start_event, conservative_action};
pub use encoding::{encode_action, encode_event, encode_observation, RankBoundary, TokenRow};

type MatchId = (u32, u64);
type DecisionKey = (u32, u64, u64, u8);

#[derive(Clone, Debug, Default)]
struct KyokuGameplay {
    riichi_mask: u8,
    open_mask: u8,
    dealt_in_mask: u8,
    discards: [u16; 4],
}

#[derive(Clone, Debug, Default)]
struct MatchGameplay {
    kyoku: u32,
    exhaustive_ryukyoku: u32,
    wins: [u32; 4],
    deal_ins: [u32; 4],
    riichi_hands: [u32; 4],
    calling_hands: [u32; 4],
    tsumo_wins: [u32; 4],
    dama_wins: [u32; 4],
    winning_points: [i64; 4],
    winning_point_events: [u32; 4],
    deal_in_points: [i64; 4],
    deal_in_point_events: [u32; 4],
    winning_turns: [u64; 4],
    winning_turn_events: [u32; 4],
}

#[derive(Clone, Copy, Debug)]
struct Lineup {
    policy_slots: [u32; 4],
    learner_mask: u8,
}

#[derive(Clone, Debug)]
struct PendingRow {
    row_id: u64,
    enqueue_sequence: u64,
    environment_id: u32,
    episode_generation: u64,
    frame_id: u64,
    seat: u8,
    policy_slot: u32,
    eligible: bool,
    phase: u8,
    tokens: Vec<TokenRow>,
    actor_query_offset: usize,
    decision_seat: u8,
    rank_boundary_features: [f32; 28],
    boundary_group_id: u64,
    actions: Vec<[u8; 15]>,
    native_actions: Vec<ActionCandidate>,
    native_representatives: Vec<u32>,
}

impl PendingRow {
    fn decision_key(&self) -> DecisionKey {
        (
            self.environment_id,
            self.episode_generation,
            self.frame_id,
            self.seat,
        )
    }

    fn queue_key(&self, context_tokens: usize) -> QueueKey {
        QueueKey {
            policy_slot: self.policy_slot,
            sequence_bucket: sequence_bucket(self.tokens.len(), context_tokens),
            action_bucket: action_bucket(self.actions.len()),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct QueueKey {
    policy_slot: u32,
    sequence_bucket: usize,
    action_bucket: usize,
}

#[derive(Clone, Debug)]
pub struct InferenceRequest {
    pub request_id: u64,
    pub policy_slot: u32,
    pub sequence_bucket: usize,
    pub action_bucket: usize,
    pub row_ids: Vec<u64>,
    pub environment_ids: Vec<u32>,
    pub episode_generations: Vec<u64>,
    pub frame_ids: Vec<u64>,
    pub seats: Vec<u8>,
    pub token_factors: Vec<i32>,
    pub token_numeric: Vec<f32>,
    pub lengths: Vec<i64>,
    pub query_offsets: Vec<i64>,
    pub decision_seats: Vec<i64>,
    pub rank_boundary_features: Vec<f32>,
    pub action_factors: Vec<i32>,
    pub action_lengths: Vec<i64>,
    pub action_offsets: Vec<i64>,
}

impl InferenceRequest {
    pub fn rows(&self) -> usize {
        self.row_ids.len()
    }
}

#[derive(Clone, Debug, Default)]
pub struct RolloutChunk {
    pub row_ids: Vec<u64>,
    pub environment_ids: Vec<u32>,
    pub episode_generations: Vec<u64>,
    pub frame_ids: Vec<u64>,
    pub seats: Vec<u8>,
    pub policy_slots: Vec<u32>,
    pub eligibility: Vec<u8>,
    pub phases: Vec<u8>,
    pub selected_groups: Vec<u32>,
    pub selected_native: Vec<u32>,
    pub old_logp: Vec<f32>,
    pub old_state_values: Vec<f32>,
    pub token_offsets: Vec<u64>,
    pub token_factors: Vec<u8>,
    pub token_numeric: Vec<f32>,
    pub query_offsets: Vec<u32>,
    pub decision_seats: Vec<u8>,
    pub rank_boundary_features: Vec<f32>,
    pub boundary_group_ids: Vec<u64>,
    pub old_boundary_values: Vec<f32>,
    pub terminal_placements: Vec<i8>,
    pub rank_order_targets: Vec<i8>,
    pub kyoku_boundary: Vec<u8>,
    pub match_boundary: Vec<u8>,
    pub rank_boundary_supervision: Vec<u8>,
    pub advantages: Vec<f32>,
    pub normalized_advantages: Vec<f32>,
    pub value_targets: Vec<f32>,
    pub action_offsets: Vec<u64>,
    pub action_factors: Vec<u8>,
    pub terminal_environment_ids: Vec<u32>,
    pub terminal_episode_generations: Vec<u64>,
    pub terminal_scores: Vec<i32>,
    pub terminal_ranks: Vec<u8>,
    pub terminal_completed_kyoku: Vec<u32>,
    pub terminal_exhaustive_ryukyoku: Vec<u32>,
    pub terminal_wins: Vec<u32>,
    pub terminal_deal_ins: Vec<u32>,
    pub terminal_riichi_hands: Vec<u32>,
    pub terminal_calling_hands: Vec<u32>,
    pub terminal_tsumo_wins: Vec<u32>,
    pub terminal_dama_wins: Vec<u32>,
    pub terminal_winning_points: Vec<i64>,
    pub terminal_winning_point_events: Vec<u32>,
    pub terminal_deal_in_points: Vec<i64>,
    pub terminal_deal_in_point_events: Vec<u32>,
    pub terminal_winning_turns: Vec<u64>,
    pub terminal_winning_turn_events: Vec<u32>,
    pub kyoku_completions: u64,
    pub match_completions: u64,
    // Completion events arrive once per match, after that match's decision
    // rows have already been appended. Keep their indices here so terminal
    // annotation is linear in the trajectory being completed instead of
    // repeatedly scanning every row accumulated by every environment.
    match_rows: BTreeMap<MatchId, Vec<usize>>,
}

impl RolloutChunk {
    fn new() -> Self {
        Self {
            token_offsets: vec![0],
            action_offsets: vec![0],
            ..Self::default()
        }
    }

    pub fn rows(&self) -> usize {
        self.row_ids.len()
    }

    fn push(
        &mut self,
        row: &PendingRow,
        selected_group: usize,
        old_logp: f32,
        old_state_value: f32,
    ) {
        let index = self.rows();
        self.match_rows
            .entry((row.environment_id, row.episode_generation))
            .or_default()
            .push(index);
        self.row_ids.push(row.row_id);
        self.environment_ids.push(row.environment_id);
        self.episode_generations.push(row.episode_generation);
        self.frame_ids.push(row.frame_id);
        self.seats.push(row.seat);
        self.policy_slots.push(row.policy_slot);
        self.eligibility.push(u8::from(row.eligible));
        self.phases.push(row.phase);
        self.selected_groups.push(selected_group as u32);
        self.selected_native
            .push(row.native_representatives[selected_group]);
        self.old_logp.push(old_logp);
        self.old_state_values.push(old_state_value);
        for token in &row.tokens {
            self.token_factors.extend_from_slice(&token.categorical);
            self.token_numeric.extend_from_slice(&token.numeric);
        }
        self.token_offsets
            .push((self.token_factors.len() / 10) as u64);
        self.query_offsets.push(row.actor_query_offset as u32);
        self.decision_seats.push(row.decision_seat);
        self.rank_boundary_features
            .extend_from_slice(&row.rank_boundary_features);
        self.boundary_group_ids.push(row.boundary_group_id);
        self.old_boundary_values.push(f32::NAN);
        self.terminal_placements.push(-1);
        self.rank_order_targets.push(-1);
        self.kyoku_boundary.push(0);
        self.match_boundary.push(0);
        self.rank_boundary_supervision.push(0);
        self.advantages.push(f32::NAN);
        self.normalized_advantages.push(f32::NAN);
        self.value_targets.push(f32::NAN);
        for factors in &row.actions {
            self.action_factors.extend_from_slice(factors);
        }
        self.action_offsets
            .push((self.action_factors.len() / 15) as u64);
    }

    pub fn set_boundary_values(
        &mut self,
        group_ids: &[u64],
        values: &[f32],
    ) -> Result<(), EnvError> {
        if group_ids.len() != values.len() {
            return Err(EnvError::InvalidArgument(
                "boundary group IDs and values must have identical lengths".into(),
            ));
        }
        let supplied = group_ids
            .iter()
            .copied()
            .zip(values.iter().copied())
            .collect::<BTreeMap<_, _>>();
        if supplied.len() != group_ids.len() || values.iter().any(|value| !value.is_finite()) {
            return Err(EnvError::InvalidArgument(
                "boundary values must be finite and group IDs unique".into(),
            ));
        }
        for (index, group) in self.boundary_group_ids.iter().enumerate() {
            if let Some(value) = supplied.get(group) {
                self.old_boundary_values[index] = *value;
            }
        }
        let known = self
            .boundary_group_ids
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        if supplied.keys().any(|group| !known.contains(group)) {
            return Err(EnvError::InvalidArgument(
                "boundary value submission contains an unknown group ID".into(),
            ));
        }
        Ok(())
    }

    pub fn finish_targets(&mut self, gae_lambda: f32) -> Result<(), EnvError> {
        if !gae_lambda.is_finite() || !(0.0..=1.0).contains(&gae_lambda) {
            return Err(EnvError::InvalidArgument(
                "GAE lambda must be finite and in [0, 1]".into(),
            ));
        }
        let eligible = self
            .eligibility
            .iter()
            .enumerate()
            .filter_map(|(index, &value)| (value != 0).then_some(index))
            .collect::<Vec<_>>();
        if eligible.iter().any(|&index| {
            !self.old_boundary_values[index].is_finite()
                || !self.old_state_values[index].is_finite()
        }) {
            return Err(EnvError::InvalidArgument(
                "every learner decision and boundary group needs a critic value".into(),
            ));
        }
        let mut trajectories = BTreeMap::<(u32, u64, u8, u32), Vec<usize>>::new();
        for &index in &eligible {
            trajectories
                .entry((
                    self.environment_ids[index],
                    self.episode_generations[index],
                    self.seats[index],
                    self.policy_slots[index],
                ))
                .or_default()
                .push(index);
        }
        for indices in trajectories.values() {
            let placement = indices
                .iter()
                .find_map(|&index| {
                    (self.terminal_placements[index] >= 0)
                        .then_some(self.terminal_placements[index])
                })
                .ok_or_else(|| {
                    EnvError::InvalidArgument("learner trajectory has no terminal placement".into())
                })?;
            let terminal = [1.0_f32, 1.0 / 3.0, -1.0 / 3.0, -1.0][placement as usize];
            let mut segments = Vec::<(u64, Vec<usize>)>::new();
            for &index in indices {
                let group = self.boundary_group_ids[index];
                if segments
                    .last()
                    .is_none_or(|(previous, _)| *previous != group)
                {
                    segments.push((group, vec![index]));
                } else {
                    segments.last_mut().expect("segment").1.push(index);
                }
            }
            for segment in 0..segments.len() {
                let critic_end = segments
                    .get(segment + 1)
                    .map_or(terminal, |(_, rows)| self.old_boundary_values[rows[0]]);
                let rows = &segments[segment].1;
                let mut accumulator = 0.0_f32;
                for position in (0..rows.len()).rev() {
                    let index = rows[position];
                    let following = rows
                        .get(position + 1)
                        .map_or(critic_end, |&next| self.old_state_values[next]);
                    self.value_targets[index] = critic_end;
                    let delta = following - self.old_state_values[index];
                    accumulator = delta + gae_lambda * accumulator;
                    self.advantages[index] = accumulator;
                }
            }
        }
        if !eligible.is_empty() {
            let mean = eligible
                .iter()
                .map(|&index| f64::from(self.advantages[index]))
                .sum::<f64>()
                / eligible.len() as f64;
            let variance = eligible
                .iter()
                .map(|&index| (f64::from(self.advantages[index]) - mean).powi(2))
                .sum::<f64>()
                / eligible.len() as f64;
            let deviation = variance.sqrt();
            for &index in &eligible {
                self.normalized_advantages[index] = if deviation < 1e-8 {
                    0.0
                } else {
                    ((f64::from(self.advantages[index]) - mean) / (deviation + 1e-8)) as f32
                };
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default)]
pub struct RolloutMetrics {
    pub inference_requests: u64,
    pub inference_rows: u64,
    pub useful_tokens: u64,
    pub padded_tokens: u64,
    pub useful_actions: u64,
    pub padded_actions: u64,
    pub native_bot_rows: u64,
    pub automatic_rows: u64,
    pub env_calls: u64,
    pub compile_fallbacks: u64,
}

pub struct RolloutEngine {
    env: BatchEnv,
    context_tokens: usize,
    token_budget: usize,
    inference_only: bool,
    active_matches: BTreeSet<MatchId>,
    completed_matches: BTreeSet<MatchId>,
    lineups: BTreeMap<MatchId, Lineup>,
    bot_policy_slots: BTreeSet<u32>,
    latest_states: Vec<Option<GameState>>,
    history_generation: Vec<u64>,
    histories: Vec<[Vec<TokenRow>; 4]>,
    boundaries: Vec<Option<RankBoundary>>,
    kyoku_gameplay: Vec<Option<KyokuGameplay>>,
    match_gameplay: Vec<MatchGameplay>,
    boundary_groups: BTreeMap<(MatchId, RankBoundary, u8, u32), u64>,
    next_boundary_group_id: u64,
    pending: Vec<PendingRow>,
    active_request: Option<(u64, Vec<PendingRow>)>,
    selections: BTreeMap<DecisionKey, ActionSelection>,
    next_row_id: u64,
    next_enqueue_sequence: u64,
    next_request_id: u64,
    chunk: RolloutChunk,
    metrics: RolloutMetrics,
}

impl RolloutEngine {
    pub fn new(
        num_envs: usize,
        master_seed: u64,
        num_threads: usize,
        context_tokens: usize,
        token_budget: usize,
        inference_only: bool,
    ) -> Result<Self, EnvError> {
        Self::new_with_rules_profile(
            num_envs,
            master_seed,
            num_threads,
            context_tokens,
            token_budget,
            inference_only,
            riichi_core::RULES_PROFILE_ID,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_with_rules_profile(
        num_envs: usize,
        master_seed: u64,
        num_threads: usize,
        context_tokens: usize,
        token_budget: usize,
        inference_only: bool,
        rules_profile_id: u32,
    ) -> Result<Self, EnvError> {
        if context_tokens < 64 {
            return Err(EnvError::InvalidArgument(
                "context_tokens must be at least 64".into(),
            ));
        }
        if token_budget < 64 {
            return Err(EnvError::InvalidArgument(
                "token_budget must be at least 64".into(),
            ));
        }
        Ok(Self {
            env: BatchEnv::new_with_rules_profile(
                num_envs,
                master_seed,
                num_threads,
                rules_profile_id,
            )?,
            context_tokens,
            token_budget,
            inference_only,
            active_matches: BTreeSet::new(),
            completed_matches: BTreeSet::new(),
            lineups: BTreeMap::new(),
            bot_policy_slots: BTreeSet::new(),
            latest_states: vec![None; num_envs],
            history_generation: vec![0; num_envs],
            histories: (0..num_envs)
                .map(|_| std::array::from_fn(|_| Vec::new()))
                .collect(),
            boundaries: vec![None; num_envs],
            kyoku_gameplay: vec![None; num_envs],
            match_gameplay: vec![MatchGameplay::default(); num_envs],
            boundary_groups: BTreeMap::new(),
            next_boundary_group_id: 0,
            pending: Vec::new(),
            active_request: None,
            selections: BTreeMap::new(),
            next_row_id: 0,
            next_enqueue_sequence: 0,
            next_request_id: 0,
            chunk: RolloutChunk::new(),
            metrics: RolloutMetrics::default(),
        })
    }

    pub fn num_envs(&self) -> usize {
        self.latest_states.len()
    }

    pub fn reset_chunk(&mut self, target_matches: usize) -> Result<Vec<MatchId>, EnvError> {
        if target_matches == 0 || target_matches > self.num_envs() {
            return Err(EnvError::InvalidArgument(format!(
                "target_matches must be in 1..={}, got {target_matches}",
                self.num_envs()
            )));
        }
        if self.active_request.is_some() {
            return Err(EnvError::InvalidArgument(
                "cannot reset a chunk with an active inference request".into(),
            ));
        }
        if !self.active_matches.is_empty() && !self.is_complete() {
            return Err(EnvError::InvalidArgument(
                "cannot reset an incomplete rollout chunk".into(),
            ));
        }
        if self.chunk.match_completions != 0 {
            return Err(EnvError::InvalidArgument(
                "take the completed rollout chunk before resetting".into(),
            ));
        }
        let ids = (0..target_matches as u32).collect::<Vec<_>>();
        self.active_matches.clear();
        self.completed_matches.clear();
        self.lineups.clear();
        self.bot_policy_slots.clear();
        self.pending.clear();
        self.boundary_groups.clear();
        self.next_boundary_group_id = 0;
        self.selections.clear();
        self.chunk = RolloutChunk::new();
        self.metrics = RolloutMetrics::default();
        for &id in &ids {
            self.histories[id as usize] = std::array::from_fn(|_| Vec::new());
            self.boundaries[id as usize] = None;
            self.latest_states[id as usize] = None;
            self.kyoku_gameplay[id as usize] = None;
            self.match_gameplay[id as usize] = MatchGameplay::default();
        }
        let transition = self.env.reset(&ids)?;
        self.metrics.env_calls += 1;
        self.ingest_transition(transition)?;
        let matches = ids
            .into_iter()
            .map(|id| {
                let state = self.latest_states[id as usize]
                    .as_ref()
                    .expect("reset materializes every requested environment");
                (id, state.episode_generation)
            })
            .collect::<Vec<_>>();
        self.active_matches.extend(matches.iter().copied());
        Ok(matches)
    }

    pub fn reset_chunk_seeded(&mut self, seeds: &[u64]) -> Result<Vec<MatchId>, EnvError> {
        let target_matches = seeds.len();
        if target_matches == 0 || target_matches > self.num_envs() {
            return Err(EnvError::InvalidArgument(format!(
                "seed count must be in 1..={}, got {target_matches}",
                self.num_envs()
            )));
        }
        if self.active_request.is_some() {
            return Err(EnvError::InvalidArgument(
                "cannot reset a chunk with an active inference request".into(),
            ));
        }
        if !self.active_matches.is_empty() && !self.is_complete() {
            return Err(EnvError::InvalidArgument(
                "cannot reset an incomplete rollout chunk".into(),
            ));
        }
        if self.chunk.match_completions != 0 {
            return Err(EnvError::InvalidArgument(
                "take the completed rollout chunk before resetting".into(),
            ));
        }
        let ids = (0..target_matches as u32).collect::<Vec<_>>();
        self.active_matches.clear();
        self.completed_matches.clear();
        self.lineups.clear();
        self.bot_policy_slots.clear();
        self.pending.clear();
        self.boundary_groups.clear();
        self.next_boundary_group_id = 0;
        self.selections.clear();
        self.chunk = RolloutChunk::new();
        self.metrics = RolloutMetrics::default();
        for &id in &ids {
            self.histories[id as usize] = std::array::from_fn(|_| Vec::new());
            self.boundaries[id as usize] = None;
            self.latest_states[id as usize] = None;
            self.kyoku_gameplay[id as usize] = None;
            self.match_gameplay[id as usize] = MatchGameplay::default();
        }
        let transition = self.env.reset_with_seeds(
            &ids.iter()
                .copied()
                .zip(seeds.iter().copied())
                .collect::<Vec<_>>(),
        )?;
        self.metrics.env_calls += 1;
        self.ingest_transition(transition)?;
        let matches = ids
            .into_iter()
            .map(|id| {
                let state = self.latest_states[id as usize]
                    .as_ref()
                    .expect("reset materializes every requested environment");
                (id, state.episode_generation)
            })
            .collect::<Vec<_>>();
        self.active_matches.extend(matches.iter().copied());
        Ok(matches)
    }

    pub fn register_lineups(
        &mut self,
        matches: &[MatchId],
        policy_slots: &[[u32; 4]],
        learner_masks: &[u8],
        bot_policy_slots: &[u32],
    ) -> Result<(), EnvError> {
        if matches.len() != policy_slots.len() || matches.len() != learner_masks.len() {
            return Err(EnvError::InvalidArgument(
                "lineup columns must have identical lengths".into(),
            ));
        }
        if matches.iter().copied().collect::<BTreeSet<_>>() != self.active_matches {
            return Err(EnvError::InvalidArgument(
                "lineups must cover every active match exactly once".into(),
            ));
        }
        self.lineups = matches
            .iter()
            .copied()
            .zip(policy_slots.iter().copied())
            .zip(learner_masks.iter().copied())
            .map(|((match_id, policy_slots), learner_mask)| {
                (
                    match_id,
                    Lineup {
                        policy_slots,
                        learner_mask: learner_mask & 0b1111,
                    },
                )
            })
            .collect();
        self.bot_policy_slots = bot_policy_slots.iter().copied().collect();
        self.rebuild_pending()?;
        self.drive_native_bots()?;
        Ok(())
    }

    pub fn next_request(&mut self) -> Result<Option<InferenceRequest>, EnvError> {
        if self.active_request.is_some() {
            return Err(EnvError::InvalidArgument(
                "submit the active inference request before requesting another".into(),
            ));
        }
        if self.lineups.len() != self.active_matches.len() {
            return Err(EnvError::InvalidArgument(
                "register lineups before requesting inference".into(),
            ));
        }
        self.drive_native_bots()?;
        if self.pending.is_empty() {
            if self.is_complete() {
                return Ok(None);
            }
            return Err(EnvError::InvalidArgument(
                "scheduler made no progress with active matches".into(),
            ));
        }

        let mut queues = BTreeMap::<QueueKey, Vec<usize>>::new();
        for (index, row) in self.pending.iter().enumerate() {
            queues
                .entry(row.queue_key(self.context_tokens))
                .or_default()
                .push(index);
        }
        for indices in queues.values_mut() {
            indices.sort_by_key(|&index| {
                let row = &self.pending[index];
                (
                    row.enqueue_sequence,
                    row.environment_id,
                    row.frame_id,
                    row.seat,
                )
            });
        }
        let full = queues
            .iter()
            .filter(|(key, indices)| {
                indices.len() >= (self.token_budget / key.sequence_bucket).max(1)
            })
            .collect::<Vec<_>>();
        let (key, selected_indices) = if full.is_empty() {
            // Every environment is waiting on inference, so no native work can
            // create more rows.  Include the globally oldest row, then fill
            // the same-policy request across shape queues up to the token
            // budget.  This keeps deterministic starvation behavior without
            // launching many mostly-empty adjacent buckets.
            let oldest = (0..self.pending.len())
                .min_by_key(|&index| {
                    let row = &self.pending[index];
                    (
                        row.enqueue_sequence,
                        row.policy_slot,
                        row.queue_key(self.context_tokens),
                    )
                })
                .expect("pending rows are nonempty");
            let policy_slot = self.pending[oldest].policy_slot;
            let oldest_shape = self.pending[oldest].queue_key(self.context_tokens);
            let mut candidates = (0..self.pending.len())
                .filter(|&index| self.pending[index].policy_slot == policy_slot)
                .collect::<Vec<_>>();
            candidates.sort_by_key(|&index| {
                let row = &self.pending[index];
                let shape = row.queue_key(self.context_tokens);
                (
                    index != oldest,
                    shape.sequence_bucket.abs_diff(oldest_shape.sequence_bucket),
                    shape.action_bucket.abs_diff(oldest_shape.action_bucket),
                    row.enqueue_sequence,
                    row.environment_id,
                    row.frame_id,
                    row.seat,
                )
            });
            let mut selected = BTreeSet::new();
            let mut sequence_bucket = 0;
            let mut action_bucket = 0;
            let mut useful_tokens = 0_usize;
            for index in candidates {
                let shape = self.pending[index].queue_key(self.context_tokens);
                let next_sequence = sequence_bucket.max(shape.sequence_bucket);
                let next_useful = useful_tokens + self.pending[index].tokens.len();
                let next_padded = (selected.len() + 1) * next_sequence;
                let padding_fraction = (next_padded - next_useful) as f64 / next_padded as f64;
                if !selected.is_empty()
                    && (next_padded > self.token_budget || padding_fraction > 0.35)
                {
                    continue;
                }
                selected.insert(index);
                sequence_bucket = next_sequence;
                action_bucket = action_bucket.max(shape.action_bucket);
                useful_tokens = next_useful;
            }
            (
                QueueKey {
                    policy_slot,
                    sequence_bucket,
                    action_bucket,
                },
                selected,
            )
        } else {
            let (&key, indices) = full
                .into_iter()
                .min_by_key(|(key, indices)| {
                    let oldest = self.pending[indices[0]].enqueue_sequence;
                    (oldest, **key)
                })
                .expect("a full queue exists");
            let row_limit = (self.token_budget / key.sequence_bucket).max(1);
            (
                key,
                indices
                    .iter()
                    .copied()
                    .take(row_limit)
                    .collect::<BTreeSet<_>>(),
            )
        };
        let mut rows = self
            .pending
            .iter()
            .enumerate()
            .filter(|(index, _)| selected_indices.contains(index))
            .map(|(_, row)| row.clone())
            .collect::<Vec<_>>();
        rows.sort_by_key(|row| (row.environment_id, row.frame_id, row.seat));
        let request_id = self.next_request_id;
        self.next_request_id = self.next_request_id.wrapping_add(1);
        let request = build_request(request_id, key, &rows);
        self.metrics.inference_requests += 1;
        self.metrics.inference_rows += rows.len() as u64;
        self.metrics.useful_tokens += rows.iter().map(|row| row.tokens.len() as u64).sum::<u64>();
        self.metrics.padded_tokens += (rows.len() * key.sequence_bucket) as u64;
        self.metrics.useful_actions += rows.iter().map(|row| row.actions.len() as u64).sum::<u64>();
        self.metrics.padded_actions += (rows.len() * key.action_bucket) as u64;
        self.active_request = Some((request_id, rows));
        Ok(Some(request))
    }

    pub fn submit(
        &mut self,
        request_id: u64,
        selected_groups: &[usize],
        old_logp: &[f32],
        old_state_values: &[f32],
    ) -> Result<(), EnvError> {
        let (active_id, rows) = self.active_request.take().ok_or_else(|| {
            EnvError::InvalidArgument("there is no active inference request".into())
        })?;
        if request_id != active_id {
            self.active_request = Some((active_id, rows));
            return Err(EnvError::InvalidArgument(format!(
                "stale inference request {request_id}; active request is {active_id}"
            )));
        }
        if selected_groups.len() != rows.len()
            || old_logp.len() != rows.len()
            || old_state_values.len() != rows.len()
        {
            self.active_request = Some((active_id, rows));
            return Err(EnvError::InvalidArgument(
                "submission columns must match the request row count".into(),
            ));
        }
        for (index, (((row, &selected), &logp), &state_value)) in rows
            .iter()
            .zip(selected_groups)
            .zip(old_logp)
            .zip(old_state_values)
            .enumerate()
        {
            if selected >= row.native_actions.len() {
                self.active_request = Some((active_id, rows));
                return Err(EnvError::InvalidArgument(format!(
                    "selected group {selected} is invalid for request row {index}"
                )));
            }
            if !logp.is_finite() || logp > 1e-5 {
                self.active_request = Some((active_id, rows));
                return Err(EnvError::InvalidArgument(format!(
                    "old_logp for request row {index} must be finite and non-positive"
                )));
            }
            if !state_value.is_finite() {
                self.active_request = Some((active_id, rows));
                return Err(EnvError::InvalidArgument(format!(
                    "old state value for request row {index} must be finite"
                )));
            }
        }
        for (((row, &selected), &logp), &state_value) in rows
            .iter()
            .zip(selected_groups)
            .zip(old_logp)
            .zip(old_state_values)
        {
            if !self.inference_only {
                self.chunk.push(row, selected, logp, state_value);
            }
            self.selections.insert(
                row.decision_key(),
                ActionSelection {
                    environment_id: row.environment_id,
                    episode_generation: row.episode_generation,
                    frame_id: row.frame_id,
                    seat: row.seat,
                    candidate_index: row.native_representatives[selected],
                },
            );
        }
        let submitted = rows.iter().map(|row| row.row_id).collect::<BTreeSet<_>>();
        self.pending.retain(|row| !submitted.contains(&row.row_id));
        self.advance_ready_environments()?;
        self.drive_native_bots()?;
        Ok(())
    }

    pub fn is_complete(&self) -> bool {
        !self.active_matches.is_empty()
            && self.completed_matches.len() == self.active_matches.len()
            && self.active_request.is_none()
            && self.pending.is_empty()
    }

    pub fn chunk(&self) -> &RolloutChunk {
        &self.chunk
    }

    pub fn take_chunk(&mut self) -> Result<RolloutChunk, EnvError> {
        if !self.is_complete() {
            return Err(EnvError::InvalidArgument(
                "cannot take an incomplete rollout chunk".into(),
            ));
        }
        Ok(std::mem::replace(&mut self.chunk, RolloutChunk::new()))
    }

    pub fn metrics(&self) -> &RolloutMetrics {
        &self.metrics
    }

    pub fn snapshots(&self) -> Result<BTreeMap<u32, Vec<u8>>, EnvError> {
        self.env
            .snapshots(&(0..self.num_envs() as u32).collect::<Vec<_>>())
    }

    pub fn restore_idle(&mut self, snapshots: &BTreeMap<u32, Vec<u8>>) -> Result<(), EnvError> {
        if self.active_request.is_some() || !self.pending.is_empty() {
            return Err(EnvError::InvalidArgument(
                "native rollout snapshots can only be restored between chunks".into(),
            ));
        }
        self.active_matches.clear();
        self.completed_matches.clear();
        self.lineups.clear();
        self.selections.clear();
        let transition = self.env.restore(snapshots)?;
        self.ingest_transition(transition)
    }

    fn ingest_transition(&mut self, transition: BatchTransition) -> Result<(), EnvError> {
        self.observe_events(&transition.events)?;
        for state in transition.states {
            let id = state.environment_id as usize;
            if state.lifecycle == EnvironmentLifecycle::Failed {
                return Err(EnvError::InvalidArgument(format!(
                    "environment {} failed during rollout",
                    state.environment_id
                )));
            }
            if state.lifecycle == EnvironmentLifecycle::Complete {
                self.completed_matches
                    .insert((state.environment_id, state.episode_generation));
            }
            self.latest_states[id] = Some(state);
        }
        if !self.lineups.is_empty() {
            self.rebuild_pending()?;
        }
        Ok(())
    }

    fn observe_events(&mut self, events: &[EventRecord]) -> Result<(), EnvError> {
        for event in events {
            let id = event.environment_id as usize;
            if id >= self.histories.len() {
                return Err(EnvError::EnvironmentOutOfRange(event.environment_id));
            }
            if self.history_generation[id] != event.episode_generation {
                self.history_generation[id] = event.episode_generation;
                self.histories[id] = std::array::from_fn(|_| Vec::new());
                self.boundaries[id] = None;
            }
            self.observe_gameplay(event)?;
            if event.kind == EventKind::StartKyoku {
                self.histories[id] = std::array::from_fn(|_| Vec::new());
                if let Some(boundary) = boundary_from_start_event(event) {
                    self.boundaries[id] = Some(boundary);
                }
            }
            for observer in 0..4_u8 {
                if let Some(row) = encode_event(event, observer) {
                    self.histories[id][usize::from(observer)].push(row);
                }
            }
            if event.kind == EventKind::EndKyoku {
                self.mark_kyoku_boundary(event.environment_id, event.episode_generation);
                self.chunk.kyoku_completions += 1;
            } else if event.kind == EventKind::EndGame {
                self.observe_terminal(event)?;
            }
        }
        Ok(())
    }

    fn observe_gameplay(&mut self, event: &EventRecord) -> Result<(), EnvError> {
        let id = event.environment_id as usize;
        match event.kind {
            EventKind::StartGame => {
                self.kyoku_gameplay[id] = None;
                self.match_gameplay[id] = MatchGameplay::default();
            }
            EventKind::StartKyoku => {
                self.kyoku_gameplay[id] = Some(KyokuGameplay::default());
            }
            EventKind::Dahai if event.actor_seat < 4 => {
                if let Some(kyoku) = self.kyoku_gameplay[id].as_mut() {
                    let seat = usize::from(event.actor_seat);
                    kyoku.discards[seat] = kyoku.discards[seat].saturating_add(1);
                }
            }
            EventKind::Chi | EventKind::Pon | EventKind::Daiminkan if event.actor_seat < 4 => {
                if let Some(kyoku) = self.kyoku_gameplay[id].as_mut() {
                    kyoku.open_mask |= 1 << event.actor_seat;
                }
            }
            EventKind::Reach if event.actor_seat < 4 => {
                if let Some(kyoku) = self.kyoku_gameplay[id].as_mut() {
                    kyoku.riichi_mask |= 1 << event.actor_seat;
                }
            }
            EventKind::Hora if event.actor_seat < 4 => {
                let actor = usize::from(event.actor_seat);
                let target = usize::from(event.target_seat);
                let settlement = (event.payload.len() >= 16).then(|| {
                    std::array::from_fn::<_, 4, _>(|seat| {
                        let start = event.payload.len() - 16 + seat * 4;
                        i32::from_le_bytes(
                            event.payload[start..start + 4]
                                .try_into()
                                .expect("four settlement bytes"),
                        )
                    })
                });
                let Some(kyoku) = self.kyoku_gameplay[id].as_mut() else {
                    return Ok(());
                };
                let game = &mut self.match_gameplay[id];
                game.wins[actor] += 1;
                let is_tsumo = actor == target;
                game.tsumo_wins[actor] += u32::from(is_tsumo);
                let actor_bit = 1 << event.actor_seat;
                game.dama_wins[actor] += u32::from(
                    kyoku.open_mask & actor_bit == 0 && kyoku.riichi_mask & actor_bit == 0,
                );
                game.winning_turns[actor] += if is_tsumo {
                    u64::from(kyoku.discards[actor]) + 1
                } else {
                    u64::from(kyoku.discards[actor].max(1))
                };
                game.winning_turn_events[actor] += 1;
                if let Some(values) = settlement {
                    game.winning_points[actor] += i64::from(values[actor].max(0));
                    game.winning_point_events[actor] += 1;
                    if actor != target && target < 4 {
                        let target_bit = 1 << event.target_seat;
                        if kyoku.dealt_in_mask & target_bit == 0 {
                            kyoku.dealt_in_mask |= target_bit;
                            game.deal_ins[target] += 1;
                            game.deal_in_points[target] += i64::from((-values[target]).max(0));
                            game.deal_in_point_events[target] += 1;
                        }
                    }
                } else if actor != target && target < 4 {
                    let target_bit = 1 << event.target_seat;
                    if kyoku.dealt_in_mask & target_bit == 0 {
                        kyoku.dealt_in_mask |= target_bit;
                        game.deal_ins[target] += 1;
                    }
                }
            }
            EventKind::Ryukyoku if event.args[0] == 3 => {
                self.match_gameplay[id].exhaustive_ryukyoku += 1;
            }
            EventKind::EndKyoku => {
                if let Some(kyoku) = self.kyoku_gameplay[id].take() {
                    let game = &mut self.match_gameplay[id];
                    game.kyoku += 1;
                    for seat in 0..4 {
                        game.riichi_hands[seat] += u32::from(kyoku.riichi_mask & (1 << seat) != 0);
                        game.calling_hands[seat] += u32::from(kyoku.open_mask & (1 << seat) != 0);
                    }
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn observe_terminal(&mut self, event: &EventRecord) -> Result<(), EnvError> {
        if event.payload.len() != 20 {
            return Err(EnvError::InvalidArgument(
                "end_game event must contain four scores and four ranks".into(),
            ));
        }
        self.chunk
            .terminal_environment_ids
            .push(event.environment_id);
        self.chunk
            .terminal_episode_generations
            .push(event.episode_generation);
        for seat in 0..4 {
            self.chunk.terminal_scores.push(i32::from_le_bytes(
                event.payload[4 * seat..4 * seat + 4]
                    .try_into()
                    .expect("four bytes"),
            ));
        }
        self.chunk.terminal_ranks.extend(
            event.payload[16..20]
                .iter()
                .map(|rank| rank.saturating_sub(1)),
        );
        self.chunk
            .terminal_completed_kyoku
            .push(event.args[0].clamp(0, i64::from(u32::MAX)) as u32);
        let gameplay = &self.match_gameplay[event.environment_id as usize];
        self.chunk
            .terminal_exhaustive_ryukyoku
            .push(gameplay.exhaustive_ryukyoku);
        self.chunk.terminal_wins.extend(gameplay.wins);
        self.chunk.terminal_deal_ins.extend(gameplay.deal_ins);
        self.chunk
            .terminal_riichi_hands
            .extend(gameplay.riichi_hands);
        self.chunk
            .terminal_calling_hands
            .extend(gameplay.calling_hands);
        self.chunk.terminal_tsumo_wins.extend(gameplay.tsumo_wins);
        self.chunk.terminal_dama_wins.extend(gameplay.dama_wins);
        self.chunk
            .terminal_winning_points
            .extend(gameplay.winning_points);
        self.chunk
            .terminal_winning_point_events
            .extend(gameplay.winning_point_events);
        self.chunk
            .terminal_deal_in_points
            .extend(gameplay.deal_in_points);
        self.chunk
            .terminal_deal_in_point_events
            .extend(gameplay.deal_in_point_events);
        self.chunk
            .terminal_winning_turns
            .extend(gameplay.winning_turns);
        self.chunk
            .terminal_winning_turn_events
            .extend(gameplay.winning_turn_events);
        let ranks =
            std::array::from_fn::<_, 4, _>(|seat| event.payload[16 + seat].saturating_sub(1));
        let absolute_order = {
            let mut seats = [0_u8, 1, 2, 3];
            seats.sort_by_key(|&seat| ranks[usize::from(seat)]);
            seats
        };
        let match_id = (event.environment_id, event.episode_generation);
        let match_rows = self
            .chunk
            .match_rows
            .get(&match_id)
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        for &index in match_rows {
            let seat = usize::from(self.chunk.seats[index]);
            self.chunk.terminal_placements[index] = ranks[seat] as i8;
            let dealer = (0..4)
                .max_by(|&left, &right| {
                    self.chunk.rank_boundary_features[index * 28 + 4 + left]
                        .total_cmp(&self.chunk.rank_boundary_features[index * 28 + 4 + right])
                })
                .expect("four dealers") as u8;
            let relative_order = absolute_order.map(|seat| (seat + 4 - dealer) % 4);
            self.chunk.rank_order_targets[index] = permutation_index(relative_order) as i8;
        }
        for seat in 0..4_u8 {
            if let Some(index) = match_rows
                .iter()
                .rev()
                .copied()
                .find(|&index| self.chunk.seats[index] == seat)
            {
                self.chunk.match_boundary[index] = 1;
            }
        }
        self.chunk.match_completions += 1;
        Ok(())
    }

    fn rebuild_pending(&mut self) -> Result<(), EnvError> {
        let existing = self
            .pending
            .iter()
            .map(PendingRow::decision_key)
            .chain(self.selections.keys().copied())
            .collect::<BTreeSet<_>>();
        for state in self.latest_states.iter().flatten() {
            if state.lifecycle == EnvironmentLifecycle::Complete {
                continue;
            }
            let Some(game) = state.hanchan.as_ref() else {
                continue;
            };
            let Some(frame) = game.hand.decision.as_ref() else {
                continue;
            };
            let match_id = (state.environment_id, state.episode_generation);
            if !self.active_matches.contains(&match_id) {
                continue;
            }
            let lineup = self.lineups.get(&match_id).ok_or_else(|| {
                EnvError::InvalidArgument(format!("missing lineup for match {match_id:?}"))
            })?;
            let boundary = self.boundaries[state.environment_id as usize]
                .unwrap_or_else(|| RankBoundary::from_hanchan(game));
            for space in &frame.action_spaces {
                let key = (
                    state.environment_id,
                    state.episode_generation,
                    frame.frame_id,
                    space.seat,
                );
                if existing.contains(&key) {
                    continue;
                }
                let (tokens, actor_query_offset, rank_boundary_features, decision_seat) =
                    encode_observation(
                        state,
                        space.seat,
                        &self.histories[state.environment_id as usize][usize::from(space.seat)],
                        boundary,
                    )
                    .expect("decision observer has an encodable observation");
                if tokens.len() > self.context_tokens {
                    return Err(EnvError::InvalidArgument(format!(
                        "encoded sequence length {} exceeds context_tokens {}",
                        tokens.len(),
                        self.context_tokens
                    )));
                }
                if space.candidates.len() > 256 {
                    return Err(EnvError::InvalidArgument(
                        "native action count exceeds 256".into(),
                    ));
                }
                let policy_slot = lineup.policy_slots[usize::from(space.seat)];
                let boundary_key = (match_id, boundary, decision_seat, policy_slot);
                let boundary_group_id =
                    *self.boundary_groups.entry(boundary_key).or_insert_with(|| {
                        let value = self.next_boundary_group_id;
                        self.next_boundary_group_id = self.next_boundary_group_id.wrapping_add(1);
                        value
                    });
                let mut group_by_factors = BTreeMap::<[u8; 15], usize>::new();
                let mut actions = Vec::new();
                let mut native_actions = Vec::new();
                let mut native_representatives = Vec::new();
                for (native_index, candidate) in space.candidates.iter().enumerate() {
                    let factors = encode_action(candidate, space.seat);
                    if group_by_factors.contains_key(&factors) {
                        continue;
                    }
                    group_by_factors.insert(factors, actions.len());
                    actions.push(factors);
                    native_actions.push(candidate.clone());
                    native_representatives.push(native_index as u32);
                }
                self.pending.push(PendingRow {
                    row_id: self.next_row_id,
                    enqueue_sequence: self.next_enqueue_sequence,
                    environment_id: state.environment_id,
                    episode_generation: state.episode_generation,
                    frame_id: frame.frame_id,
                    seat: space.seat,
                    policy_slot,
                    eligible: lineup.learner_mask & (1 << space.seat) != 0,
                    phase: frame.phase as u8,
                    tokens,
                    actor_query_offset,
                    decision_seat,
                    rank_boundary_features,
                    boundary_group_id,
                    actions,
                    native_actions,
                    native_representatives,
                });
                self.next_row_id = self.next_row_id.wrapping_add(1);
                self.next_enqueue_sequence = self.next_enqueue_sequence.wrapping_add(1);
            }
        }
        self.pending
            .sort_by_key(|row| (row.environment_id, row.frame_id, row.seat));
        Ok(())
    }

    fn drive_native_bots(&mut self) -> Result<(), EnvError> {
        loop {
            let automatic = self
                .pending
                .iter()
                .filter(|row| row.native_actions.len() == 1)
                .cloned()
                .collect::<Vec<_>>();
            if !automatic.is_empty() {
                let ids = automatic
                    .iter()
                    .map(|row| row.row_id)
                    .collect::<BTreeSet<_>>();
                for row in &automatic {
                    self.selections.insert(
                        row.decision_key(),
                        ActionSelection {
                            environment_id: row.environment_id,
                            episode_generation: row.episode_generation,
                            frame_id: row.frame_id,
                            seat: row.seat,
                            candidate_index: row.native_representatives[0],
                        },
                    );
                }
                self.pending.retain(|row| !ids.contains(&row.row_id));
                self.metrics.automatic_rows += automatic.len() as u64;
                self.advance_ready_environments()?;
                continue;
            }
            let bot_ids = self
                .pending
                .iter()
                .filter(|row| self.bot_policy_slots.contains(&row.policy_slot))
                .map(|row| row.row_id)
                .collect::<BTreeSet<_>>();
            if bot_ids.is_empty() {
                return Ok(());
            }
            let bot_rows = self
                .pending
                .iter()
                .filter(|row| bot_ids.contains(&row.row_id))
                .cloned()
                .collect::<Vec<_>>();
            for row in &bot_rows {
                if row.eligible {
                    return Err(EnvError::InvalidArgument(
                        "native deterministic bot rows cannot be learner-eligible".into(),
                    ));
                }
                let state = self.latest_states[row.environment_id as usize]
                    .as_ref()
                    .expect("pending row has state");
                let selected = conservative_action(state, row.seat, &row.native_actions);
                self.selections.insert(
                    row.decision_key(),
                    ActionSelection {
                        environment_id: row.environment_id,
                        episode_generation: row.episode_generation,
                        frame_id: row.frame_id,
                        seat: row.seat,
                        candidate_index: row.native_representatives[selected],
                    },
                );
                self.metrics.native_bot_rows += 1;
            }
            self.pending.retain(|row| !bot_ids.contains(&row.row_id));
            let before = self.selections.len();
            self.advance_ready_environments()?;
            if self.selections.len() == before {
                return Ok(());
            }
        }
    }

    fn advance_ready_environments(&mut self) -> Result<(), EnvError> {
        let mut selections = Vec::new();
        for state in self.latest_states.iter().flatten() {
            let Some(frame) = state
                .hanchan
                .as_ref()
                .and_then(|game| game.hand.decision.as_ref())
            else {
                continue;
            };
            let keys = frame
                .action_spaces
                .iter()
                .map(|space| {
                    (
                        state.environment_id,
                        state.episode_generation,
                        frame.frame_id,
                        space.seat,
                    )
                })
                .collect::<Vec<_>>();
            if keys.iter().all(|key| self.selections.contains_key(key)) {
                for key in keys {
                    selections.push(
                        self.selections
                            .remove(&key)
                            .expect("complete decision selection"),
                    );
                }
            }
        }
        if selections.is_empty() {
            return Ok(());
        }
        let transition = self.env.step(&selections)?;
        self.metrics.env_calls += 1;
        self.ingest_transition(transition)?;
        self.advance_automatic_states()
    }

    fn advance_automatic_states(&mut self) -> Result<(), EnvError> {
        loop {
            let ids = self
                .latest_states
                .iter()
                .flatten()
                .filter(|state| {
                    self.active_matches
                        .contains(&(state.environment_id, state.episode_generation))
                        && state.lifecycle != EnvironmentLifecycle::Complete
                        && state
                            .hanchan
                            .as_ref()
                            .and_then(|game| game.hand.decision.as_ref())
                            .is_none()
                })
                .map(|state| state.environment_id)
                .collect::<Vec<_>>();
            if ids.is_empty() {
                return Ok(());
            }
            let transition = self.env.advance(&ids)?;
            self.metrics.env_calls += 1;
            self.ingest_transition(transition)?;
        }
    }

    fn mark_kyoku_boundary(&mut self, environment_id: u32, generation: u64) {
        let Some(boundary) = self.boundaries[environment_id as usize] else {
            return;
        };
        let features = boundary.features();
        let match_rows = self
            .chunk
            .match_rows
            .get(&(environment_id, generation))
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        let groups = match_rows
            .iter()
            .rev()
            .copied()
            .filter(|&index| {
                self.chunk.rank_boundary_features[index * 28..index * 28 + 28] == features
            })
            .map(|index| self.chunk.boundary_group_ids[index])
            .collect::<BTreeSet<_>>();
        for seat in 0..4_u8 {
            if let Some(index) = match_rows.iter().rev().copied().find(|&index| {
                self.chunk.seats[index] == seat
                    && groups.contains(&self.chunk.boundary_group_ids[index])
            }) {
                self.chunk.kyoku_boundary[index] = 1;
            }
        }
        let mut policies = BTreeSet::new();
        for &index in match_rows {
            if groups.contains(&self.chunk.boundary_group_ids[index])
                && self.chunk.eligibility[index] != 0
                && policies.insert(self.chunk.policy_slots[index])
            {
                self.chunk.rank_boundary_supervision[index] = 1;
            }
        }
    }
}

fn build_request(request_id: u64, key: QueueKey, rows: &[PendingRow]) -> InferenceRequest {
    let mut request = InferenceRequest {
        request_id,
        policy_slot: key.policy_slot,
        sequence_bucket: key.sequence_bucket,
        action_bucket: key.action_bucket,
        row_ids: Vec::with_capacity(rows.len()),
        environment_ids: Vec::with_capacity(rows.len()),
        episode_generations: Vec::with_capacity(rows.len()),
        frame_ids: Vec::with_capacity(rows.len()),
        seats: Vec::with_capacity(rows.len()),
        token_factors: vec![0; rows.len() * key.sequence_bucket * 10],
        token_numeric: vec![0.0; rows.len() * key.sequence_bucket * 8],
        lengths: Vec::with_capacity(rows.len()),
        query_offsets: Vec::with_capacity(rows.len()),
        decision_seats: Vec::with_capacity(rows.len()),
        rank_boundary_features: Vec::with_capacity(rows.len() * 28),
        action_factors: vec![0; rows.len() * key.action_bucket * 15],
        action_lengths: Vec::with_capacity(rows.len()),
        action_offsets: vec![0],
    };
    for (batch, row) in rows.iter().enumerate() {
        request.row_ids.push(row.row_id);
        request.environment_ids.push(row.environment_id);
        request.episode_generations.push(row.episode_generation);
        request.frame_ids.push(row.frame_id);
        request.seats.push(row.seat);
        request.lengths.push(row.tokens.len() as i64);
        request.query_offsets.push(row.actor_query_offset as i64);
        request.decision_seats.push(i64::from(row.decision_seat));
        request
            .rank_boundary_features
            .extend_from_slice(&row.rank_boundary_features);
        for (token, values) in row.tokens.iter().enumerate() {
            let factor_start = (batch * key.sequence_bucket + token) * 10;
            request.token_factors[factor_start..factor_start + 10]
                .iter_mut()
                .zip(values.categorical)
                .for_each(|(target, value)| *target = i32::from(value));
            let numeric_start = (batch * key.sequence_bucket + token) * 8;
            request.token_numeric[numeric_start..numeric_start + 8]
                .copy_from_slice(&values.numeric);
        }
        for (action, values) in row.actions.iter().enumerate() {
            let start = (batch * key.action_bucket + action) * 15;
            request.action_factors[start..start + 15]
                .iter_mut()
                .zip(values)
                .for_each(|(target, value)| *target = i32::from(*value));
        }
        request.action_lengths.push(row.actions.len() as i64);
        request
            .action_offsets
            .push(request.action_offsets.last().copied().unwrap_or(0) + row.actions.len() as i64);
    }
    request
}

fn sequence_bucket(length: usize, context_tokens: usize) -> usize {
    let mut bucket = 64_usize;
    while bucket < length && bucket < context_tokens {
        let grown = ((bucket as f64) * 1.2).ceil() as usize;
        bucket = grown.div_ceil(32) * 32;
    }
    bucket.min(context_tokens).max(length)
}

fn action_bucket(length: usize) -> usize {
    length.max(2).next_power_of_two().min(256)
}

fn permutation_index(order: [u8; 4]) -> usize {
    let mut result = 0;
    for index in 0..4 {
        let smaller = order[index + 1..]
            .iter()
            .filter(|&&value| value < order[index])
            .count();
        result += smaller * [6, 2, 1, 1][index];
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn engine() -> RolloutEngine {
        RolloutEngine::new(4, 7, 1, 2048, 4096, false).unwrap()
    }

    #[test]
    fn scheduler_orders_rows_and_rejects_stale_submissions() {
        let mut engine = engine();
        let matches = engine.reset_chunk(4).unwrap();
        engine
            .register_lineups(&matches, &[[0; 4]; 4], &[0b1111; 4], &[])
            .unwrap();
        let request = engine.next_request().unwrap().unwrap();
        assert!(request
            .environment_ids
            .windows(2)
            .all(|pair| pair[0] <= pair[1]));
        assert!(engine
            .submit(
                request.request_id + 1,
                &vec![0; request.rows()],
                &vec![0.0; request.rows()],
                &vec![0.0; request.rows()],
            )
            .is_err());
        engine
            .submit(
                request.request_id,
                &vec![0; request.rows()],
                &vec![0.0; request.rows()],
                &vec![0.0; request.rows()],
            )
            .unwrap();
    }

    #[test]
    fn invalid_group_keeps_request_active_for_retry() {
        let mut engine = engine();
        let matches = engine.reset_chunk(1).unwrap();
        engine
            .register_lineups(&matches, &[[0; 4]], &[0b1111], &[])
            .unwrap();
        let request = engine.next_request().unwrap().unwrap();
        let mut selected = vec![0; request.rows()];
        selected[0] = usize::MAX;
        assert!(engine
            .submit(
                request.request_id,
                &selected,
                &vec![0.0; request.rows()],
                &vec![0.0; request.rows()],
            )
            .is_err());
        selected[0] = 0;
        engine
            .submit(
                request.request_id,
                &selected,
                &vec![0.0; request.rows()],
                &vec![0.0; request.rows()],
            )
            .unwrap();
    }

    #[test]
    fn native_bot_can_finish_a_chunk_without_python_requests() {
        let mut engine = engine();
        let matches = engine.reset_chunk(1).unwrap();
        engine
            .register_lineups(&matches, &[[9; 4]], &[0], &[9])
            .unwrap();
        assert!(engine.is_complete());
        assert!(engine.next_request().unwrap().is_none());
        assert_eq!(engine.chunk().match_completions, 1);
        assert!(engine.metrics().native_bot_rows > 0);
    }

    #[test]
    fn buckets_follow_contract() {
        assert_eq!(sequence_bucket(1, 2048), 64);
        assert_eq!(sequence_bucket(65, 2048), 96);
        assert_eq!(sequence_bucket(97, 2048), 128);
        assert_eq!(action_bucket(1), 2);
        assert_eq!(action_bucket(5), 8);
        assert_eq!(action_bucket(256), 256);
    }

    #[test]
    fn gameplay_events_use_per_player_and_conditional_counters() {
        fn event(
            kind: EventKind,
            actor: u8,
            target: u8,
            args: [i64; 4],
            payload: Vec<u8>,
        ) -> EventRecord {
            EventRecord {
                environment_id: 0,
                episode_generation: 1,
                sequence: 0,
                kind,
                actor_seat: actor,
                target_seat: target,
                visibility_mask: 15,
                args,
                payload,
            }
        }
        fn settlement(values: [i32; 4]) -> Vec<u8> {
            values.into_iter().flat_map(i32::to_le_bytes).collect()
        }
        let mut engine = engine();
        let events = [
            event(EventKind::StartKyoku, 255, 255, [0; 4], vec![]),
            event(EventKind::Reach, 0, 255, [0; 4], vec![]),
            event(EventKind::Chi, 1, 255, [0; 4], vec![]),
            event(EventKind::Dahai, 2, 255, [0; 4], vec![]),
            event(EventKind::Dahai, 2, 255, [0; 4], vec![]),
            event(EventKind::Reach, 2, 255, [0; 4], vec![]),
            event(
                EventKind::Hora,
                2,
                2,
                [0; 4],
                settlement([-1000, -2000, 5000, -2000]),
            ),
            event(EventKind::EndKyoku, 255, 255, [0; 4], vec![]),
            event(EventKind::StartKyoku, 255, 255, [0; 4], vec![]),
            event(
                EventKind::Hora,
                0,
                3,
                [0; 4],
                settlement([8000, 4000, 0, -12000]),
            ),
            event(
                EventKind::Hora,
                1,
                3,
                [0; 4],
                settlement([8000, 4000, 0, -12000]),
            ),
            event(EventKind::EndKyoku, 255, 255, [0; 4], vec![]),
            event(EventKind::StartKyoku, 255, 255, [0; 4], vec![]),
            event(EventKind::Reach, 0, 255, [0; 4], vec![]),
            event(EventKind::Pon, 1, 255, [0; 4], vec![]),
            event(
                EventKind::Ryukyoku,
                255,
                255,
                [3, 15, 0, 0],
                settlement([0; 4]),
            ),
            event(EventKind::EndKyoku, 255, 255, [0; 4], vec![]),
        ];
        for value in &events {
            engine.observe_gameplay(value).unwrap();
        }
        let game = &engine.match_gameplay[0];
        assert_eq!(game.kyoku, 3);
        assert_eq!(game.exhaustive_ryukyoku, 1);
        assert_eq!(game.wins, [1, 1, 1, 0]);
        assert_eq!(game.deal_ins, [0, 0, 0, 1]);
        assert_eq!(game.riichi_hands, [2, 0, 1, 0]);
        assert_eq!(game.calling_hands, [0, 2, 0, 0]);
        assert_eq!(game.tsumo_wins, [0, 0, 1, 0]);
        assert_eq!(game.dama_wins, [1, 1, 0, 0]);
        assert_eq!(game.winning_points.iter().sum::<i64>(), 17_000);
        assert_eq!(game.deal_in_points[3], 12_000);
        assert_eq!(game.winning_turns.iter().sum::<u64>(), 5);
    }
}
