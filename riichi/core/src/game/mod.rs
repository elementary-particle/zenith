#[path = "../action.rs"]
pub mod action;
pub mod event;
pub mod phase;
pub mod replay;
pub mod rules;
pub mod search;
pub mod state;
pub mod transition;

struct EventFields {
    actor_seat: u8,
    target_seat: u8,
    visibility_mask: u8,
    args: [i64; 4],
    payload: Vec<u8>,
}

use action::{ActionCandidate, ActionKind, ActionSelection, ABSENT};
use event::{event_kind_for_action, start_kyoku_payload, wind_code, EventKind, EventRecord};
use phase::{EnvironmentLifecycle, HandPhase, TerminalReason, Wind};
use rules::hand::{deal, initialize_wall};
use rules::{
    hand::tile_type_counts,
    scoring::evaluate_hand,
    settlement::{exhaustive_draw, multiple_ron, ranking, tsumo as settle_tsumo, RonClaim},
    shanten,
};
use state::{GameState, HanchanState, HandState, PlayerState, RngState};

impl GameState {
    pub fn new(environment_id: u32) -> Self {
        Self::uninitialized_with_rules_profile(environment_id, rules::profile::RULES_PROFILE_ID)
    }

    pub fn new_with_rules_profile(environment_id: u32, rules_profile_id: u32) -> Self {
        assert!(
            rules::profile::by_id(rules_profile_id).is_some(),
            "unsupported rules profile"
        );
        Self::uninitialized_with_rules_profile(environment_id, rules_profile_id)
    }

    pub fn rules_profile(&self) -> &'static rules::profile::RulesProfile {
        rules::profile::by_id(self.rules_profile_id).expect("validated rules profile")
    }

    pub fn reset_from_seed(&mut self, master_seed: u64) {
        self.reset_from_seed_and_count(master_seed);
    }

    pub fn reset_from_seed_and_count(&mut self, master_seed: u64) -> u64 {
        let generation = self.episode_generation.wrapping_add(1);
        self.reset_and_count(state::derive_rng(
            master_seed,
            self.environment_id,
            generation,
        ))
    }

    pub fn uninitialized(environment_id: u32) -> Self {
        Self::uninitialized_with_rules_profile(environment_id, rules::profile::RULES_PROFILE_ID)
    }

    fn uninitialized_with_rules_profile(environment_id: u32, rules_profile_id: u32) -> Self {
        Self {
            environment_id,
            rules_profile_id,
            episode_generation: 0,
            lifecycle: EnvironmentLifecycle::Uninitialized,
            rng: RngState {
                state: 0,
                stream: 1,
            },
            hanchan: None,
            next_frame_id: 1,
            next_event_sequence: 0,
            failure: None,
            externally_loaded: false,
            pending_dora_reveal: false,
            pending_events: Vec::new(),
            automatic_decisions: 0,
        }
    }

    pub fn reset(&mut self, rng: RngState) {
        self.reset_and_count(rng);
    }

    pub fn reset_and_count(&mut self, rng: RngState) -> u64 {
        self.episode_generation = self.episode_generation.wrapping_add(1);
        self.lifecycle = EnvironmentLifecycle::Ready;
        self.rng = rng;
        self.failure = None;
        self.externally_loaded = false;
        self.pending_dora_reveal = false;
        self.next_frame_id = 1;
        self.next_event_sequence = 0;
        self.pending_events.clear();
        self.automatic_decisions = 0;
        let (players, hand) = fresh_hand(&mut self.rng, 0);
        self.hanchan = Some(HanchanState {
            round_wind: Wind::East,
            hand_number: 0,
            dealer: 0,
            honba: 0,
            riichi_deposits: 0,
            completed_kyoku: 0,
            scores: [self.rules_profile().starting_points; 4],
            initial_seats: [0, 1, 2, 3],
            players,
            hand,
        });
        self.emit_start_game();
        self.start_current_hand();
        self.automatic_decisions
    }

    pub fn legal_selections(&self) -> Vec<ActionSelection> {
        self.hanchan
            .as_ref()
            .and_then(|game| game.hand.decision.as_ref())
            .into_iter()
            .flat_map(|decision| {
                decision.action_spaces.iter().flat_map(move |space| {
                    space.candidates.iter().enumerate().map(move |(index, _)| {
                        ActionSelection::bind(decision, space.seat, index as u32)
                    })
                })
            })
            .collect()
    }

    /// Validates the complete simultaneous action set before changing state.
    pub fn step(&mut self, selections: &[ActionSelection]) -> Result<(), crate::error::CoreError> {
        self.step_and_count(selections).map(|_| ())
    }

    /// Resolve one submitted choice frame. Returns the number of no-choice
    /// seats collapsed while constructing the resulting frame.
    pub fn step_and_count(
        &mut self,
        selections: &[ActionSelection],
    ) -> Result<u64, crate::error::CoreError> {
        let before = self.automatic_decisions;
        let selected = self.validate_selections(selections)?;
        self.apply_actions(selected);
        Ok(self.automatic_decisions.saturating_sub(before))
    }

    /// Checks a complete frame-local action set without mutating the game.
    pub fn validate_selections(
        &self,
        selections: &[ActionSelection],
    ) -> Result<Vec<(u8, ActionCandidate)>, crate::error::CoreError> {
        use std::collections::HashSet;

        let Some(frame) = self
            .hanchan
            .as_ref()
            .and_then(|game| game.hand.decision.as_ref())
        else {
            return Err(invalid_actions(
                crate::error::FrameStatus::StaleFrame,
                crate::error::ErrorCode::EnvironmentNotReady,
                0,
                0,
            ));
        };
        let mut seen = HashSet::new();
        let mut selected = Vec::with_capacity(selections.len());
        for selection in selections {
            if selection.environment_id != self.environment_id {
                return Err(invalid_actions(
                    crate::error::FrameStatus::InvalidAction,
                    crate::error::ErrorCode::EnvironmentOutOfRange,
                    i64::from(selection.environment_id),
                    i64::from(self.environment_id),
                ));
            }
            if selection.episode_generation != frame.episode_generation {
                return Err(invalid_actions(
                    crate::error::FrameStatus::StaleFrame,
                    crate::error::ErrorCode::EpisodeGenerationMismatch,
                    selection.episode_generation as i64,
                    frame.episode_generation as i64,
                ));
            }
            if selection.frame_id != frame.frame_id {
                return Err(invalid_actions(
                    crate::error::FrameStatus::StaleFrame,
                    crate::error::ErrorCode::FrameIdMismatch,
                    selection.frame_id as i64,
                    frame.frame_id as i64,
                ));
            }
            if !seen.insert(selection.seat) {
                return Err(invalid_actions(
                    crate::error::FrameStatus::DuplicateSeat,
                    crate::error::ErrorCode::DuplicateSeat,
                    i64::from(selection.seat),
                    0,
                ));
            }
            let Some(space) = frame
                .action_spaces
                .iter()
                .find(|value| value.seat == selection.seat)
            else {
                return Err(invalid_actions(
                    crate::error::FrameStatus::IneligibleSeat,
                    crate::error::ErrorCode::IneligibleSeat,
                    i64::from(selection.seat),
                    0,
                ));
            };
            let Some(value) = space.candidates.get(selection.candidate_index as usize) else {
                return Err(invalid_actions(
                    crate::error::FrameStatus::InvalidAction,
                    crate::error::ErrorCode::ActionOutOfRange,
                    i64::from(selection.candidate_index),
                    space.candidates.len() as i64,
                ));
            };
            selected.push((selection.seat, value.clone()));
        }
        if selected.len() != frame.action_spaces.len() {
            return Err(invalid_actions(
                crate::error::FrameStatus::IncompleteActionSet,
                crate::error::ErrorCode::MissingSeat,
                selected.len() as i64,
                frame.action_spaces.len() as i64,
            ));
        }
        selected.sort_by_key(|value| value.0);
        Ok(selected)
    }

    /// Continue while the current rules state has no genuine decision. No
    /// forced frame or forced action is persisted: legal actions are derived
    /// from authoritative state at the instant they are needed.
    pub fn stabilize_automatic_decisions(&mut self) {
        self.stabilize_automatic_decisions_inner(true);
    }

    /// Replay supplies authoritative settlement records, so automatic rules
    /// resolution stops at Settlement instead of scoring and starting a new
    /// random hand.
    pub fn stabilize_replay_decisions(&mut self) {
        self.stabilize_automatic_decisions_inner(false);
    }

    /// Advance exactly one frame-free rules transition. This is the native
    /// half of Python's parameterless `advance`: it never invents a policy
    /// action and never skips the resulting observable state.
    pub fn advance_automatic_once(&mut self) -> bool {
        self.advance_automatic_once_inner(true)
    }

    fn stabilize_automatic_decisions_inner(&mut self, settle: bool) {
        for _ in 0..16_384 {
            if !self.advance_automatic_once_inner(settle) {
                return;
            }
        }
        panic!("automatic decision stabilization exceeded its safety bound")
    }

    fn advance_automatic_once_inner(&mut self, settle: bool) -> bool {
        let pending = self
            .hanchan
            .as_ref()
            .map(|game| (game.hand.phase, game.hand.decision.is_some()));
        let Some((phase, has_frame)) = pending else {
            return false;
        };
        if has_frame || self.lifecycle == EnvironmentLifecycle::Complete {
            return false;
        }
        let frame_id = self.next_frame_id.saturating_sub(1);
        match phase {
            HandPhase::SelfTurnDecision => {
                let (seat, actions) = {
                    let h = self.hanchan.as_ref().expect("initialized");
                    let seat = h.hand.current_seat;
                    (
                        seat,
                        action::semantic_representatives(
                            seat,
                            rules::legal::self_turn_for_hanchan(h, seat),
                        ),
                    )
                };
                assert_eq!(actions.len(), 1, "frame-free self turn is automatic");
                self.apply_actions_at_mode(
                    frame_id,
                    phase,
                    vec![(seat, actions.into_iter().next().expect("one action"))],
                    settle,
                );
                true
            }
            HandPhase::DiscardReactionFrame | HandPhase::KanRobReactionFrame => {
                self.apply_actions_at_mode(frame_id, phase, Vec::new(), settle);
                true
            }
            HandPhase::Settlement if settle => {
                self.finish_exhaustive_draw(frame_id);
                true
            }
            HandPhase::Settlement => false,
            _ => false,
        }
    }

    pub(crate) fn apply_actions(&mut self, selected: Vec<(u8, ActionCandidate)>) {
        self.apply_actions_mode(selected, true);
    }

    pub(crate) fn apply_replay_actions(&mut self, selected: Vec<(u8, ActionCandidate)>) {
        self.apply_actions_mode(selected, false);
    }

    fn apply_actions_mode(&mut self, selected: Vec<(u8, ActionCandidate)>, settle: bool) {
        let (frame_id, phase) = self
            .hanchan
            .as_ref()
            .and_then(|h| h.hand.decision.as_ref())
            .map(|frame| (frame.frame_id, frame.phase))
            .expect("validated frame");
        self.apply_actions_at_mode(frame_id, phase, selected, settle);
    }

    fn apply_actions_at_mode(
        &mut self,
        frame_id: u64,
        phase: HandPhase,
        selected: Vec<(u8, ActionCandidate)>,
        settle: bool,
    ) {
        if phase == HandPhase::SelfTurnDecision {
            let (seat, op) = &selected[0];
            let tile = op.tiles[0];
            let tsumogiri = self
                .hanchan
                .as_ref()
                .is_some_and(|h| tile == h.hand.current_draw);
            let had_pending_dora = self.pending_dora_reveal;
            // Tenhou's pending daiminkan/kakan indicator is revealed before
            // the next replacement-turn action, except an immediate rinshan
            // win. This also handles consecutive kans: the first indicator
            // precedes the second kan declaration.
            if !matches!(op.kind, ActionKind::Tsumo | ActionKind::AddedKan) {
                self.reveal_pending_dora(frame_id);
            }
            transition::apply_self_turn(self, *seat, op);
            match op.kind {
                ActionKind::Discard | ActionKind::RiichiDiscard => {
                    if op.kind == ActionKind::RiichiDiscard {
                        self.emit_public_at(
                            frame_id,
                            EventKind::Reach,
                            *seat,
                            ABSENT,
                            [0; 4],
                            Vec::new(),
                        );
                    }
                    self.emit_public_at(
                        frame_id,
                        EventKind::Dahai,
                        *seat,
                        ABSENT,
                        [i64::from(tile), i64::from(tsumogiri), 0, 0],
                        Vec::new(),
                    );
                }
                ActionKind::Tsumo => {
                    self.emit_public_at(
                        frame_id,
                        EventKind::Hora,
                        *seat,
                        *seat,
                        [i64::from(tile), 0, 0, 0],
                        Vec::new(),
                    );
                    self.finish_tsumo(frame_id, *seat, tile);
                }
                ActionKind::ClosedKan | ActionKind::AddedKan => {
                    let committed = self
                        .hanchan
                        .as_ref()
                        .is_some_and(|h| h.hand.phase != HandPhase::KanRobReactionFrame);
                    if committed {
                        self.emit_resolved_action(frame_id, *seat, op);
                        if op.kind == ActionKind::AddedKan && had_pending_dora {
                            // Tenhou places the previous kan's delayed dora
                            // after this kakan declaration but before its
                            // rinshan draw. Preserve the new kakan's pending
                            // indicator for the following action.
                            self.reveal_pending_dora(frame_id);
                            self.pending_dora_reveal = true;
                        }
                        if !self.pending_dora_reveal {
                            self.emit_latest_dora(frame_id);
                        }
                        if let Some((actor, draw)) = self
                            .hanchan
                            .as_ref()
                            .map(|h| (h.hand.current_seat, h.hand.current_draw))
                        {
                            self.emit_tsumo(frame_id, actor, draw);
                        }
                    }
                }
                ActionKind::AbortiveDeclaration => {
                    self.finish_abortive_draw(frame_id);
                }
                _ => unreachable!("self-turn action generation is authoritative"),
            }
        } else {
            let reaction_phase = phase;
            let source = self
                .hanchan
                .as_ref()
                .and_then(|h| {
                    if phase == HandPhase::KanRobReactionFrame {
                        h.hand.provisional_kan.as_ref().map(|kan| kan.seat)
                    } else {
                        h.hand.last_discard.map(|(seat, _)| seat)
                    }
                })
                .expect("reaction context");
            let resolved = rules::precedence::resolve(source, &selected);
            let accepts_riichi = self.hanchan.as_ref().is_some_and(|h| {
                reaction_phase == HandPhase::DiscardReactionFrame
                    && h.players[source as usize].riichi_state == phase::RiichiState::Declared
                    && !matches!(&resolved, rules::precedence::ReactionResolution::Ron(_))
            });
            let provisional_kan = self.hanchan.as_ref().and_then(|h| {
                (reaction_phase == HandPhase::KanRobReactionFrame)
                    .then(|| {
                        h.hand
                            .provisional_kan
                            .as_ref()
                            .map(|kan| kan.action.clone())
                    })
                    .flatten()
            });
            let pending_before_reaction = self.pending_dora_reveal;
            transition::apply_reactions(self, &selected);
            let abortive_after_discard = self.hanchan.as_ref().is_some_and(|h| {
                transition::is_four_winds_abortive(h)
                    || transition::is_four_riichi_abortive(h)
                    || transition::is_four_kans_abortive(h)
            });
            if accepts_riichi && !abortive_after_discard {
                let (score, deposits) = self
                    .hanchan
                    .as_ref()
                    .map(|h| (h.scores[source as usize], h.riichi_deposits))
                    .expect("initialized hanchan");
                self.emit_public_at(
                    frame_id,
                    EventKind::ReachAccepted,
                    source,
                    ABSENT,
                    [i64::from(score), i64::from(deposits), 0, 0],
                    Vec::new(),
                );
            }
            match resolved {
                rules::precedence::ReactionResolution::Ron(winners) => {
                    for (seat, action) in &winners {
                        self.emit_resolved_action(frame_id, *seat, action);
                    }
                    self.finish_ron(
                        frame_id,
                        source,
                        &winners,
                        reaction_phase == HandPhase::KanRobReactionFrame,
                    );
                }
                rules::precedence::ReactionResolution::Call(seat, action) => {
                    self.emit_resolved_action(frame_id, seat, &action);
                    if action.kind == ActionKind::OpenKan {
                        if !self.pending_dora_reveal {
                            self.emit_latest_dora(frame_id);
                        }
                        if let Some((actor, tile)) = self
                            .hanchan
                            .as_ref()
                            .map(|h| (h.hand.current_seat, h.hand.current_draw))
                        {
                            self.emit_tsumo(frame_id, actor, tile);
                        }
                    }
                }
                rules::precedence::ReactionResolution::AllPass => {
                    if reaction_phase == HandPhase::KanRobReactionFrame {
                        self.emit_resolved_action(
                            frame_id,
                            source,
                            provisional_kan
                                .as_ref()
                                .expect("kan-rob frame retains its proposal until resolution"),
                        );
                        if pending_before_reaction
                            && provisional_kan
                                .as_ref()
                                .is_some_and(|kan| kan.kind == ActionKind::AddedKan)
                        {
                            self.reveal_pending_dora(frame_id);
                            self.pending_dora_reveal = true;
                        }
                        if !self.pending_dora_reveal {
                            self.emit_latest_dora(frame_id);
                        }
                    }
                    if let Some((actor, tile)) = self
                        .hanchan
                        .as_ref()
                        .filter(|h| h.hand.phase == HandPhase::SelfTurnDecision)
                        .map(|h| (h.hand.current_seat, h.hand.current_draw))
                    {
                        self.emit_tsumo(frame_id, actor, tile);
                    }
                    if self
                        .hanchan
                        .as_ref()
                        .is_some_and(|h| h.hand.phase == HandPhase::Settlement)
                        && settle
                    {
                        if abortive_after_discard {
                            self.finish_abortive_draw(frame_id);
                        } else {
                            self.finish_exhaustive_draw(frame_id);
                        }
                    }
                }
            }
        }
    }

    pub(crate) fn install_decision(
        &mut self,
        phase: HandPhase,
        offered: Vec<action::ActionSpace>,
    ) -> u64 {
        let frame_id = self.next_frame_id;
        self.next_frame_id += 1;
        let (frame, automatic) = action::Decision::from_offered(
            self.environment_id,
            self.episode_generation,
            frame_id,
            phase,
            offered,
        );
        self.automatic_decisions = self.automatic_decisions.saturating_add(automatic);
        self.hanchan.as_mut().expect("initialized").hand.decision = frame;
        self.lifecycle = EnvironmentLifecycle::Running;
        frame_id
    }

    fn start_current_hand(&mut self) {
        self.pending_dora_reveal = false;
        let dealer = self.hanchan.as_ref().expect("initialized hanchan").dealer;
        let needs_fresh_hand = self
            .hanchan
            .as_ref()
            .is_some_and(|h| h.hand.phase != HandPhase::Setup);
        if needs_fresh_hand {
            let (players, hand) = fresh_hand(&mut self.rng, dealer);
            let h = self.hanchan.as_mut().expect("initialized hanchan");
            h.players = players;
            h.hand = hand;
        }
        self.emit_start_kyoku();
        let frame_id = transition::draw_and_offer(self).expect("fresh hand has a live draw");
        let (actor, tile) = self
            .hanchan
            .as_ref()
            .map(|h| (h.hand.current_seat, h.hand.current_draw))
            .expect("initialized hanchan");
        self.emit_tsumo(frame_id, actor, tile);
    }

    fn finish_exhaustive_draw(&mut self, frame_id: u64) {
        let rules = self.rules_profile();
        let (tenpai_mask, settlement, complete, scores, initial_seats) = {
            let h = self.hanchan.as_mut().expect("initialized hanchan");
            let tenpai_mask = h.players.iter().fold(0_u8, |mask, player| {
                let analysis = shanten::calculate(
                    &tile_type_counts(&player.concealed_tiles),
                    player.melds.len() as u8,
                );
                mask | if analysis.overall == 0 {
                    1 << player.seat
                } else {
                    0
                }
            });
            let settlement = exhaustive_draw(tenpai_mask, h.dealer, h.riichi_deposits);
            let complete =
                transition::apply_settlement_and_advance(h, rules, &settlement, true, false);
            h.completed_kyoku = h.completed_kyoku.saturating_add(1);
            (tenpai_mask, settlement, complete, h.scores, h.initial_seats)
        };

        let mut payload = Vec::with_capacity(16);
        for delta in settlement.deltas {
            payload.extend_from_slice(&delta.to_le_bytes());
        }
        self.emit_public_at(
            frame_id,
            EventKind::Ryukyoku,
            ABSENT,
            ABSENT,
            [
                i64::from(TerminalReason::ExhaustiveDraw as u8),
                i64::from(tenpai_mask),
                0,
                0,
            ],
            payload,
        );
        self.emit_public_at(
            frame_id,
            EventKind::EndKyoku,
            ABSENT,
            ABSENT,
            [0; 4],
            Vec::new(),
        );

        if complete {
            self.finish_hanchan(frame_id, scores, initial_seats);
        } else {
            self.start_current_hand();
        }
    }

    fn finish_tsumo(&mut self, frame_id: u64, winner: u8, win_tile: u8) {
        let rules = self.rules_profile();
        let (settlement, complete, scores, initial_seats, evaluation) = {
            let h = self.hanchan.as_mut().expect("initialized hanchan");
            let player = &h.players[winner as usize];
            let mut pre_win_tiles = player.concealed_tiles.clone();
            let tile_index = pre_win_tiles
                .iter()
                .rposition(|&tile| tile == win_tile)
                .expect("winning tile remains in the tsumo hand");
            pre_win_tiles.remove(tile_index);
            let context = rules::legal::winning_context(h, winner, true, false);
            let indicator_count = h.hand.wall.dora_indicator_count as usize;
            let ura_count = if context.riichi { indicator_count } else { 0 };
            let evaluation = evaluate_hand(
                &pre_win_tiles,
                &player.melds,
                win_tile,
                &h.hand.wall.revealed_dora_indicators[..indicator_count],
                &h.hand.wall.ura_indicators[..ura_count],
                &context,
            );
            assert!(
                evaluation.is_win,
                "legal tsumo must contain at least one yaku"
            );
            let settlement = settle_tsumo(
                winner,
                h.dealer,
                &evaluation.value,
                h.honba,
                h.riichi_deposits,
            );
            let complete =
                transition::apply_settlement_and_advance(h, rules, &settlement, false, false);
            h.completed_kyoku = h.completed_kyoku.saturating_add(1);
            (settlement, complete, h.scores, h.initial_seats, evaluation)
        };

        if let Some(hora) = self.pending_events.last_mut() {
            hora.args[1] = i64::from(evaluation.han);
            hora.args[2] = i64::from(evaluation.fu);
            hora.args[3] = i64::from(evaluation.value.base_points);
            for delta in settlement.deltas {
                hora.payload.extend_from_slice(&delta.to_le_bytes());
            }
        }
        self.emit_public_at(
            frame_id,
            EventKind::EndKyoku,
            ABSENT,
            ABSENT,
            [0; 4],
            Vec::new(),
        );
        if complete {
            self.finish_hanchan(frame_id, scores, initial_seats);
        } else {
            self.start_current_hand();
        }
    }

    fn finish_abortive_draw(&mut self, frame_id: u64) {
        let rules = self.rules_profile();
        let (complete, scores, initial_seats) = {
            let h = self.hanchan.as_mut().expect("initialized hanchan");
            let settlement = rules::settlement::Settlement {
                deltas: [0; 4],
                dealer_continues: true,
                deposits_after: h.riichi_deposits,
            };
            let complete =
                transition::apply_settlement_and_advance(h, rules, &settlement, true, true);
            h.completed_kyoku = h.completed_kyoku.saturating_add(1);
            (complete, h.scores, h.initial_seats)
        };
        self.emit_public_at(
            frame_id,
            EventKind::Ryukyoku,
            ABSENT,
            ABSENT,
            [i64::from(TerminalReason::AbortiveDraw as u8), 0, 0, 0],
            Vec::new(),
        );
        self.emit_public_at(
            frame_id,
            EventKind::EndKyoku,
            ABSENT,
            ABSENT,
            [0; 4],
            Vec::new(),
        );
        if complete {
            self.finish_hanchan(frame_id, scores, initial_seats);
        } else {
            self.start_current_hand();
        }
    }

    fn finish_ron(
        &mut self,
        frame_id: u64,
        loser: u8,
        winners: &[(u8, ActionCandidate)],
        chankan: bool,
    ) {
        let (claims, evaluations, dealer, honba, deposits) = {
            let h = self.hanchan.as_ref().expect("initialized hanchan");
            let indicator_count = h.hand.wall.dora_indicator_count as usize;
            let mut claims = Vec::with_capacity(winners.len());
            let mut evaluations = Vec::with_capacity(winners.len());
            for (winner, action) in winners {
                let player = &h.players[*winner as usize];
                let context = rules::legal::winning_context(h, *winner, false, chankan);
                let ura_count = if context.riichi { indicator_count } else { 0 };
                let evaluation = evaluate_hand(
                    &player.concealed_tiles,
                    &player.melds,
                    action.tiles[0],
                    &h.hand.wall.revealed_dora_indicators[..indicator_count],
                    &h.hand.wall.ura_indicators[..ura_count],
                    &context,
                );
                assert!(
                    evaluation.is_win,
                    "legal ron must contain at least one yaku"
                );
                claims.push(RonClaim {
                    winner: *winner,
                    value: evaluation.value.clone(),
                    liable_seat: None,
                    liable_base_points: 0,
                });
                evaluations.push(evaluation);
            }
            (claims, evaluations, h.dealer, h.honba, h.riichi_deposits)
        };
        let settlement = multiple_ron(&claims, loser, dealer, honba, deposits);
        let rules = self.rules_profile();
        let (complete, scores, initial_seats) = {
            let h = self.hanchan.as_mut().expect("initialized hanchan");
            let complete =
                transition::apply_settlement_and_advance(h, rules, &settlement, false, false);
            h.completed_kyoku = h.completed_kyoku.saturating_add(1);
            (complete, h.scores, h.initial_seats)
        };

        let first_hora = self.pending_events.len().saturating_sub(evaluations.len());
        for (index, evaluation) in evaluations.into_iter().enumerate() {
            if let Some(hora) = self.pending_events.get_mut(first_hora + index) {
                hora.args[1] = i64::from(evaluation.han);
                hora.args[2] = i64::from(evaluation.fu);
                hora.args[3] = i64::from(evaluation.value.base_points);
                for delta in settlement.deltas {
                    hora.payload.extend_from_slice(&delta.to_le_bytes());
                }
            }
        }
        self.emit_public_at(
            frame_id,
            EventKind::EndKyoku,
            ABSENT,
            ABSENT,
            [0; 4],
            Vec::new(),
        );
        if complete {
            self.finish_hanchan(frame_id, scores, initial_seats);
        } else {
            self.start_current_hand();
        }
    }

    fn finish_hanchan(&mut self, frame_id: u64, scores: [i32; 4], initial_seats: [u8; 4]) {
        let ranks = ranking(scores, initial_seats);
        let completed_kyoku = self
            .hanchan
            .as_ref()
            .expect("initialized hanchan")
            .completed_kyoku;
        let mut payload = Vec::with_capacity(20);
        for score in scores {
            payload.extend_from_slice(&score.to_le_bytes());
        }
        payload.extend_from_slice(&ranks);
        self.emit_public_at(
            frame_id,
            EventKind::EndGame,
            ABSENT,
            ABSENT,
            [i64::from(completed_kyoku), 0, 0, 0],
            payload,
        );
        self.lifecycle = EnvironmentLifecycle::Complete;
    }

    pub fn take_events(&mut self) -> Vec<EventRecord> {
        std::mem::take(&mut self.pending_events)
    }
    pub(crate) fn emit_start_game(&mut self) {
        self.emit_canonical_at(
            0,
            EventKind::StartGame,
            EventFields {
                actor_seat: ABSENT,
                target_seat: ABSENT,
                visibility_mask: 0b1111,
                args: [0; 4],
                payload: Vec::new(),
            },
        );
    }

    pub(crate) fn emit_start_kyoku(&mut self) {
        let h = self.hanchan.as_ref().expect("initialized hanchan");
        let args = [
            wind_code(h.round_wind),
            i64::from(h.hand_number) + 1,
            i64::from(h.honba),
            i64::from(h.dealer),
        ];
        let payload = start_kyoku_payload(h);
        self.emit_canonical_at(
            0,
            EventKind::StartKyoku,
            EventFields {
                actor_seat: ABSENT,
                target_seat: ABSENT,
                visibility_mask: 0,
                args,
                payload,
            },
        );
    }

    pub(crate) fn emit_tsumo(&mut self, frame_id: u64, actor: u8, tile: u8) {
        self.emit_canonical_at(
            frame_id,
            EventKind::Tsumo,
            EventFields {
                actor_seat: actor,
                target_seat: ABSENT,
                visibility_mask: 1 << actor,
                args: [i64::from(tile), 0, 0, 0],
                payload: Vec::new(),
            },
        );
    }

    fn emit_resolved_action(&mut self, frame_id: u64, actor: u8, action: &ActionCandidate) {
        let Some(kind) = event_kind_for_action(action.kind) else {
            return;
        };
        let (target, args) = if action.kind == ActionKind::Ron {
            (action.source_seat, [i64::from(action.tiles[0]), 0, 0, 0])
        } else if matches!(action.kind, ActionKind::ClosedKan | ActionKind::AddedKan) {
            let mut args = [i64::from(ABSENT); 4];
            for (index, tile) in action
                .tiles
                .iter()
                .copied()
                .take(action.tile_count as usize)
                .enumerate()
            {
                args[index] = i64::from(tile);
            }
            (ABSENT, args)
        } else {
            let called = self
                .hanchan
                .as_ref()
                .and_then(|h| h.hand.last_discard)
                .map_or(action.tiles[0], |(_, tile)| tile);
            let mut consumed = action
                .tiles
                .iter()
                .copied()
                .take(action.tile_count as usize)
                .collect::<Vec<_>>();
            if let Some(index) = consumed.iter().position(|&tile| tile == called) {
                consumed.remove(index);
            }
            let mut args = [
                i64::from(called),
                i64::from(ABSENT),
                i64::from(ABSENT),
                i64::from(ABSENT),
            ];
            for (index, tile) in consumed.into_iter().take(3).enumerate() {
                args[index + 1] = i64::from(tile);
            }
            (action.source_seat, args)
        };
        self.emit_public_at(frame_id, kind, actor, target, args, Vec::new());
    }

    fn emit_latest_dora(&mut self, frame_id: u64) {
        let indicator = self.hanchan.as_ref().map(|h| {
            h.hand.wall.revealed_dora_indicators
                [h.hand.wall.dora_indicator_count.saturating_sub(1) as usize]
        });
        if let Some(indicator) = indicator {
            self.emit_public_at(
                frame_id,
                EventKind::Dora,
                ABSENT,
                ABSENT,
                [i64::from(indicator), 0, 0, 0],
                Vec::new(),
            );
        }
    }

    fn reveal_pending_dora(&mut self, frame_id: u64) {
        if !self.pending_dora_reveal {
            return;
        }
        let revealed = self
            .hanchan
            .as_mut()
            .is_some_and(|h| rules::hand::reveal_next_dora(&mut h.hand.wall));
        self.pending_dora_reveal = false;
        if revealed {
            self.emit_latest_dora(frame_id);
        }
    }

    fn emit_public_at(
        &mut self,
        frame_id: u64,
        kind: EventKind,
        actor: u8,
        target: u8,
        args: [i64; 4],
        payload: Vec<u8>,
    ) {
        self.emit_canonical_at(
            frame_id,
            kind,
            EventFields {
                actor_seat: actor,
                target_seat: target,
                visibility_mask: 0b1111,
                args,
                payload,
            },
        );
    }

    fn emit_canonical_at(&mut self, _frame_id: u64, kind: EventKind, fields: EventFields) {
        let sequence = self.next_event_sequence;
        self.next_event_sequence += 1;
        self.pending_events.push(EventRecord {
            environment_id: self.environment_id,
            episode_generation: self.episode_generation,
            sequence,
            kind,
            actor_seat: fields.actor_seat,
            target_seat: fields.target_seat,
            visibility_mask: fields.visibility_mask,
            args: fields.args,
            payload: fields.payload,
        });
    }
}

fn invalid_actions(
    status: crate::error::FrameStatus,
    code: crate::error::ErrorCode,
    arg0: i64,
    arg1: i64,
) -> crate::error::CoreError {
    crate::error::CoreError::InvalidActions {
        status,
        code,
        arg0,
        arg1,
    }
}

fn fresh_hand(rng: &mut RngState, dealer: u8) -> ([PlayerState; 4], HandState) {
    let mut players = [
        PlayerState::new(0),
        PlayerState::new(1),
        PlayerState::new(2),
        PlayerState::new(3),
    ];
    let mut wall = initialize_wall(rng);
    deal(&mut wall, &mut players);
    let hand = HandState {
        phase: HandPhase::Setup,
        wall,
        current_seat: dealer,
        current_draw: ABSENT,
        current_draw_is_replacement: false,
        last_discard: None,
        provisional_kan: None,
        decision: None,
    };
    (players, hand)
}

#[cfg(test)]
mod decision_filter_tests {
    use super::*;
    use crate::snapshot;

    #[test]
    fn frame_free_settlement_is_completed_automatically() {
        let mut state = GameState::new(0);
        state.reset_from_seed(1);
        state.take_events();
        {
            let hand = &mut state.hanchan.as_mut().unwrap().hand;
            hand.phase = HandPhase::Settlement;
            hand.decision = None;
        }

        state.stabilize_automatic_decisions();

        let hand = &state.hanchan.as_ref().unwrap().hand;
        assert!(
            state.lifecycle == EnvironmentLifecycle::Complete || hand.decision.is_some(),
            "stabilization must end at a queryable decision or match completion"
        );
        let kinds = state
            .take_events()
            .into_iter()
            .map(|event| event.kind)
            .collect::<Vec<_>>();
        assert!(kinds.contains(&EventKind::Ryukyoku));
        assert!(kinds.contains(&EventKind::EndKyoku));
    }

    #[test]
    fn frame_free_all_pass_and_explicit_resolution_are_identical() {
        let mut stopped = None;
        for seed in 1..64 {
            let mut state = GameState::new(0);
            state.reset_from_seed(seed);
            state.take_events();
            for action in state.legal_selections() {
                let selected = state
                    .validate_selections(std::slice::from_ref(&action))
                    .unwrap();
                if !matches!(
                    selected[0].1.kind,
                    ActionKind::Discard | ActionKind::RiichiDiscard
                ) {
                    continue;
                }
                let mut candidate = state.clone();
                candidate.apply_actions(selected);
                if candidate.hanchan.as_ref().unwrap().hand.decision.is_none()
                    && candidate.hanchan.as_ref().unwrap().hand.phase
                        == HandPhase::DiscardReactionFrame
                {
                    stopped = Some(candidate);
                    break;
                }
            }
            if stopped.is_some() {
                break;
            }
        }
        let mut automatic = stopped.expect("a frame-free all-pass reaction");
        let mut explicit = automatic.clone();
        automatic.stabilize_automatic_decisions();
        let frame_id = explicit.next_frame_id - 1;
        explicit.apply_actions_at_mode(frame_id, HandPhase::DiscardReactionFrame, Vec::new(), true);
        assert_eq!(automatic.take_events(), explicit.take_events());
        assert_eq!(
            snapshot::encode(&automatic).unwrap(),
            snapshot::encode(&explicit).unwrap()
        );
    }
}
