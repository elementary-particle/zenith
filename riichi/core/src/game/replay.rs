//! Validated externally-authored hanchan replay.

use super::{
    action::{ActionCandidate, ActionKind, ActionSelection, ABSENT},
    event::{EventKind, EventRecord},
    phase::{EnvironmentLifecycle, HandPhase, Wind},
    rules::hand::{deal, wall_from_tiles},
    state::{derive_rng, GameState, HanchanState, PlayerState},
    transition,
};
use crate::error::CoreError;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReplayHanchan {
    pub environment_id: u32,
    pub round_wind: u8,
    pub hand_number: u8,
    pub dealer: u8,
    pub honba: u16,
    pub riichi_deposits: u16,
    pub completed_kyoku: u32,
    pub scores: [i32; 4],
    pub initial_seats: [u8; 4],
    pub wall: [u8; 136],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReplayEvent {
    pub environment_id: u32,
    pub kind: EventKind,
    pub actor_seat: u8,
    pub target_seat: u8,
    pub tile: u8,
    pub consumed: Vec<u8>,
    pub tsumogiri: bool,
    pub deltas: [i32; 4],
    pub dora_marker: u8,
    pub ura_markers: Vec<u8>,
}

impl ReplayEvent {
    fn primary_tile(&self) -> u8 {
        if self.kind == EventKind::Dora {
            self.dora_marker
        } else {
            self.tile
        }
    }
}

fn invalid(message: impl Into<String>) -> CoreError {
    CoreError::InvalidArgument(message.into())
}

fn validate_event_input(event: &ReplayEvent) -> Result<(), CoreError> {
    if (event.actor_seat != ABSENT && event.actor_seat >= 4)
        || (event.target_seat != ABSENT && event.target_seat >= 4)
    {
        return Err(invalid("replay event has an invalid actor or target seat"));
    }
    if (event.tile != ABSENT && event.tile >= 136)
        || (event.dora_marker != ABSENT && event.dora_marker >= 136)
        || event
            .consumed
            .iter()
            .chain(&event.ura_markers)
            .any(|tile| *tile >= 136)
    {
        return Err(invalid("replay event has an invalid physical tile"));
    }
    let actor_required = matches!(
        event.kind,
        EventKind::Tsumo
            | EventKind::Dahai
            | EventKind::Chi
            | EventKind::Pon
            | EventKind::Daiminkan
            | EventKind::Ankan
            | EventKind::Kakan
            | EventKind::Reach
            | EventKind::ReachAccepted
            | EventKind::Hora
    );
    let target_required = matches!(
        event.kind,
        EventKind::Chi | EventKind::Pon | EventKind::Daiminkan | EventKind::Hora
    );
    let tile_required = matches!(
        event.kind,
        EventKind::Tsumo
            | EventKind::Dahai
            | EventKind::Chi
            | EventKind::Pon
            | EventKind::Daiminkan
            | EventKind::Kakan
            | EventKind::Hora
    );
    if (actor_required && event.actor_seat == ABSENT)
        || (target_required && event.target_seat == ABSENT)
        || (tile_required && event.tile == ABSENT)
        || (event.kind == EventKind::Dora && event.dora_marker == ABSENT)
    {
        return Err(invalid("replay event is missing a required typed field"));
    }
    Ok(())
}

fn semantic_tile(tile: u8) -> (u8, bool) {
    let tile_type = tile / 4;
    (tile_type, matches!(tile_type, 4 | 13 | 22) && tile % 4 == 0)
}

fn same_semantic_tile(left: u8, right: u8) -> bool {
    semantic_tile(left) == semantic_tile(right)
}

fn action_matches(action: &ActionCandidate, event: &ReplayEvent) -> bool {
    let expected = match event.kind {
        EventKind::Dahai => ActionKind::Discard,
        EventKind::Reach => ActionKind::RiichiDiscard,
        EventKind::Chi => ActionKind::Chi,
        EventKind::Pon => ActionKind::Pon,
        EventKind::Daiminkan => ActionKind::OpenKan,
        EventKind::Ankan => ActionKind::ClosedKan,
        EventKind::Kakan => ActionKind::AddedKan,
        EventKind::Hora if event.actor_seat == event.target_seat => ActionKind::Tsumo,
        EventKind::Hora => ActionKind::Ron,
        EventKind::Ryukyoku => ActionKind::AbortiveDeclaration,
        _ => return false,
    };
    if action.kind != expected {
        return false;
    }
    match expected {
        ActionKind::Discard
        | ActionKind::RiichiDiscard
        | ActionKind::AddedKan
        | ActionKind::Ron
        | ActionKind::Tsumo => same_semantic_tile(action.tiles[0], event.tile),
        ActionKind::Chi | ActionKind::Pon | ActionKind::OpenKan => {
            let mut actual = action.tiles[..usize::from(action.tile_count)]
                .iter()
                .copied()
                .map(semantic_tile)
                .collect::<Vec<_>>();
            let mut expected = event.consumed.clone();
            expected.push(event.tile);
            let mut expected = expected.into_iter().map(semantic_tile).collect::<Vec<_>>();
            actual.sort_unstable();
            expected.sort_unstable();
            actual == expected && action.source_seat == event.target_seat
        }
        ActionKind::ClosedKan => {
            let mut actual = action.tiles[..usize::from(action.tile_count)]
                .iter()
                .copied()
                .map(semantic_tile)
                .collect::<Vec<_>>();
            let mut expected = event
                .consumed
                .iter()
                .copied()
                .map(semantic_tile)
                .collect::<Vec<_>>();
            actual.sort_unstable();
            expected.sort_unstable();
            actual == expected
        }
        ActionKind::AbortiveDeclaration => true,
        ActionKind::Pass => false,
    }
}

fn specialize_action(action: &ActionCandidate, event: &ReplayEvent) -> ActionCandidate {
    let mut result = action.clone();
    match action.kind {
        ActionKind::Discard
        | ActionKind::RiichiDiscard
        | ActionKind::AddedKan
        | ActionKind::Ron
        | ActionKind::Tsumo => result.tiles[0] = event.tile,
        ActionKind::Chi | ActionKind::Pon | ActionKind::OpenKan => {
            let mut tiles = event.consumed.clone();
            tiles.push(event.tile);
            tiles.sort_unstable();
            result.tile_count = tiles.len() as u8;
            result.tiles = [ABSENT; 4];
            result.tiles[..tiles.len()].copy_from_slice(&tiles);
        }
        ActionKind::ClosedKan => {
            let mut tiles = event.consumed.clone();
            tiles.sort_unstable();
            result.tile_count = tiles.len() as u8;
            result.tiles = [ABSENT; 4];
            result.tiles[..tiles.len()].copy_from_slice(&tiles);
        }
        ActionKind::Pass | ActionKind::AbortiveDeclaration => {}
    }
    result
}

fn owns_all(owned: &[u8], required: &[u8]) -> bool {
    let mut remaining = owned.to_vec();
    required.iter().all(|tile| {
        remaining
            .iter()
            .position(|owned| owned == tile)
            .is_some_and(|index| {
                remaining.remove(index);
                true
            })
    })
}

fn compare_event(actual: &EventRecord, expected: &ReplayEvent) -> Result<(), CoreError> {
    if actual.kind != expected.kind
        || (expected.actor_seat != ABSENT && actual.actor_seat != expected.actor_seat)
        || (expected.target_seat != ABSENT && actual.target_seat != expected.target_seat)
    {
        return Err(invalid(format!(
            "replay event mismatch: native={} actor={} target={}, source={} actor={} target={}",
            actual.kind.mjai_name(),
            actual.actor_seat,
            actual.target_seat,
            expected.kind.mjai_name(),
            expected.actor_seat,
            expected.target_seat,
        )));
    }
    let expected_tile = expected.primary_tile();
    let physical_tile_matches = actual.args[0] == i64::from(expected_tile);
    // An ankan names a tile type, not a distinguished physical copy.  In
    // particular, Tenhou records may choose the red or a non-red five as the
    // representative while the native action representative chooses another
    // copy from the same four-tile group.
    let semantic_ankan_matches = actual.kind == EventKind::Ankan
        && u8::try_from(actual.args[0]).is_ok_and(|tile| tile / 4 == expected_tile / 4);
    if expected_tile != ABSENT && !physical_tile_matches && !semantic_ankan_matches {
        return Err(invalid(format!(
            "replay tile mismatch for {}: native={}, source={expected_tile}",
            actual.kind.mjai_name(),
            actual.args[0],
        )));
    }
    if actual.kind == EventKind::Dahai
        && bool::try_from(actual.args[1]).unwrap_or(false) != expected.tsumogiri
    {
        return Err(invalid("replay tsumogiri flag mismatch"));
    }
    Ok(())
}

impl GameState {
    pub fn load_replay_hanchan(&mut self, value: &ReplayHanchan) -> Result<(), CoreError> {
        if value.environment_id != self.environment_id {
            return Err(invalid("replay hanchan environment does not match slot"));
        }
        let round_wind = match value.round_wind {
            0 => Wind::East,
            1 => Wind::South,
            2 => Wind::West,
            3 => Wind::North,
            other => return Err(invalid(format!("invalid replay round wind {other}"))),
        };
        if value.hand_number >= 4 || value.dealer >= 4 {
            return Err(invalid("invalid replay hand number or dealer"));
        }
        if value.scores.iter().sum::<i32>() + i32::from(value.riichi_deposits) * 1_000
            != self.rules_profile().starting_points * 4
        {
            return Err(invalid(
                "replay scores and riichi deposits do not conserve points",
            ));
        }
        let mut seen = [false; 4];
        for &seat in &value.initial_seats {
            let Some(entry) = seen.get_mut(usize::from(seat)) else {
                return Err(invalid("invalid replay initial seat"));
            };
            if *entry {
                return Err(invalid("duplicate replay initial seat"));
            }
            *entry = true;
        }
        let mut wall = wall_from_tiles(value.wall)
            .ok_or_else(|| invalid("replay wall is not a physical tile permutation"))?;
        let mut players = [
            PlayerState::new(0),
            PlayerState::new(1),
            PlayerState::new(2),
            PlayerState::new(3),
        ];
        deal(&mut wall, &mut players);
        let generation = self.episode_generation.wrapping_add(1);
        self.episode_generation = generation;
        self.lifecycle = EnvironmentLifecycle::Ready;
        self.rng = derive_rng(0, self.environment_id, generation);
        self.next_frame_id = 1;
        self.next_event_sequence = 0;
        self.failure = None;
        self.externally_loaded = true;
        self.pending_dora_reveal = false;
        self.pending_events.clear();
        self.automatic_decisions = 0;
        self.hanchan = Some(HanchanState {
            round_wind,
            hand_number: value.hand_number,
            dealer: value.dealer,
            honba: value.honba,
            riichi_deposits: value.riichi_deposits,
            completed_kyoku: value.completed_kyoku,
            scores: value.scores,
            initial_seats: value.initial_seats,
            players,
            hand: super::state::HandState {
                phase: HandPhase::Setup,
                wall,
                current_seat: value.dealer,
                current_draw: ABSENT,
                current_draw_is_replacement: false,
                last_discard: None,
                provisional_kan: None,
                decision: None,
            },
        });
        self.emit_start_game();
        self.emit_start_kyoku();
        Ok(())
    }

    pub fn apply_replay_events(
        &mut self,
        events: &[ReplayEvent],
    ) -> Result<Vec<ActionSelection>, CoreError> {
        if events.is_empty() {
            return Err(invalid("replay event group must not be empty"));
        }
        if events
            .iter()
            .any(|event| event.environment_id != self.environment_id)
        {
            return Err(invalid("replay event environment does not match slot"));
        }
        for event in events {
            validate_event_input(event)?;
        }
        let phase = self
            .hanchan
            .as_ref()
            .ok_or_else(|| invalid("replay slot is uninitialized"))?
            .hand
            .phase;
        if phase == HandPhase::Setup {
            if events.len() != 1 || events[0].kind != EventKind::Tsumo {
                return Err(invalid("a replay hand must begin with exactly one tsumo"));
            }
            let expected = events[0].tile;
            let actual = self.hanchan.as_ref().unwrap().hand.wall.tiles
                [usize::from(self.hanchan.as_ref().unwrap().hand.wall.live_start)];
            if actual != expected || events[0].actor_seat != self.hanchan.as_ref().unwrap().dealer {
                return Err(invalid(
                    "replay initial draw does not match fixed wall/dealer",
                ));
            }
            let frame_id = transition::draw_and_offer(self)
                .ok_or_else(|| invalid("replay fixed wall has no initial draw"))?;
            self.emit_tsumo(frame_id, events[0].actor_seat, actual);
            self.lifecycle = EnvironmentLifecycle::Running;
            return Ok(Vec::new());
        }

        let terminal = events
            .iter()
            .any(|event| matches!(event.kind, EventKind::Hora | EventKind::Ryukyoku));
        if phase == HandPhase::Settlement {
            if !terminal || events.iter().any(|event| event.kind != EventKind::Ryukyoku) {
                return Err(invalid(
                    "settlement replay requires an authoritative ryukyoku",
                ));
            }
            self.apply_authoritative_settlement(events);
            return Ok(Vec::new());
        }
        let frame = self
            .hanchan
            .as_ref()
            .and_then(|h| h.hand.decision.as_ref())
            .ok_or_else(|| invalid("replay event group has no pending decision"))?
            .clone();
        let combined_reach = if events
            .first()
            .is_some_and(|event| event.kind == EventKind::Reach)
        {
            let discard = events
                .get(1)
                .filter(|event| event.kind == EventKind::Dahai)
                .ok_or_else(|| invalid("reach must be followed by dahai in one replay group"))?;
            let mut event = events[0].clone();
            event.tile = discard.tile;
            event.tsumogiri = discard.tsumogiri;
            Some(event)
        } else {
            None
        };
        let mut descriptors = Vec::with_capacity(frame.action_spaces.len());
        let mut bound = Vec::with_capacity(frame.action_spaces.len());
        for decision in &frame.action_spaces {
            let source = combined_reach
                .as_ref()
                .filter(|event| event.actor_seat == decision.seat)
                .or_else(|| {
                    events.iter().find(|event| {
                        matches!(
                            event.kind,
                            EventKind::Dahai
                                | EventKind::Chi
                                | EventKind::Pon
                                | EventKind::Daiminkan
                                | EventKind::Ankan
                                | EventKind::Kakan
                                | EventKind::Hora
                        ) && event.actor_seat == decision.seat
                    })
                })
                .or_else(|| {
                    events.iter().find(|event| {
                        frame.phase == HandPhase::SelfTurnDecision
                            && event.kind == EventKind::Ryukyoku
                    })
                });
            let (descriptor, index) = if let Some(event) = source {
                let (index, representative) = decision
                    .candidates
                    .iter()
                    .enumerate()
                    .find(|(_, action)| action_matches(action, event))
                    .ok_or_else(|| {
                        invalid(format!(
                            "recorded {} is not legal for seat {}",
                            event.kind.mjai_name(),
                            decision.seat,
                        ))
                    })?;
                self.validate_replay_action(decision.seat, event)?;
                (specialize_action(representative, event), index)
            } else {
                let (index, representative) = decision
                    .candidates
                    .iter()
                    .enumerate()
                    .find(|(_, action)| action.kind == ActionKind::Pass)
                    .ok_or_else(|| {
                        invalid(format!(
                            "record omits a non-pass decision for seat {}",
                            decision.seat,
                        ))
                    })?;
                (representative.clone(), index)
            };
            bound.push(ActionSelection::bind(&frame, decision.seat, index as u32));
            descriptors.push((decision.seat, descriptor));
        }

        if terminal {
            self.apply_authoritative_settlement(events);
            return Ok(bound);
        }

        let pending_start = self.pending_events.len();
        self.apply_replay_actions(descriptors);
        self.stabilize_replay_decisions();
        let generated_count = self.pending_events.len().saturating_sub(pending_start);
        if generated_count == 0
            && events.len() == 1
            && matches!(events[0].kind, EventKind::Ankan | EventKind::Kakan)
            && self
                .hanchan
                .as_ref()
                .is_some_and(|h| h.hand.phase == HandPhase::KanRobReactionFrame)
        {
            self.emit_external_event(&events[0]);
            return Ok(bound);
        }
        if frame.phase == HandPhase::KanRobReactionFrame
            && self
                .pending_events
                .get(pending_start)
                .is_some_and(|event| matches!(event.kind, EventKind::Ankan | EventKind::Kakan))
        {
            // The proposal was already emitted on the preceding replay call
            // so the source record could expose the intervening chankan
            // decision. Resolution commits the kan internally and emits it
            // again; hide that duplicate before matching the subsequent
            // source dora, rinshan draw, or terminal event.
            self.pending_events.remove(pending_start);
            for event in self.pending_events.iter_mut().skip(pending_start) {
                event.sequence = event.sequence.saturating_sub(1);
            }
            self.next_event_sequence = self.next_event_sequence.saturating_sub(1);
        }
        let generated = &self.pending_events[pending_start..];
        if generated.len() < events.len() {
            return Err(invalid(format!(
                "replay event count mismatch: native={}, source={} (source exceeds native)",
                generated.len(),
                events.len(),
            )));
        }
        for (actual, expected) in generated.iter().take(events.len()).zip(events) {
            compare_event(actual, expected)?;
        }
        Ok(bound)
    }

    fn apply_authoritative_settlement(&mut self, events: &[ReplayEvent]) {
        let h = self.hanchan.as_mut().expect("replay hanchan exists");
        for event in events {
            for (score, delta) in h.scores.iter_mut().zip(event.deltas) {
                *score = score.saturating_add(delta);
            }
        }
        if events.iter().any(|event| event.kind == EventKind::Hora) {
            h.riichi_deposits = 0;
        }
        h.hand.decision = None;
        h.hand.phase = HandPhase::HanchanComplete;
        self.lifecycle = EnvironmentLifecycle::Complete;
        for event in events {
            self.emit_external_event(event);
        }
    }

    fn validate_replay_action(&self, seat: u8, event: &ReplayEvent) -> Result<(), CoreError> {
        let h = self.hanchan.as_ref().expect("replay hanchan exists");
        let player = &h.players[usize::from(seat)];
        let required = match event.kind {
            EventKind::Dahai | EventKind::Reach | EventKind::Kakan => {
                std::slice::from_ref(&event.tile)
            }
            EventKind::Chi | EventKind::Pon | EventKind::Daiminkan | EventKind::Ankan => {
                event.consumed.as_slice()
            }
            _ => &[],
        };
        if !owns_all(&player.concealed_tiles, required) {
            return Err(invalid(format!(
                "recorded {} uses tiles not owned by seat {seat}",
                event.kind.mjai_name(),
            )));
        }
        if matches!(event.kind, EventKind::Dahai | EventKind::Reach)
            && event.tsumogiri != (event.tile == h.hand.current_draw)
        {
            return Err(invalid(
                "recorded tsumogiri flag does not match the physical draw",
            ));
        }
        if matches!(
            event.kind,
            EventKind::Chi | EventKind::Pon | EventKind::Daiminkan
        ) && h.hand.last_discard != Some((event.target_seat, event.tile))
        {
            return Err(invalid(
                "recorded call does not consume the latest physical discard",
            ));
        }
        if event.kind == EventKind::Hora {
            let winning_tile = if event.actor_seat == event.target_seat {
                h.hand.current_draw
            } else if h.hand.phase == HandPhase::KanRobReactionFrame {
                h.hand
                    .provisional_kan
                    .as_ref()
                    .map_or(ABSENT, |kan| kan.action.tiles[0])
            } else {
                h.hand.last_discard.map_or(ABSENT, |(_, tile)| tile)
            };
            if event.tile != winning_tile {
                return Err(invalid(
                    "recorded hora uses the wrong physical winning tile",
                ));
            }
        }
        Ok(())
    }

    fn emit_external_event(&mut self, event: &ReplayEvent) {
        let mut args = [0_i64; 4];
        if event.kind == EventKind::Ankan {
            for (index, &tile) in event.consumed.iter().take(4).enumerate() {
                args[index] = i64::from(tile);
            }
        } else {
            args[0] = i64::from(event.primary_tile());
            if event.kind == EventKind::Dahai {
                args[1] = i64::from(event.tsumogiri);
            }
            for (index, &tile) in event.consumed.iter().take(3).enumerate() {
                args[index + 1] = i64::from(tile);
            }
        }
        let mut payload = Vec::new();
        if matches!(event.kind, EventKind::Hora | EventKind::Ryukyoku) {
            for delta in event.deltas {
                payload.extend_from_slice(&delta.to_le_bytes());
            }
        }
        let sequence = self.next_event_sequence;
        self.next_event_sequence += 1;
        self.pending_events.push(EventRecord {
            environment_id: self.environment_id,
            episode_generation: self.episode_generation,
            sequence,
            kind: event.kind,
            actor_seat: event.actor_seat,
            target_seat: event.target_seat,
            visibility_mask: if event.kind == EventKind::Tsumo {
                1 << event.actor_seat
            } else {
                0b1111
            },
            args,
            payload,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ankan_event_comparison_uses_tile_type_not_physical_copy() {
        let actual = EventRecord {
            environment_id: 0,
            episode_generation: 1,
            sequence: 0,
            kind: EventKind::Ankan,
            actor_seat: 2,
            target_seat: ABSENT,
            visibility_mask: 0b1111,
            args: [16, 17, 18, 19],
            payload: Vec::new(),
        };
        let expected = ReplayEvent {
            environment_id: 0,
            kind: EventKind::Ankan,
            actor_seat: 2,
            target_seat: ABSENT,
            tile: 17,
            consumed: vec![16, 17, 18, 19],
            tsumogiri: false,
            deltas: [0; 4],
            dora_marker: ABSENT,
            ura_markers: Vec::new(),
        };

        compare_event(&actual, &expected).expect("same-type ankan representatives must match");
    }
}
