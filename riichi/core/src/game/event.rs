use serde::{Deserialize, Serialize};

use super::{action::ActionKind, phase::Wind, state::HanchanState};

/// Stable binary discriminants whose names and semantics match MJAI game-event `type` values.
///
/// Network-only RiichiLab messages (`request_action` and `action_ack`) intentionally do not appear
/// here. A pending [`DecisionFrame`](crate::game::action::DecisionFrame) is the environment's
/// timing-free action-request boundary.
#[repr(u16)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum EventKind {
    StartGame = 1,
    StartKyoku = 2,
    Tsumo = 3,
    Dahai = 4,
    Chi = 5,
    Pon = 6,
    Daiminkan = 7,
    Ankan = 8,
    Kakan = 9,
    Dora = 10,
    Reach = 11,
    ReachAccepted = 12,
    Hora = 13,
    Ryukyoku = 14,
    EndKyoku = 15,
    EndGame = 16,
}

impl EventKind {
    pub const fn mjai_name(self) -> &'static str {
        match self {
            Self::StartGame => "start_game",
            Self::StartKyoku => "start_kyoku",
            Self::Tsumo => "tsumo",
            Self::Dahai => "dahai",
            Self::Chi => "chi",
            Self::Pon => "pon",
            Self::Daiminkan => "daiminkan",
            Self::Ankan => "ankan",
            Self::Kakan => "kakan",
            Self::Dora => "dora",
            Self::Reach => "reach",
            Self::ReachAccepted => "reach_accepted",
            Self::Hora => "hora",
            Self::Ryukyoku => "ryukyoku",
            Self::EndKyoku => "end_kyoku",
            Self::EndGame => "end_game",
        }
    }
}

pub const MJAI_EVENT_NAMES: [(u16, &str); 16] = [
    (1, "start_game"),
    (2, "start_kyoku"),
    (3, "tsumo"),
    (4, "dahai"),
    (5, "chi"),
    (6, "pon"),
    (7, "daiminkan"),
    (8, "ankan"),
    (9, "kakan"),
    (10, "dora"),
    (11, "reach"),
    (12, "reach_accepted"),
    (13, "hora"),
    (14, "ryukyoku"),
    (15, "end_kyoku"),
    (16, "end_game"),
];

/// One canonical, unmasked game event.
///
/// `visibility_mask` documents which seats may see sensitive fields at full fidelity under ordinary
/// play. The canonical values remain present so Python can construct standard MJAI views or
/// intentionally reveal hidden features for experimental training.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct EventRecord {
    pub environment_id: u32,
    pub episode_generation: u64,
    pub sequence: u64,
    pub kind: EventKind,
    pub actor_seat: u8,
    pub target_seat: u8,
    pub visibility_mask: u8,
    pub args: [i64; 4],
    pub payload: Vec<u8>,
}

pub fn start_kyoku_payload(hanchan: &HanchanState) -> Vec<u8> {
    let tile_count: usize = hanchan
        .players
        .iter()
        .map(|player| player.concealed_tiles.len())
        .sum();
    let mut payload = Vec::with_capacity(24 + tile_count);
    payload.push(2); // payload version
    payload.push(hanchan.hand.wall.revealed_dora_indicators[0]);
    payload.extend_from_slice(&hanchan.riichi_deposits.to_le_bytes());
    for score in hanchan.scores {
        payload.extend_from_slice(&score.to_le_bytes());
    }
    for player in &hanchan.players {
        payload.push(u8::try_from(player.concealed_tiles.len()).expect("hand length is bounded"));
    }
    for player in &hanchan.players {
        payload.extend_from_slice(&player.concealed_tiles);
    }
    payload
}

pub const fn wind_code(wind: Wind) -> i64 {
    wind as i64
}

pub const fn event_kind_for_action(kind: ActionKind) -> Option<EventKind> {
    match kind {
        ActionKind::Chi => Some(EventKind::Chi),
        ActionKind::Pon => Some(EventKind::Pon),
        ActionKind::OpenKan => Some(EventKind::Daiminkan),
        ActionKind::ClosedKan => Some(EventKind::Ankan),
        ActionKind::AddedKan => Some(EventKind::Kakan),
        ActionKind::Ron | ActionKind::Tsumo => Some(EventKind::Hora),
        _ => None,
    }
}
