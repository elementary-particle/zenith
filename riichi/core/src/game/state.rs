use serde::{Deserialize, Serialize};

use crate::error::FailureRecord;

use super::{
    action::DecisionFrame,
    event::EventRecord,
    phase::{EnvironmentLifecycle, HandPhase, MeldKind, RiichiState, Wind},
};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RngState {
    pub state: u64,
    pub stream: u64,
}

impl RngState {
    pub fn next_u32(&mut self) -> u32 {
        let old = self.state;
        self.state = old
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(self.stream | 1);
        let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
        let rot = (old >> 59) as u32;
        xorshifted.rotate_right(rot)
    }
    pub fn bounded(&mut self, bound: u32) -> u32 {
        assert!(bound > 0);
        let threshold = bound.wrapping_neg() % bound;
        loop {
            let value = self.next_u32();
            if value >= threshold {
                return value % bound;
            }
        }
    }
}

pub fn derive_rng(master_seed: u64, environment_id: u32, episode_generation: u64) -> RngState {
    let state = mix64(
        master_seed
            ^ (u64::from(environment_id) << 32)
            ^ episode_generation
            ^ 0x5243_4849_5354_4154,
    );
    let stream = mix64(
        master_seed
            ^ u64::from(environment_id)
            ^ episode_generation.rotate_left(17)
            ^ 0x5243_4849_5354_524d,
    ) | 1;
    RngState { state, stream }
}

fn mix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Meld {
    pub kind: MeldKind,
    pub tiles: [u8; 4],
    pub tile_count: u8,
    pub called_tile: u8,
    pub from_seat: u8,
    pub created_sequence: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RiverEntry {
    pub tile: u8,
    pub sequence: u64,
    pub riichi_declaration: bool,
    pub called: bool,
    pub tsumogiri: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PlayerState {
    pub seat: u8,
    pub concealed_tiles: Vec<u8>,
    pub melds: Vec<Meld>,
    pub river: Vec<RiverEntry>,
    pub riichi_state: RiichiState,
    pub ippatsu_eligible: bool,
    pub permanent_furiten: bool,
    pub temporary_furiten: bool,
    pub riichi_furiten: bool,
    pub forbidden_discard_mask: u64,
}

impl PlayerState {
    pub fn new(seat: u8) -> Self {
        Self {
            seat,
            concealed_tiles: Vec::with_capacity(14),
            melds: Vec::with_capacity(4),
            river: Vec::with_capacity(24),
            riichi_state: RiichiState::None,
            ippatsu_eligible: false,
            permanent_furiten: false,
            temporary_furiten: false,
            riichi_furiten: false,
            forbidden_discard_mask: 0,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct WallState {
    #[serde(with = "array136")]
    pub tiles: [u8; 136],
    #[serde(with = "array34")]
    pub live_wall_counts: [u8; 34],
    pub live_start: u8,
    pub live_end: u8,
    pub rinshan_index: u8,
    pub dora_indicator_count: u8,
    pub revealed_dora_indicators: [u8; 5],
    pub ura_indicators: [u8; 5],
}

mod array34 {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(value: &[u8; 34], serializer: S) -> Result<S::Ok, S::Error> {
        value.as_slice().serialize(serializer)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<[u8; 34], D::Error> {
        let value = Vec::<u8>::deserialize(deserializer)?;
        value.try_into().map_err(|value: Vec<u8>| {
            serde::de::Error::invalid_length(value.len(), &"exactly 34 tile-type counts")
        })
    }
}

mod array136 {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(value: &[u8; 136], serializer: S) -> Result<S::Ok, S::Error> {
        value.as_slice().serialize(serializer)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<[u8; 136], D::Error> {
        let value = Vec::<u8>::deserialize(deserializer)?;
        value.try_into().map_err(|value: Vec<u8>| {
            serde::de::Error::invalid_length(value.len(), &"exactly 136 physical tiles")
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct HandState {
    pub phase: HandPhase,
    pub wall: WallState,
    pub current_seat: u8,
    pub current_draw: u8,
    pub current_draw_is_replacement: bool,
    pub last_discard: Option<(u8, u8)>,
    pub provisional_kan: Option<ProvisionalKan>,
    pub decision_frame: Option<DecisionFrame>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ProvisionalKan {
    pub seat: u8,
    pub action: super::action::ActionDescriptor,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct HanchanState {
    pub round_wind: Wind,
    pub hand_number: u8,
    pub dealer: u8,
    pub honba: u16,
    pub riichi_deposits: u16,
    /// Number of completed kyoku in this match, including dealer repeats.
    pub completed_kyoku: u32,
    pub scores: [i32; 4],
    pub initial_seats: [u8; 4],
    pub players: [PlayerState; 4],
    pub hand: HandState,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct GameState {
    pub environment_id: u32,
    pub episode_generation: u64,
    pub lifecycle: EnvironmentLifecycle,
    pub rng: RngState,
    pub hanchan: Option<HanchanState>,
    pub next_frame_id: u64,
    pub next_event_sequence: u64,
    pub failure: Option<FailureRecord>,
    #[serde(skip)]
    pub pending_events: Vec<EventRecord>,
    /// Monotonic process-local observability counter. This is deliberately
    /// absent from snapshots because it is not gameplay state.
    #[serde(skip)]
    pub automatic_decisions: u64,
}
