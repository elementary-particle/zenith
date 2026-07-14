use serde::{Deserialize, Serialize};

use super::phase::HandPhase;

pub const ABSENT: u8 = 255;

#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub enum ActionKind {
    Pass = 0,
    Discard = 1,
    RiichiDiscard = 2,
    Chi = 3,
    Pon = 4,
    OpenKan = 5,
    ClosedKan = 6,
    AddedKan = 7,
    Ron = 8,
    Tsumo = 9,
    AbortiveDeclaration = 10,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ActionDescriptor {
    pub kind: ActionKind,
    pub primary_tile_type: u8,
    pub source_seat: u8,
    pub tile_count: u8,
    pub tiles: [u8; 4],
    pub aux: u16,
    pub flags: u16,
}

impl ActionDescriptor {
    pub fn pass() -> Self {
        Self {
            kind: ActionKind::Pass,
            primary_tile_type: ABSENT,
            source_seat: ABSENT,
            tile_count: 0,
            tiles: [ABSENT; 4],
            aux: 0,
            flags: 0,
        }
    }
    pub fn discard(tile: u8) -> Self {
        Self {
            kind: ActionKind::Discard,
            primary_tile_type: tile / 4,
            source_seat: ABSENT,
            tile_count: 1,
            tiles: [tile, ABSENT, ABSENT, ABSENT],
            aux: 0,
            flags: 0,
        }
    }
}

impl Ord for ActionDescriptor {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (
            self.kind,
            self.primary_tile_type,
            self.source_seat,
            self.tiles,
            self.aux,
            self.flags,
        )
            .cmp(&(
                other.kind,
                other.primary_tile_type,
                other.source_seat,
                other.tiles,
                other.aux,
                other.flags,
            ))
    }
}
impl PartialOrd for ActionDescriptor {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SeatDecision {
    pub seat: u8,
    pub actions: Vec<ActionDescriptor>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct DecisionFrame {
    pub environment_id: u32,
    pub episode_generation: u64,
    pub frame_id: u64,
    pub phase: HandPhase,
    pub eligible_mask: u8,
    pub decisions: Vec<SeatDecision>,
}

/// One immutable legal action bound to the exact decision that produced it.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Action {
    pub environment_id: u32,
    pub episode_generation: u64,
    pub frame_id: u64,
    pub seat: u8,
    pub action_index: u32,
    pub kind: ActionKind,
    pub primary_tile_type: u8,
    pub source_seat: u8,
    pub tile_count: u8,
    pub tiles: [u8; 4],
    pub aux: u16,
    pub flags: u16,
}

impl Action {
    pub fn bind(
        frame: &DecisionFrame,
        seat: u8,
        action_index: u32,
        value: &ActionDescriptor,
    ) -> Self {
        Self {
            environment_id: frame.environment_id,
            episode_generation: frame.episode_generation,
            frame_id: frame.frame_id,
            seat,
            action_index,
            kind: value.kind,
            primary_tile_type: value.primary_tile_type,
            source_seat: value.source_seat,
            tile_count: value.tile_count,
            tiles: value.tiles,
            aux: value.aux,
            flags: value.flags,
        }
    }

    pub fn descriptor(&self) -> ActionDescriptor {
        ActionDescriptor {
            kind: self.kind,
            primary_tile_type: self.primary_tile_type,
            source_seat: self.source_seat,
            tile_count: self.tile_count,
            tiles: self.tiles,
            aux: self.aux,
            flags: self.flags,
        }
    }
}
