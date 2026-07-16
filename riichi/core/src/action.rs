use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

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
    /// Decisions with at least two semantically distinct choices. Seats with
    /// no choice never become part of a decision frame.
    pub decisions: Vec<SeatDecision>,
}

impl DecisionFrame {
    pub fn from_offered(
        environment_id: u32,
        episode_generation: u64,
        frame_id: u64,
        phase: HandPhase,
        offered: Vec<SeatDecision>,
    ) -> (Option<Self>, u64) {
        let mut decisions = Vec::new();
        let mut eligible_mask = 0;
        let mut automatic = 0;
        for decision in offered {
            assert!(decision.seat < 4 && !decision.actions.is_empty());
            let actions = semantic_representatives(decision.seat, decision.actions);
            if actions.len() == 1 {
                automatic += 1;
            } else {
                eligible_mask |= 1 << decision.seat;
                decisions.push(SeatDecision {
                    seat: decision.seat,
                    actions,
                });
            }
        }
        decisions.sort_by_key(|decision| decision.seat);
        let frame = (!decisions.is_empty()).then_some(Self {
            environment_id,
            episode_generation,
            frame_id,
            phase,
            eligible_mask,
            decisions,
        });
        (frame, automatic)
    }
}

/// Canonicalize physical-copy variants exactly as the model action encoder.
/// BTreeMap is used only for equality; representatives retain native order.
pub(crate) fn semantic_representatives(
    observer: u8,
    actions: Vec<ActionDescriptor>,
) -> Vec<ActionDescriptor> {
    let mut groups = BTreeMap::<SemanticActionKey, (usize, ActionDescriptor)>::new();
    for (index, action) in actions.into_iter().enumerate() {
        groups
            .entry(SemanticActionKey::new(observer, &action))
            .or_insert((index, action));
    }
    let mut representatives = groups.into_values().collect::<Vec<_>>();
    representatives.sort_by_key(|(index, _)| *index);
    representatives
        .into_iter()
        .map(|(_, action)| action)
        .collect()
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct SemanticActionKey {
    kind: ActionKind,
    primary_tile_type: u8,
    relative_source_seat: u8,
    primary_suit: u8,
    primary_rank: u8,
    primary_red: u8,
    tile_count: u8,
    semantic_tiles: [u8; 4],
    aux: u16,
    flags: u16,
}

impl SemanticActionKey {
    fn new(observer: u8, action: &ActionDescriptor) -> Self {
        let primary = action.tiles.first().copied().filter(|tile| *tile != ABSENT);
        let (primary_suit, primary_rank, primary_red) = primary.map_or((0, 0, 0), |tile| {
            let tile_type = tile / 4;
            let suit = if tile_type < 27 { tile_type / 9 + 1 } else { 4 };
            let rank = if tile_type < 27 {
                tile_type % 9 + 1
            } else {
                tile_type - 27 + 1
            };
            let red = u8::from(matches!(tile_type, 4 | 13 | 22) && tile % 4 == 0);
            (suit, rank, red)
        });
        let relative_source_seat = if action.source_seat == ABSENT {
            ABSENT
        } else {
            (action.source_seat + 4 - observer) % 4 + 1
        };
        let semantic_tiles = std::array::from_fn(|index| {
            let tile = action.tiles[index];
            if index >= usize::from(action.tile_count) || tile == ABSENT {
                ABSENT
            } else {
                let tile_type = tile / 4;
                let red = matches!(tile_type, 4 | 13 | 22) && tile % 4 == 0;
                tile_type * 4 + if red { 0 } else { 1 }
            }
        });
        Self {
            kind: action.kind,
            primary_tile_type: action.primary_tile_type,
            relative_source_seat,
            primary_suit,
            primary_rank,
            primary_red,
            tile_count: action.tile_count,
            semantic_tiles,
            aux: action.aux,
            flags: action.flags,
        }
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn discard(tile: u8) -> ActionDescriptor {
        ActionDescriptor::discard(tile)
    }

    #[test]
    fn canonical_groups_non_red_copies_and_keeps_lowest_native_representative() {
        let (frame, automatic) = DecisionFrame::from_offered(
            0,
            1,
            1,
            HandPhase::SelfTurnDecision,
            vec![SeatDecision {
                seat: 0,
                actions: vec![discard(3), discard(1), discard(4)],
            }],
        );
        let frame = frame.expect("two semantic choices");
        assert_eq!(automatic, 0);
        assert_eq!(frame.decisions[0].actions, vec![discard(3), discard(4)]);
    }

    #[test]
    fn red_and_non_red_fives_remain_distinct_semantic_choices() {
        let (frame, automatic) = DecisionFrame::from_offered(
            0,
            1,
            1,
            HandPhase::SelfTurnDecision,
            vec![SeatDecision {
                seat: 0,
                actions: vec![discard(16), discard(17)],
            }],
        );
        let frame = frame.expect("red and non-red are distinct");
        assert_eq!(automatic, 0);
        assert_eq!(frame.decisions[0].actions.len(), 2);
    }

    #[test]
    fn one_semantic_group_does_not_create_a_decision_frame() {
        let (frame, automatic) = DecisionFrame::from_offered(
            0,
            1,
            1,
            HandPhase::SelfTurnDecision,
            vec![SeatDecision {
                seat: 2,
                actions: vec![discard(0), discard(1), discard(2)],
            }],
        );
        assert!(frame.is_none());
        assert_eq!(automatic, 1);
        assert_eq!(
            semantic_representatives(2, vec![discard(0), discard(1)]),
            vec![discard(0)]
        );
    }
}
