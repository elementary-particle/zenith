use super::{
    rules::{hand::tile_type_counts, shanten::calculate},
    state::GameState,
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlayerObservation {
    pub environment_id: u32,
    pub episode_generation: u64,
    pub frame_id: u64,
    pub seat: u8,
    pub scores: [i32; 4],
    pub concealed_counts: [u8; 34],
    pub shanten: [i8; 4],
    pub improving_actual_counts: [u8; 34],
}

pub fn project(slot: &GameState, seat: u8) -> Option<PlayerObservation> {
    let h = slot.hanchan.as_ref()?;
    let player = &h.players[seat as usize];
    let counts = tile_type_counts(&player.concealed_tiles);
    let s = calculate(&counts, player.melds.len() as u8);
    let actual_remaining = live_wall_counts(slot);
    Some(PlayerObservation {
        environment_id: slot.environment_id,
        episode_generation: slot.episode_generation,
        frame_id: h.hand.decision.as_ref().map_or(0, |f| f.frame_id),
        seat,
        scores: h.scores,
        concealed_counts: counts,
        shanten: [s.overall, s.standard, s.seven_pairs, s.thirteen_orphans],
        improving_actual_counts: improving_tiles_from_remaining(
            &counts,
            player.melds.len() as u8,
            &actual_remaining,
        ),
    })
}

fn live_wall_counts(slot: &GameState) -> [u8; 34] {
    let Some(h) = &slot.hanchan else {
        return [0; 34];
    };
    h.hand.wall.live_wall_counts
}

fn improving_tiles_from_remaining(
    counts: &[u8; 34],
    open_melds: u8,
    remaining: &[u8; 34],
) -> [u8; 34] {
    let baseline = calculate(counts, open_melds).overall;
    let mut result = [0; 34];
    for index in 0..34 {
        if remaining[index] == 0 || counts[index] >= 4 {
            continue;
        }
        let mut next = *counts;
        next[index] += 1;
        if calculate(&next, open_melds).overall < baseline {
            result[index] = remaining[index];
        }
    }
    result
}

#[cfg(test)]
fn visible_counts(slot: &GameState, _owner: u8) -> [u8; 34] {
    let mut result = [0; 34];
    let Some(h) = &slot.hanchan else {
        return result;
    };
    for player in &h.players {
        for river in &player.river {
            result[(river.tile / 4) as usize] += 1;
        }
        for meld in &player.melds {
            for tile in meld.tiles.iter().take(meld.tile_count as usize) {
                result[(*tile / 4) as usize] += 1;
            }
        }
    }
    for tile in h
        .hand
        .wall
        .revealed_dora_indicators
        .iter()
        .take(h.hand.wall.dora_indicator_count as usize)
    {
        result[(*tile / 4) as usize] += 1;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{batch::engine::BatchEngine, game::state::RiverEntry};

    #[test]
    fn own_public_tiles_reduce_visible_remaining_copies() {
        let mut engine = BatchEngine::new(1, 7, 1).unwrap();
        engine.reset(&[0]).unwrap();
        let slot = &mut engine.slots[0];
        slot.hanchan.as_mut().unwrap().players[0]
            .river
            .push(RiverEntry {
                tile: 0,
                sequence: 99,
                riichi_declaration: false,
                called: false,
                tsumogiri: false,
            });
        assert_eq!(visible_counts(slot, 0)[0], 1);
    }

    #[test]
    fn exhausted_live_wall_with_reversed_bounds_has_no_remaining_tiles() {
        let mut engine = BatchEngine::new(1, 11, 1).unwrap();
        engine.reset(&[0]).unwrap();
        let slot = &mut engine.slots[0];
        let wall = &mut slot.hanchan.as_mut().unwrap().hand.wall;
        while super::super::rules::hand::draw_live(wall).is_some() {}
        assert!(super::super::rules::hand::draw_replacement(wall).is_some());

        assert_eq!(live_wall_counts(slot), [0; 34]);
        assert!(project(slot, 0).is_some());
    }
}
