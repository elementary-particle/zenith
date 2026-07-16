use crate::game::state::{PlayerState, RngState, WallState};
use std::collections::BTreeSet;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum GroupKind {
    Sequence,
    Triplet,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct Group {
    pub kind: GroupKind,
    pub tile_type: u8,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct StandardDecomposition {
    pub pair: u8,
    pub groups: Vec<Group>,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum HandShape {
    Standard(StandardDecomposition),
    SevenPairs,
    ThirteenOrphans { thirteen_sided: bool },
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum WaitKind {
    Ryanmen,
    Kanchan,
    Penchan,
    Shanpon,
    Tanki,
    SevenPairs,
    KokushiSingle,
    KokushiThirteen,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct WinningCandidate {
    pub shape: HandShape,
    pub wait: WaitKind,
}

pub fn shuffled_wall(rng: &mut RngState) -> [u8; 136] {
    let mut wall = [0_u8; 136];
    for (index, tile) in wall.iter_mut().enumerate() {
        *tile = index as u8;
    }
    for i in (1..wall.len()).rev() {
        let j = rng.bounded((i + 1) as u32) as usize;
        wall.swap(i, j);
    }
    wall
}

pub fn initialize_wall(rng: &mut RngState) -> WallState {
    wall_from_tiles(shuffled_wall(rng)).expect("shuffle preserves every physical tile")
}

/// Builds a wall from an explicit physical-tile draw order.
///
/// Physical IDs are the integers `0..136`; four consecutive IDs share one
/// tile type. Keeping this constructor strict makes fixed-wall tests useful as
/// a rules oracle instead of allowing duplicate or missing tiles into state.
pub fn wall_from_tiles(tiles: [u8; 136]) -> Option<WallState> {
    let mut seen = [false; 136];
    for tile in tiles {
        let entry = seen.get_mut(tile as usize)?;
        if *entry {
            return None;
        }
        *entry = true;
    }
    let live_wall_counts = tile_type_counts(&tiles[..122]);
    Some(WallState {
        tiles,
        live_wall_counts,
        live_start: 0,
        live_end: 122,
        rinshan_index: 135,
        dora_indicator_count: 1,
        revealed_dora_indicators: [tiles[130], 255, 255, 255, 255],
        ura_indicators: [tiles[131], tiles[129], tiles[127], tiles[125], tiles[123]],
    })
}

pub fn deal(wall: &mut WallState, players: &mut [PlayerState; 4]) {
    // Japanese mahjong deals three packets of four tiles, then one tile to
    // each seat. The dealer's fourteenth tile is the ordinary first draw.
    for _ in 0..3 {
        for player in players.iter_mut() {
            for _ in 0..4 {
                player
                    .concealed_tiles
                    .push(draw_live(wall).expect("fresh wall"));
            }
        }
    }
    for player in players.iter_mut() {
        player
            .concealed_tiles
            .push(draw_live(wall).expect("fresh wall"));
    }
    for player in players {
        player.concealed_tiles.sort_unstable();
    }
}

pub fn draw_live(wall: &mut WallState) -> Option<u8> {
    if wall.live_start >= wall.live_end {
        return None;
    }
    let tile = wall.tiles[wall.live_start as usize];
    let count = &mut wall.live_wall_counts[usize::from(tile / 4)];
    *count = count
        .checked_sub(1)
        .expect("live wall counts track the live draw cursor");
    wall.live_start += 1;
    Some(tile)
}

pub fn draw_replacement(wall: &mut WallState) -> Option<u8> {
    if wall.dora_indicator_count >= 5 || wall.rinshan_index < 132 {
        return None;
    }
    let tile = wall.tiles[wall.rinshan_index as usize];
    wall.rinshan_index -= 1;
    // A tile moves from the live wall into the dead wall as rinshan is drawn,
    // keeping the dead wall at fourteen physical tiles.
    if wall.live_start < wall.live_end {
        let shifted_index = usize::from(wall.live_end - 1);
        let shifted_tile = wall.tiles[shifted_index];
        let count = &mut wall.live_wall_counts[usize::from(shifted_tile / 4)];
        *count = count
            .checked_sub(1)
            .expect("live wall counts track the dead-wall boundary");
    }
    wall.live_end = wall.live_end.saturating_sub(1);
    let indicator_index = 130 - usize::from(wall.dora_indicator_count) * 2;
    wall.revealed_dora_indicators[wall.dora_indicator_count as usize] = wall.tiles[indicator_index];
    wall.dora_indicator_count += 1;
    Some(tile)
}

pub fn tile_type_counts(tiles: &[u8]) -> [u8; 34] {
    let mut counts = [0; 34];
    for &tile in tiles {
        if tile < 136 {
            counts[(tile / 4) as usize] += 1;
        }
    }
    counts
}

/// Recomputes the maintained live-wall counts for invariant validation and
/// restoration of snapshots that omit this derived cache.
pub(crate) fn recompute_live_wall_counts(wall: &WallState) -> [u8; 34] {
    let start = usize::from(wall.live_start);
    let end = usize::from(wall.live_end);
    tile_type_counts(wall.tiles.get(start..end).unwrap_or_default())
}

pub fn standard_complete(counts: &mut [u8; 34], melds_needed: u8) -> bool {
    for pair in 0..34 {
        if counts[pair] >= 2 {
            counts[pair] -= 2;
            if groups_complete(counts, melds_needed) {
                counts[pair] += 2;
                return true;
            }
            counts[pair] += 2;
        }
    }
    false
}

pub fn hand_complete(counts: &[u8; 34], open_melds: u8) -> bool {
    let mut standard = *counts;
    if standard_complete(&mut standard, 4 - open_melds) {
        return true;
    }
    open_melds == 0 && (seven_pairs_complete(counts) || kokushi_complete(counts))
}

pub fn seven_pairs_complete(counts: &[u8; 34]) -> bool {
    counts.iter().filter(|&&count| count == 2).count() == 7
        && counts.iter().map(|&count| u16::from(count)).sum::<u16>() == 14
}

pub fn kokushi_complete(counts: &[u8; 34]) -> bool {
    const ORPHANS: [usize; 13] = [0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33];
    ORPHANS.iter().all(|&index| counts[index] > 0)
        && ORPHANS.iter().any(|&index| counts[index] >= 2)
        && counts.iter().map(|&count| u16::from(count)).sum::<u16>() == 14
}

pub fn wait_types(counts: &[u8; 34], open_melds: u8) -> [bool; 34] {
    let mut waits = [false; 34];
    for tile_type in 0..34 {
        if counts[tile_type] < 4 {
            let mut candidate = *counts;
            candidate[tile_type] += 1;
            waits[tile_type] = hand_complete(&candidate, open_melds);
        }
    }
    waits
}

pub fn standard_decompositions(counts: &[u8; 34], open_melds: u8) -> Vec<StandardDecomposition> {
    if open_melds > 4 {
        return Vec::new();
    }
    let groups_needed = 4 - open_melds;
    let mut result = BTreeSet::new();
    for pair in 0..34 {
        if counts[pair] < 2 {
            continue;
        }
        let mut remaining = *counts;
        remaining[pair] -= 2;
        enumerate_groups(
            &mut remaining,
            groups_needed,
            &mut Vec::with_capacity(groups_needed as usize),
            pair as u8,
            &mut result,
        );
    }
    result.into_iter().collect()
}

pub fn winning_candidates(
    counts: &[u8; 34],
    open_melds: u8,
    winning_tile_type: u8,
) -> Vec<WinningCandidate> {
    let mut candidates = BTreeSet::new();
    for decomposition in standard_decompositions(counts, open_melds) {
        if decomposition.pair == winning_tile_type {
            candidates.insert(WinningCandidate {
                shape: HandShape::Standard(decomposition.clone()),
                wait: WaitKind::Tanki,
            });
        }
        for group in &decomposition.groups {
            let wait = match group.kind {
                GroupKind::Triplet if group.tile_type == winning_tile_type => {
                    Some(WaitKind::Shanpon)
                }
                GroupKind::Sequence
                    if (group.tile_type..group.tile_type + 3).contains(&winning_tile_type) =>
                {
                    let offset = winning_tile_type - group.tile_type;
                    Some(if offset == 1 {
                        WaitKind::Kanchan
                    } else if (group.tile_type % 9 == 0 && offset == 2)
                        || (group.tile_type % 9 == 6 && offset == 0)
                    {
                        WaitKind::Penchan
                    } else {
                        WaitKind::Ryanmen
                    })
                }
                _ => None,
            };
            if let Some(wait) = wait {
                candidates.insert(WinningCandidate {
                    shape: HandShape::Standard(decomposition.clone()),
                    wait,
                });
            }
        }
    }
    if open_melds == 0 && seven_pairs_complete(counts) {
        candidates.insert(WinningCandidate {
            shape: HandShape::SevenPairs,
            wait: WaitKind::SevenPairs,
        });
    }
    if open_melds == 0 && kokushi_complete(counts) {
        let thirteen_sided = ORPHAN_TYPES.iter().all(|&index| {
            counts[index] == 1 + usize::from(index == winning_tile_type as usize) as u8
        });
        candidates.insert(WinningCandidate {
            shape: HandShape::ThirteenOrphans { thirteen_sided },
            wait: if thirteen_sided {
                WaitKind::KokushiThirteen
            } else {
                WaitKind::KokushiSingle
            },
        });
    }
    candidates.into_iter().collect()
}

const ORPHAN_TYPES: [usize; 13] = [0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33];

fn enumerate_groups(
    counts: &mut [u8; 34],
    groups_needed: u8,
    groups: &mut Vec<Group>,
    pair: u8,
    result: &mut BTreeSet<StandardDecomposition>,
) {
    let Some(tile_type) = counts.iter().position(|&count| count > 0) else {
        if groups.len() == groups_needed as usize {
            let mut canonical = groups.clone();
            canonical.sort();
            result.insert(StandardDecomposition {
                pair,
                groups: canonical,
            });
        }
        return;
    };
    if groups.len() >= groups_needed as usize {
        return;
    }
    if counts[tile_type] >= 3 {
        counts[tile_type] -= 3;
        groups.push(Group {
            kind: GroupKind::Triplet,
            tile_type: tile_type as u8,
        });
        enumerate_groups(counts, groups_needed, groups, pair, result);
        groups.pop();
        counts[tile_type] += 3;
    }
    if tile_type < 27
        && tile_type % 9 <= 6
        && counts[tile_type + 1] > 0
        && counts[tile_type + 2] > 0
    {
        counts[tile_type] -= 1;
        counts[tile_type + 1] -= 1;
        counts[tile_type + 2] -= 1;
        groups.push(Group {
            kind: GroupKind::Sequence,
            tile_type: tile_type as u8,
        });
        enumerate_groups(counts, groups_needed, groups, pair, result);
        groups.pop();
        counts[tile_type] += 1;
        counts[tile_type + 1] += 1;
        counts[tile_type + 2] += 1;
    }
}

fn groups_complete(counts: &mut [u8; 34], remaining: u8) -> bool {
    if remaining == 0 {
        return counts.iter().all(|&c| c == 0);
    }
    let Some(i) = counts.iter().position(|&c| c > 0) else {
        return false;
    };
    if counts[i] >= 3 {
        counts[i] -= 3;
        if groups_complete(counts, remaining - 1) {
            counts[i] += 3;
            return true;
        }
        counts[i] += 3;
    }
    if i < 27 && i % 9 <= 6 && counts[i + 1] > 0 && counts[i + 2] > 0 {
        counts[i] -= 1;
        counts[i + 1] -= 1;
        counts[i + 2] -= 1;
        if groups_complete(counts, remaining - 1) {
            counts[i] += 1;
            counts[i + 1] += 1;
            counts[i + 2] += 1;
            return true;
        }
        counts[i] += 1;
        counts[i + 1] += 1;
        counts[i + 2] += 1;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_wall_rejects_non_physical_permutations() {
        let mut duplicate = std::array::from_fn(|index| index as u8);
        duplicate[135] = 0;
        assert!(wall_from_tiles(duplicate).is_none());

        let mut out_of_range = std::array::from_fn(|index| index as u8);
        out_of_range[135] = 255;
        assert!(wall_from_tiles(out_of_range).is_none());
    }

    #[test]
    fn replacement_draws_shift_live_wall_and_reveal_indicators() {
        let tiles = std::array::from_fn(|index| index as u8);
        let mut wall = wall_from_tiles(tiles).unwrap();
        let expected_counts = [(30, 1), (30, 0), (29, 3), (29, 2)];
        for ((draw, indicator), (tile_type, expected_count)) in
            [(135, 128), (134, 126), (133, 124), (132, 122)]
                .into_iter()
                .zip(expected_counts)
        {
            assert_eq!(draw_replacement(&mut wall), Some(draw));
            assert_eq!(
                wall.revealed_dora_indicators[wall.dora_indicator_count as usize - 1],
                indicator
            );
            assert_eq!(wall.live_wall_counts[tile_type], expected_count);
        }
        assert_eq!(wall.live_end, 118);
        assert_eq!(wall.live_wall_counts.iter().sum::<u8>(), 118);
        assert_eq!(draw_replacement(&mut wall), None);
    }

    #[test]
    fn exhausted_live_wall_with_reversed_bounds_has_zero_counts() {
        let tiles = std::array::from_fn(|index| index as u8);
        let mut wall = wall_from_tiles(tiles).unwrap();
        assert_eq!(wall.live_wall_counts.iter().sum::<u8>(), 122);
        while draw_live(&mut wall).is_some() {}
        assert_eq!(wall.live_wall_counts, [0; 34]);

        assert_eq!(draw_replacement(&mut wall), Some(135));
        assert!(wall.live_start > wall.live_end);

        assert_eq!(wall.live_wall_counts, [0; 34]);
    }

    #[test]
    fn winning_candidates_retain_competing_standard_and_special_shapes() {
        let mut counts = [0; 34];
        for count in counts.iter_mut().take(7) {
            *count = 2;
        }
        let candidates = winning_candidates(&counts, 0, 6);
        assert!(candidates
            .iter()
            .any(|candidate| candidate.shape == HandShape::SevenPairs));
        assert!(candidates
            .iter()
            .any(|candidate| matches!(candidate.shape, HandShape::Standard(_))));
        assert!(candidates.windows(2).all(|pair| pair[0] < pair[1]));
    }

    #[test]
    fn wait_candidates_distinguish_edges_closed_waits_and_kokushi() {
        let mut edge = [0; 34];
        for tile_type in [0, 1, 2, 9, 10, 11, 18, 19, 20, 27, 27, 27, 28, 28] {
            edge[tile_type] += 1;
        }
        assert!(winning_candidates(&edge, 0, 2)
            .iter()
            .any(|candidate| candidate.wait == WaitKind::Penchan));

        let mut kokushi = [0; 34];
        for tile_type in ORPHAN_TYPES {
            kokushi[tile_type] = 1;
        }
        kokushi[0] = 2;
        assert!(winning_candidates(&kokushi, 0, 0)
            .iter()
            .any(|candidate| candidate.wait == WaitKind::KokushiThirteen));
    }
}
