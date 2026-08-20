//! Information-state hand-efficiency primitives built on exact shanten.

use super::shanten::{self, Shanten};
use crate::SHANTEN_UNAVAILABLE;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HandEfficiency {
    pub shanten: Shanten,
    pub improving_tile_mask: u64,
}

/// Evaluate exact shanten and the tile types whose addition improves it.
pub fn evaluate(counts: &[u8; 34], open_melds: u8) -> HandEfficiency {
    let value = shanten::calculate(counts, open_melds);
    let mut improving_tile_mask = 0_u64;
    if value.overall != SHANTEN_UNAVAILABLE {
        for tile_type in 0..34 {
            if counts[tile_type] >= 4 {
                continue;
            }
            let mut drawn = *counts;
            drawn[tile_type] += 1;
            if shanten::calculate(&drawn, open_melds).overall < value.overall {
                improving_tile_mask |= 1_u64 << tile_type;
            }
        }
    }
    HandEfficiency {
        shanten: value,
        improving_tile_mask,
    }
}

/// Count publicly plausible remaining copies of improving tile types.
pub fn ukeire(improving_tile_mask: u64, known_counts: &[u8; 34]) -> u16 {
    (0..34)
        .filter(|tile_type| improving_tile_mask & (1_u64 << tile_type) != 0)
        .map(|tile_type| u16::from(4_u8.saturating_sub(known_counts[tile_type])))
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn improving_mask_and_ukeire_use_only_available_copies() {
        let mut counts = [0_u8; 34];
        for tile in [0, 1, 2, 9, 10, 11, 18, 19, 20] {
            counts[tile] = 1;
        }
        counts[27] = 2;
        counts[28] = 2;
        let result = evaluate(&counts, 0);
        assert_eq!(result.shanten.overall, 0);
        assert_ne!(result.improving_tile_mask, 0);
        let mut known = counts;
        let before = ukeire(result.improving_tile_mask, &known);
        let improving = result.improving_tile_mask.trailing_zeros() as usize;
        known[improving] += 1;
        assert_eq!(ukeire(result.improving_tile_mask, &known), before - 1);
    }
}
