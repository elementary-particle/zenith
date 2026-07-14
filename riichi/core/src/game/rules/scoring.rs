use crate::evaluator::{
    hand_evaluator::HandEvaluator,
    types::{
        Conditions, Meld as ReferenceMeld, MeldType as ReferenceMeldType, Wind as ReferenceWind,
    },
};
use crate::game::{
    phase::{MeldKind, Wind},
    state::Meld,
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ScoreValue {
    pub han: u8,
    pub fu: u16,
    pub yakuman: u8,
    pub base_points: u32,
    pub limit: Limit,
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Limit {
    None = 0,
    Mangan = 1,
    Haneman = 2,
    Baiman = 3,
    Sanbaiman = 4,
    Yakuman = 5,
}

/// Rules context that cannot be inferred from the winning tiles alone.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WinningContext {
    pub tsumo: bool,
    pub riichi: bool,
    pub double_riichi: bool,
    pub ippatsu: bool,
    pub haitei: bool,
    pub houtei: bool,
    pub rinshan: bool,
    pub chankan: bool,
    pub first_turn_tsumo: bool,
    pub seat_wind: Wind,
    pub round_wind: Wind,
}

impl Default for WinningContext {
    fn default() -> Self {
        Self {
            tsumo: false,
            riichi: false,
            double_riichi: false,
            ippatsu: false,
            haitei: false,
            houtei: false,
            rinshan: false,
            chankan: false,
            first_turn_tsumo: false,
            seat_wind: Wind::East,
            round_wind: Wind::East,
        }
    }
}

/// Maximum-value interpretation of a physical winning hand under the pinned
/// RiichiLab/Mahjong Soul rules profile.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HandEvaluation {
    pub is_win: bool,
    pub has_win_shape: bool,
    pub han: u8,
    pub fu: u16,
    pub yakuman: u8,
    pub yaku_ids: Vec<u8>,
    pub value: ScoreValue,
}

/// Evaluates a physical hand through Zenith's private, conformance-pinned
/// evaluator. Internal evaluator types and allocation stay out of state
/// transitions; only Zenith-owned stable values cross this boundary.
pub fn evaluate_hand(
    concealed_tiles: &[u8],
    melds: &[Meld],
    win_tile: u8,
    dora_indicators: &[u8],
    ura_indicators: &[u8],
    context: &WinningContext,
) -> HandEvaluation {
    let reference_melds = melds
        .iter()
        .map(|meld| {
            let meld_type = match meld.kind {
                MeldKind::Chi => ReferenceMeldType::Chi,
                MeldKind::Pon => ReferenceMeldType::Pon,
                MeldKind::OpenKan => ReferenceMeldType::Daiminkan,
                MeldKind::ClosedKan => ReferenceMeldType::Ankan,
                MeldKind::AddedKan => ReferenceMeldType::Kakan,
            };
            let opened = meld.kind != MeldKind::ClosedKan;
            ReferenceMeld::new(
                meld_type,
                meld.tiles[..meld.tile_count as usize].to_vec(),
                opened,
                if meld.from_seat == u8::MAX {
                    -1
                } else {
                    meld.from_seat as i8
                },
                (meld.called_tile != u8::MAX).then_some(meld.called_tile),
            )
        })
        .collect();
    let evaluator = HandEvaluator::new(concealed_tiles.to_vec(), reference_melds);
    let conditions = Conditions {
        tsumo: context.tsumo,
        riichi: context.riichi,
        double_riichi: context.double_riichi,
        ippatsu: context.ippatsu,
        haitei: context.haitei,
        houtei: context.houtei,
        rinshan: context.rinshan,
        player_wind: reference_wind(context.seat_wind),
        round_wind: reference_wind(context.round_wind),
        chankan: context.chankan,
        tsumo_first_turn: context.first_turn_tsumo,
        riichi_sticks: 0,
        honba: 0,
        kita_count: 0,
        is_sanma: false,
        num_players: 4,
    };
    let result = evaluator.calc(
        win_tile,
        dora_indicators.to_vec(),
        ura_indicators.to_vec(),
        Some(conditions),
    );
    let han = u8::try_from(result.han).expect("four-player han fits u8");
    let fu = u16::try_from(result.fu).expect("four-player fu fits u16");
    let yakuman = if result.yakuman { han / 13 } else { 0 };
    HandEvaluation {
        is_win: result.is_win,
        has_win_shape: result.has_win_shape,
        han,
        fu,
        yakuman,
        yaku_ids: result
            .yaku
            .into_iter()
            .map(|id| u8::try_from(id).expect("pinned yaku IDs fit u8"))
            .collect(),
        value: score_value(han, fu, yakuman),
    }
}

pub fn score_value(han: u8, fu: u16, yakuman: u8) -> ScoreValue {
    if yakuman > 0 {
        return ScoreValue {
            han,
            fu,
            yakuman,
            base_points: 8000 * u32::from(yakuman),
            limit: Limit::Yakuman,
        };
    }
    let (base_points, limit) = match han {
        13.. => (8000, Limit::Yakuman),
        11..=12 => (6000, Limit::Sanbaiman),
        8..=10 => (4000, Limit::Baiman),
        6..=7 => (3000, Limit::Haneman),
        5 => (2000, Limit::Mangan),
        _ => {
            let raw = u32::from(fu) * (1_u32 << (han + 2));
            if raw >= 2000 {
                (2000, Limit::Mangan)
            } else {
                (raw, Limit::None)
            }
        }
    };
    ScoreValue {
        han,
        fu,
        yakuman,
        base_points,
        limit,
    }
}

pub fn round_100(value: u32) -> i32 {
    value.div_ceil(100).saturating_mul(100) as i32
}

fn reference_wind(wind: Wind) -> ReferenceWind {
    match wind {
        Wind::East => ReferenceWind::East,
        Wind::South => ReferenceWind::South,
        Wind::West => ReferenceWind::West,
        Wind::North => ReferenceWind::North,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn official_limits() {
        assert_eq!(score_value(5, 30, 0).base_points, 2000);
        assert_eq!(score_value(6, 30, 0).base_points, 3000);
        assert_eq!(score_value(13, 30, 0).base_points, 8000);
    }
}
