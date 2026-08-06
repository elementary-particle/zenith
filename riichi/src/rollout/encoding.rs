use riichi_core::{
    game::{
        event::EventRecord,
        rules::hand::tile_type_counts,
        state::{GameState, HanchanState},
    },
    ActionCandidate, EventKind,
};

const SEGMENT_MATCH: u8 = 1;
const SEGMENT_MATCH_SUMMARY: u8 = 2;
const SEGMENT_KYOKU: u8 = 3;
const SEGMENT_KYOKU_SUMMARY: u8 = 4;
const SEGMENT_ACTOR_QUERY: u8 = 5;

const KIND_EVENT: u8 = 1;
const KIND_SCORE: u8 = 2;
const KIND_COUNTER: u8 = 3;
const KIND_TILE_COUNT: u8 = 4;
const KIND_MELD: u8 = 5;
const KIND_RIVER: u8 = 6;
const KIND_MASKED: u8 = 7;
const KIND_QUERY: u8 = 8;

#[derive(Clone, Debug, PartialEq)]
pub struct TokenRow {
    pub categorical: [u8; 10],
    pub numeric: [f32; 8],
}

impl TokenRow {
    fn categorical(values: [u8; 10]) -> Self {
        Self {
            categorical: values,
            numeric: [0.0; 8],
        }
    }

    fn numeric(values: [u8; 10], field: u8, value: f32) -> Self {
        Self {
            categorical: values,
            numeric: numeric_features(field, value),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct RankBoundary {
    pub scores: [i32; 4],
    pub dealer: u8,
    pub round_wind: u8,
    pub hand_number: u8,
    pub honba: u16,
    pub riichi_deposits: u16,
}

impl RankBoundary {
    pub fn from_hanchan(game: &HanchanState) -> Self {
        Self {
            scores: game.scores,
            dealer: game.dealer,
            round_wind: game.round_wind as u8,
            hand_number: game.hand_number,
            honba: game.honba,
            riichi_deposits: game.riichi_deposits,
        }
    }

    pub fn features(self) -> [f32; 28] {
        let dealer = usize::from(self.dealer.min(3));
        let round_index = usize::from(
            (self
                .round_wind
                .saturating_mul(4)
                .saturating_add(self.hand_number))
            .min(15),
        );
        let mut result = [0.0; 28];
        for (offset, value) in result.iter_mut().take(4).enumerate() {
            *value = self.scores[(dealer + offset) % 4] as f32 / 24_000.0;
        }
        result[4 + dealer] = 1.0;
        result[8 + round_index] = 1.0;
        result[24] = (f32::from(self.honba) / 5.0).min(2.0);
        result[25] = (f32::from(self.riichi_deposits) / 5.0).min(2.0);
        result[26] = 7_usize.saturating_sub(round_index) as f32 / 8.0;
        result[27] = f32::from(round_index >= 8);
        result
    }
}

pub fn boundary_from_start_event(event: &EventRecord) -> Option<RankBoundary> {
    if event.kind != EventKind::StartKyoku || event.payload.len() < 20 {
        return None;
    }
    let scores = std::array::from_fn(|seat| {
        i32::from_le_bytes(
            event.payload[4 + 4 * seat..8 + 4 * seat]
                .try_into()
                .expect("four bytes"),
        )
    });
    Some(RankBoundary {
        scores,
        dealer: event.args[3].clamp(0, 3) as u8,
        round_wind: event.args[0].clamp(0, 3) as u8,
        hand_number: event.args[1].saturating_sub(1).clamp(0, 3) as u8,
        honba: event.args[2].clamp(0, i64::from(u16::MAX)) as u16,
        riichi_deposits: u16::from_le_bytes([event.payload[2], event.payload[3]]),
    })
}

pub fn encode_event(event: &EventRecord, observer: u8) -> Option<TokenRow> {
    if event.kind == EventKind::Tsumo {
        return None;
    }
    let kind = event.kind as u8;
    let visible = event.visibility_mask & (1 << observer) != 0;
    let args = if visible { event.args } else { [0; 4] };
    let (suit, rank, red) = if visible && matches!(kind, 4..=10 | 13) {
        physical_tile(args[0].clamp(0, 255) as u8)
    } else {
        (0, 0, 0)
    };
    let detail = if visible {
        compact_event_detail(kind, args, observer, event.target_seat)
    } else {
        0
    };
    let values = [
        SEGMENT_KYOKU,
        KIND_EVENT,
        kind,
        relative_seat(observer, event.actor_seat),
        suit,
        rank,
        red,
        0,
        detail,
        if visible { 1 } else { 2 },
    ];
    Some(if kind == EventKind::ReachAccepted as u8 && visible {
        TokenRow::numeric(values, 1, args[0] as f32)
    } else {
        TokenRow::categorical(values)
    })
}

fn compact_event_detail(kind: u8, args: [i64; 4], observer: u8, target: u8) -> u8 {
    match kind {
        2 => 1 + args[0].clamp(0, 3) as u8 * 4 + args[1].saturating_sub(1).clamp(0, 3) as u8,
        4 => 1 + u8::from(args[1] != 0),
        5 => {
            let types = args
                .iter()
                .copied()
                .filter(|tile| (0..136).contains(tile))
                .map(|tile| tile / 4)
                .collect::<Vec<_>>();
            let offset = types
                .iter()
                .min()
                .map_or(0, |minimum| (args[0] / 4 - minimum).clamp(0, 2) as u8);
            1 + offset * 2 + u8::from(args.into_iter().any(is_red_i64))
        }
        6..=9 => 1 + u8::from(args.into_iter().any(is_red_i64)),
        13 => relative_seat(observer, target),
        _ => 0,
    }
}

fn is_red_i64(tile: i64) -> bool {
    (0..136).contains(&tile) && matches!(tile / 4, 4 | 13 | 22) && tile % 4 == 0
}

pub fn encode_observation(
    state: &GameState,
    observer: u8,
    history: &[TokenRow],
    boundary: RankBoundary,
) -> Option<(Vec<TokenRow>, usize, [f32; 28], u8)> {
    let game = state.hanchan.as_ref()?;
    let frame = game.hand.decision.as_ref()?;
    let space = frame
        .action_spaces
        .iter()
        .find(|space| space.seat == observer)?;
    let player = &game.players[usize::from(observer)];
    let mut rows = Vec::with_capacity(64 + history.len());

    for (seat, score) in game.scores.into_iter().enumerate() {
        rows.push(TokenRow::numeric(
            [
                SEGMENT_MATCH,
                KIND_SCORE,
                1,
                relative_seat(observer, seat as u8),
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            1,
            score as f32,
        ));
    }
    for (field, value) in [
        (1, game.round_wind as u16),
        (2, u16::from(game.hand_number)),
        (3, game.honba),
        (4, game.riichi_deposits),
    ] {
        rows.push(TokenRow::numeric(
            [SEGMENT_MATCH, KIND_COUNTER, field, 0, 0, 0, 0, 0, 0, 0],
            2,
            f32::from(value),
        ));
    }
    rows.push(TokenRow::categorical([
        SEGMENT_MATCH,
        KIND_COUNTER,
        6,
        relative_seat(observer, game.dealer),
        0,
        0,
        0,
        0,
        0,
        0,
    ]));
    rows.push(TokenRow::categorical([
        SEGMENT_MATCH,
        KIND_COUNTER,
        7,
        0,
        0,
        0,
        0,
        (observer + 4 - game.dealer) % 4 + 1,
        0,
        0,
    ]));
    rows.push(query(SEGMENT_MATCH_SUMMARY, 2));
    rows.extend_from_slice(history);

    rows.push(TokenRow::numeric(
        [SEGMENT_KYOKU, KIND_COUNTER, 5, 0, 0, 0, 0, 0, 0, 0],
        2,
        f32::from(
            game.hand
                .wall
                .live_end
                .saturating_sub(game.hand.wall.live_start),
        ),
    ));
    for &tile in game
        .hand
        .wall
        .revealed_dora_indicators
        .iter()
        .take(usize::from(game.hand.wall.dora_indicator_count))
    {
        let (suit, rank, red) = physical_tile(tile);
        rows.push(TokenRow::categorical([
            SEGMENT_KYOKU,
            KIND_TILE_COUNT,
            3,
            0,
            suit,
            rank,
            red,
            0,
            0,
            1,
        ]));
    }
    let flags = decision_flags(player) as u8;
    rows.push(TokenRow::categorical([
        SEGMENT_KYOKU,
        KIND_COUNTER,
        8,
        1,
        0,
        0,
        0,
        0,
        flags,
        0,
    ]));
    for (tile_type, count) in tile_type_counts(&player.concealed_tiles)
        .into_iter()
        .enumerate()
    {
        if count == 0 {
            continue;
        }
        let (suit, rank, red) = tile_type_factors(tile_type as u8, 0);
        rows.push(TokenRow::categorical([
            SEGMENT_KYOKU,
            KIND_TILE_COUNT,
            1,
            1,
            suit,
            rank,
            red,
            count,
            0,
            1,
        ]));
    }
    for (seat, opponent) in game.players.iter().enumerate() {
        let public_flags = decision_flags(opponent) as u8 & 0b111;
        rows.push(TokenRow::categorical([
            SEGMENT_KYOKU,
            KIND_COUNTER,
            9,
            relative_seat(observer, seat as u8),
            0,
            0,
            0,
            0,
            public_flags,
            1,
        ]));
    }
    for player in &game.players {
        for river in &player.river {
            let (suit, rank, red) = physical_tile(river.tile);
            let river_flags = u8::from(river.riichi_declaration)
                | (u8::from(river.called) << 1)
                | (u8::from(river.tsumogiri) << 2);
            rows.push(TokenRow::categorical([
                SEGMENT_KYOKU,
                KIND_RIVER,
                1,
                relative_seat(observer, player.seat),
                suit,
                rank,
                red,
                1,
                river_flags,
                1,
            ]));
        }
        for meld in &player.melds {
            for &tile in meld.tiles.iter().take(usize::from(meld.tile_count)) {
                let (suit, rank, red) = physical_tile(tile);
                rows.push(TokenRow::categorical([
                    SEGMENT_KYOKU,
                    KIND_MELD,
                    meld.kind as u8,
                    relative_seat(observer, player.seat),
                    suit,
                    rank,
                    red,
                    1,
                    0,
                    1,
                ]));
            }
        }
    }
    for relative in 2..=4 {
        rows.push(TokenRow::categorical([
            SEGMENT_KYOKU,
            KIND_MASKED,
            1,
            relative,
            0,
            0,
            0,
            0,
            0,
            2,
        ]));
    }
    rows.push(query(SEGMENT_KYOKU_SUMMARY, 3));
    let actor_query = rows.len();
    let mut actor = query(SEGMENT_ACTOR_QUERY, 1);
    actor.categorical[8] = game.hand.phase as u8;
    actor.categorical[9] = u8::from(!space.candidates.is_empty());
    rows.push(actor);
    Some((
        rows,
        actor_query,
        boundary.features(),
        (observer + 4 - game.dealer) % 4,
    ))
}

pub fn encode_action(action: &ActionCandidate, observer: u8) -> [u8; 15] {
    let first = action.tiles.first().copied().unwrap_or(255);
    let (suit, rank, red) = physical_tile(first);
    let mut semantic = [255; 4];
    for (index, &tile) in action
        .tiles
        .iter()
        .take(usize::from(action.tile_count))
        .enumerate()
    {
        if tile != 255 {
            let tile_type = tile / 4;
            let red = matches!(tile_type, 4 | 13 | 22) && tile % 4 == 0;
            semantic[index] = tile_type * 4 + if red { 0 } else { 1 };
        }
    }
    [
        action.kind as u8,
        action.primary_tile_type,
        if action.source_seat == 255 {
            0
        } else {
            relative_seat(observer, action.source_seat)
        },
        suit,
        rank,
        red,
        action.tile_count,
        semantic[0],
        semantic[1],
        semantic[2],
        semantic[3],
        action.aux as u8,
        (action.aux >> 8) as u8,
        action.flags as u8,
        (action.flags >> 8) as u8,
    ]
}

pub fn conservative_action(state: &GameState, seat: u8, actions: &[ActionCandidate]) -> usize {
    if let Some(index) = actions
        .iter()
        .position(|action| matches!(action.kind as u8, 8 | 9))
    {
        return index;
    }
    let discards = actions
        .iter()
        .enumerate()
        .filter(|(_, action)| matches!(action.kind as u8, 1 | 2))
        .collect::<Vec<_>>();
    if !discards.is_empty() {
        let safe = state.hanchan.as_ref().and_then(|game| {
            let declared = game
                .players
                .iter()
                .filter(|player| player.seat != seat && decision_flags(player) & 0b11 != 0)
                .collect::<Vec<_>>();
            if declared.is_empty() {
                return None;
            }
            let mut intersection = [true; 34];
            for player in declared {
                let mut local = [false; 34];
                for tile in &player.river {
                    local[usize::from(tile.tile / 4)] = true;
                }
                for index in 0..34 {
                    intersection[index] &= local[index];
                }
            }
            Some(intersection)
        });
        return discards
            .iter()
            .filter(|(_, action)| {
                safe.as_ref().is_none_or(|types| {
                    types[usize::from(action.tiles[0] / 4)]
                        || !discards
                            .iter()
                            .any(|(_, candidate)| types[usize::from(candidate.tiles[0] / 4)])
                })
            })
            .min_by_key(|(index, action)| {
                (
                    u8::from(action.kind as u8 == 2),
                    action.tiles[0] / 4,
                    *index,
                )
            })
            .map_or(0, |(index, _)| *index);
    }
    actions
        .iter()
        .position(|action| action.kind as u8 == 0)
        .unwrap_or(0)
}

fn query(segment: u8, field: u8) -> TokenRow {
    TokenRow::categorical([segment, KIND_QUERY, field, 1, 0, 0, 0, 0, 0, 0])
}

fn numeric_features(field: u8, value: f32) -> [f32; 8] {
    if field == 0 {
        return [0.0; 8];
    }
    let periods = if field == 1 {
        [100.0, 1_000.0, 10_000.0, 100_000.0]
    } else {
        [2.0, 8.0, 32.0, 128.0]
    };
    let mut result = [0.0; 8];
    for (index, period) in periods.into_iter().enumerate() {
        // The Python oracle evaluates ``math.sin``/``math.cos`` in f64 and
        // casts the completed feature to NumPy f32. Keep that order here so
        // large exact score multiples do not acquire f32 range-reduction
        // noise before the trigonometric operation.
        let angle = 2.0 * std::f64::consts::PI * f64::from(value) / period;
        result[2 * index] = angle.sin() as f32;
        result[2 * index + 1] = angle.cos() as f32;
    }
    result
}

fn physical_tile(tile: u8) -> (u8, u8, u8) {
    if tile == 255 {
        return (0, 0, 0);
    }
    let tile_type = tile / 4;
    tile_type_factors(
        tile_type,
        u8::from(matches!(tile_type, 4 | 13 | 22) && tile % 4 == 0),
    )
}

fn tile_type_factors(tile_type: u8, red: u8) -> (u8, u8, u8) {
    if tile_type < 27 {
        (tile_type / 9 + 1, tile_type % 9 + 1, red)
    } else {
        (4, tile_type - 26, red)
    }
}

fn relative_seat(observer: u8, seat: u8) -> u8 {
    if seat == 255 {
        0
    } else {
        (seat + 4 - observer) % 4 + 1
    }
}

fn decision_flags(player: &riichi_core::game::state::PlayerState) -> u32 {
    use riichi_core::game::phase::RiichiState;
    u32::from(player.riichi_state == RiichiState::Declared)
        | (u32::from(player.riichi_state == RiichiState::Accepted) << 1)
        | (u32::from(player.ippatsu_eligible) << 2)
        | (u32::from(player.permanent_furiten) << 3)
        | (u32::from(player.temporary_furiten) << 4)
        | (u32::from(player.riichi_furiten) << 5)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rank_features_have_expected_rotated_scores() {
        let features = RankBoundary {
            scores: [12_000, 24_000, 36_000, 48_000],
            dealer: 2,
            round_wind: 1,
            hand_number: 3,
            honba: 2,
            riichi_deposits: 1,
        }
        .features();
        assert_eq!(&features[..4], &[1.5, 2.0, 0.5, 1.0]);
        assert_eq!(features[6], 1.0);
        assert_eq!(features[15], 1.0);
    }

    #[test]
    fn action_metadata_is_split_losslessly() {
        let mut action = ActionCandidate::discard(16);
        action.aux = 0x1234;
        action.flags = 0xabcd;
        let row = encode_action(&action, 0);
        assert_eq!(&row[11..], &[0x34, 0x12, 0xcd, 0xab]);
    }
}
