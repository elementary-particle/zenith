use crate::game::{
    action::{ActionCandidate, ActionKind, ABSENT},
    phase::{MeldKind, RiichiState, Wind},
    rules::scoring::{evaluate_hand, WinningContext},
    state::{HanchanState, HandState, PlayerState},
};

use super::{
    hand::{hand_complete, kokushi_complete, tile_type_counts, wait_types},
    shanten,
};

pub fn self_turn(player: &PlayerState, hand: &HandState, score: i32) -> Vec<ActionCandidate> {
    let mut actions = Vec::new();
    let riichi_locked = player.riichi_state == RiichiState::Accepted;
    for &tile in &player.concealed_tiles {
        if (!riichi_locked || tile == hand.current_draw)
            && player.forbidden_discard_mask & (1_u64 << (tile / 4)) == 0
        {
            actions.push(ActionCandidate::discard(tile));
        }
    }

    let counts = tile_type_counts(&player.concealed_tiles);
    if player.concealed_tiles.len() % 3 == 2 && hand_complete(&counts, player.melds.len() as u8) {
        actions.push(ActionCandidate {
            kind: ActionKind::Tsumo,
            primary_tile_type: hand.current_draw / 4,
            source_seat: ABSENT,
            tile_count: 1,
            tiles: [hand.current_draw, ABSENT, ABSENT, ABSENT],
            aux: 0,
            flags: 0,
        });
    }

    add_closed_kans(&mut actions, player, hand, &counts, riichi_locked);
    if !riichi_locked {
        add_added_kans(&mut actions, player);
        add_riichi_discards(&mut actions, player, hand, score);
    }
    actions.sort();
    actions.dedup();
    actions
}

/// Context-aware self-turn actions used by the live engine.
///
/// The shape-only helper remains useful for decomposition tests, but a win is
/// a legal action only when the current round context supplies at least one
/// yaku. This function is pure and never mutates the hanchan.
pub fn self_turn_for_hanchan(h: &HanchanState, seat: u8) -> Vec<ActionCandidate> {
    let player = &h.players[seat as usize];
    let mut actions = self_turn(player, &h.hand, h.scores[seat as usize]);
    actions.retain(|action| {
        action.kind != ActionKind::Tsumo
            || contextual_win_is_legal(h, seat, action.tiles[0], true, false)
    });
    if player.river.is_empty()
        && h.players.iter().all(|player| player.melds.is_empty())
        && distinct_terminal_or_honor_types(&player.concealed_tiles) >= 9
    {
        actions.push(ActionCandidate {
            kind: ActionKind::AbortiveDeclaration,
            primary_tile_type: ABSENT,
            source_seat: ABSENT,
            tile_count: 0,
            tiles: [ABSENT; 4],
            aux: 1, // kyuushu kyuuhai
            flags: 0,
        });
        actions.sort();
    }
    actions
}

fn distinct_terminal_or_honor_types(tiles: &[u8]) -> usize {
    let mut present = [false; 34];
    for &tile in tiles {
        let tile_type = usize::from(tile / 4);
        if tile_type >= 27
            || (tile_type < 27 && tile_type % 9 == 0)
            || (tile_type < 27 && tile_type % 9 == 8)
        {
            present[tile_type] = true;
        }
    }
    present.into_iter().filter(|present| *present).count()
}

pub fn reactions(player: &PlayerState, seat: u8, source: u8, tile: u8) -> Vec<ActionCandidate> {
    let mut result = vec![ActionCandidate::pass()];
    let counts = tile_type_counts(&player.concealed_tiles);
    let tile_type = (tile / 4) as usize;

    if player.riichi_state == RiichiState::None {
        for owned in combinations_of_type(&player.concealed_tiles, tile_type, 2) {
            result.push(call(ActionKind::Pon, source, tile, &owned));
        }
        if let Some(owned) = combinations_of_type(&player.concealed_tiles, tile_type, 3).pop() {
            result.push(call(ActionKind::OpenKan, source, tile, &owned));
        }
        if seat == (source + 1) % 4 && tile_type < 27 {
            add_chi(&mut result, player, source, tile, tile_type);
        }
    }

    let waits = wait_types(&counts, player.melds.len() as u8);
    let permanent_furiten = player
        .river
        .iter()
        .any(|entry| waits[(entry.tile / 4) as usize]);
    let mut winning = counts;
    winning[tile_type] += 1;
    if !permanent_furiten
        && !player.permanent_furiten
        && !player.temporary_furiten
        && !player.riichi_furiten
        && hand_complete(&winning, player.melds.len() as u8)
    {
        result.push(ron(source, tile));
    }
    result.retain(|action| {
        !matches!(action.kind, ActionKind::Chi | ActionKind::Pon)
            || call_leaves_legal_discard(player, action, tile)
    });
    result.sort();
    result
}

fn call_leaves_legal_discard(
    player: &PlayerState,
    action: &ActionCandidate,
    called_tile: u8,
) -> bool {
    let mut remaining = player.concealed_tiles.clone();
    for &tile in action
        .tiles
        .iter()
        .take(action.tile_count as usize)
        .filter(|&&tile| tile != called_tile)
    {
        let Some(index) = remaining.iter().position(|&owned| owned == tile) else {
            return false;
        };
        remaining.remove(index);
    }
    let forbidden = kuikae_mask(action, called_tile);
    remaining
        .iter()
        .any(|tile| forbidden & (1_u64 << (tile / 4)) == 0)
}

pub(crate) fn kuikae_mask(action: &ActionCandidate, called_tile: u8) -> u64 {
    let called_type = called_tile / 4;
    let mut mask = 1_u64 << called_type;
    if action.kind == ActionKind::Chi {
        let mut types = action.tiles[..action.tile_count as usize]
            .iter()
            .map(|tile| tile / 4)
            .collect::<Vec<_>>();
        types.sort_unstable();
        // Kuikae forbids discarding the tile on the *other side* of the
        // original ryanmen shape.  If x was called into x,x+1,x+2, that tile
        // is x+3; if x+2 was called, it is x-1.  The previous +/-1 formula
        // rejected harmless adjacent discards and could cross suit boundaries
        // (for example, calling 1p could forbid 9m).
        if called_type == types[0] && called_type % 9 <= 5 {
            mask |= 1_u64 << (called_type + 3);
        } else if called_type == types[2] && called_type % 9 >= 3 {
            mask |= 1_u64 << (called_type - 3);
        }
    }
    mask
}

/// Context-aware discard reactions used by the live engine.
pub fn reactions_for_hanchan(
    h: &HanchanState,
    seat: u8,
    source: u8,
    tile: u8,
) -> Vec<ActionCandidate> {
    let mut actions = reactions(&h.players[seat as usize], seat, source, tile);
    actions.retain(|action| {
        action.kind != ActionKind::Ron || contextual_win_is_legal(h, seat, tile, false, false)
    });
    actions
}

pub fn kan_rob_reactions(
    player: &PlayerState,
    source: u8,
    tile: u8,
    concealed_kan: bool,
) -> Vec<ActionCandidate> {
    let mut result = vec![ActionCandidate::pass()];
    let mut counts = tile_type_counts(&player.concealed_tiles);
    let tile_type = (tile / 4) as usize;
    counts[tile_type] += 1;
    let may_win = if concealed_kan {
        kokushi_complete(&counts)
    } else {
        hand_complete(&counts, player.melds.len() as u8)
    };
    if may_win && !player.permanent_furiten && !player.temporary_furiten && !player.riichi_furiten {
        result.push(ron(source, tile));
    }
    result
}

pub fn kan_rob_reactions_for_hanchan(
    h: &HanchanState,
    seat: u8,
    source: u8,
    tile: u8,
    concealed_kan: bool,
) -> Vec<ActionCandidate> {
    let mut actions = kan_rob_reactions(&h.players[seat as usize], source, tile, concealed_kan);
    actions.retain(|action| {
        action.kind != ActionKind::Ron || contextual_win_is_legal(h, seat, tile, false, true)
    });
    actions
}

fn contextual_win_is_legal(
    h: &HanchanState,
    seat: u8,
    win_tile: u8,
    tsumo: bool,
    chankan: bool,
) -> bool {
    let player = &h.players[seat as usize];
    let mut concealed = player.concealed_tiles.clone();
    if tsumo {
        let Some(index) = concealed.iter().rposition(|&tile| tile == win_tile) else {
            return false;
        };
        concealed.remove(index);
    }
    let context = winning_context(h, seat, tsumo, chankan);
    let indicator_count = h.hand.wall.dora_indicator_count as usize;
    let ura_count = if context.riichi { indicator_count } else { 0 };
    evaluate_hand(
        &concealed,
        &player.melds,
        win_tile,
        &h.hand.wall.revealed_dora_indicators[..indicator_count],
        &h.hand.wall.ura_indicators[..ura_count],
        &context,
    )
    .is_win
}

pub(crate) fn winning_context(
    h: &HanchanState,
    seat: u8,
    tsumo: bool,
    chankan: bool,
) -> WinningContext {
    let player = &h.players[seat as usize];
    let wall_exhausted = h.hand.wall.live_start >= h.hand.wall.live_end;
    WinningContext {
        tsumo,
        riichi: player.riichi_state == RiichiState::Accepted,
        double_riichi: is_double_riichi(h, seat),
        ippatsu: player.ippatsu_eligible,
        haitei: tsumo && !h.hand.current_draw_is_replacement && wall_exhausted,
        houtei: !tsumo && !chankan && wall_exhausted,
        rinshan: tsumo && h.hand.current_draw_is_replacement,
        chankan,
        first_turn_tsumo: tsumo
            && player.river.is_empty()
            && h.players.iter().all(|player| player.melds.is_empty()),
        seat_wind: seat_wind(seat, h.dealer),
        round_wind: h.round_wind,
    }
}

/// Double riichi is derivable from permanent hand history: the declaration
/// must be the player's first discard and no call or kan may precede it.
fn is_double_riichi(h: &HanchanState, seat: u8) -> bool {
    let player = &h.players[seat as usize];
    if player.riichi_state != RiichiState::Accepted {
        return false;
    }
    let Some(declaration) = player
        .river
        .first()
        .filter(|entry| entry.riichi_declaration)
    else {
        return false;
    };
    !h.players
        .iter()
        .flat_map(|player| &player.melds)
        .any(|meld| meld.created_sequence < declaration.sequence)
}

fn seat_wind(seat: u8, dealer: u8) -> Wind {
    match (seat + 4 - dealer) % 4 {
        0 => Wind::East,
        1 => Wind::South,
        2 => Wind::West,
        3 => Wind::North,
        _ => unreachable!(),
    }
}

fn add_riichi_discards(
    actions: &mut Vec<ActionCandidate>,
    player: &PlayerState,
    hand: &HandState,
    score: i32,
) {
    let closed = player
        .melds
        .iter()
        .all(|meld| meld.kind == MeldKind::ClosedKan);
    if !closed
        || player.riichi_state != RiichiState::None
        || score < 1_000
        || hand.wall.live_end.saturating_sub(hand.wall.live_start) < 4
    {
        return;
    }
    for &tile in &player.concealed_tiles {
        if player.forbidden_discard_mask & (1_u64 << (tile / 4)) != 0 {
            continue;
        }
        let mut remaining = player.concealed_tiles.clone();
        remaining.remove(remaining.iter().position(|&owned| owned == tile).unwrap());
        if shanten::calculate(&tile_type_counts(&remaining), player.melds.len() as u8).overall == 0
        {
            let mut action = ActionCandidate::discard(tile);
            action.kind = ActionKind::RiichiDiscard;
            actions.push(action);
        }
    }
}

fn add_closed_kans(
    actions: &mut Vec<ActionCandidate>,
    player: &PlayerState,
    hand: &HandState,
    counts: &[u8; 34],
    riichi_locked: bool,
) {
    let waits_before = if riichi_locked {
        let mut before = player.concealed_tiles.clone();
        if let Some(index) = before.iter().position(|&tile| tile == hand.current_draw) {
            before.remove(index);
        }
        Some(wait_types(
            &tile_type_counts(&before),
            player.melds.len() as u8,
        ))
    } else {
        None
    };
    for (tile_type, &count) in counts.iter().enumerate() {
        if count != 4 {
            continue;
        }
        if let Some(waits_before) = waits_before {
            if hand.current_draw as usize / 4 != tile_type {
                continue;
            }
            let mut after = *counts;
            after[tile_type] -= 4;
            if wait_types(&after, player.melds.len() as u8 + 1) != waits_before {
                continue;
            }
        }
        let owned = combinations_of_type(&player.concealed_tiles, tile_type, 4)
            .pop()
            .expect("count checked");
        actions.push(kan(ActionKind::ClosedKan, tile_type as u8, &owned, 0));
    }
}

fn add_added_kans(actions: &mut Vec<ActionCandidate>, player: &PlayerState) {
    for (meld_index, meld) in player.melds.iter().enumerate() {
        if meld.kind != MeldKind::Pon {
            continue;
        }
        let tile_type = (meld.tiles[0] / 4) as usize;
        for &tile in &player.concealed_tiles {
            if tile as usize / 4 == tile_type {
                actions.push(kan(
                    ActionKind::AddedKan,
                    tile_type as u8,
                    &[tile],
                    meld_index as u16,
                ));
            }
        }
    }
}

fn call(kind: ActionKind, source: u8, called: u8, owned: &[u8]) -> ActionCandidate {
    let mut tiles = [ABSENT; 4];
    tiles[0] = called;
    tiles[1..=owned.len()].copy_from_slice(owned);
    tiles[..=owned.len()].sort_unstable();
    ActionCandidate {
        kind,
        primary_tile_type: called / 4,
        source_seat: source,
        tile_count: (owned.len() + 1) as u8,
        tiles,
        aux: 0,
        flags: 0,
    }
}

fn kan(kind: ActionKind, tile_type: u8, owned: &[u8], aux: u16) -> ActionCandidate {
    let mut tiles = [ABSENT; 4];
    tiles[..owned.len()].copy_from_slice(owned);
    tiles[..owned.len()].sort_unstable();
    ActionCandidate {
        kind,
        primary_tile_type: tile_type,
        source_seat: ABSENT,
        tile_count: owned.len() as u8,
        tiles,
        aux,
        flags: 0,
    }
}

fn ron(source: u8, tile: u8) -> ActionCandidate {
    ActionCandidate {
        kind: ActionKind::Ron,
        primary_tile_type: tile / 4,
        source_seat: source,
        tile_count: 1,
        tiles: [tile, ABSENT, ABSENT, ABSENT],
        aux: 0,
        flags: 0,
    }
}

fn combinations_of_type(tiles: &[u8], tile_type: usize, count: usize) -> Vec<Vec<u8>> {
    let candidates = tiles
        .iter()
        .copied()
        .filter(|&tile| tile as usize / 4 == tile_type)
        .collect::<Vec<_>>();
    let mut result = Vec::new();
    combinations(&candidates, count, 0, &mut Vec::new(), &mut result);
    result
}

fn combinations(
    candidates: &[u8],
    count: usize,
    start: usize,
    current: &mut Vec<u8>,
    result: &mut Vec<Vec<u8>>,
) {
    if current.len() == count {
        result.push(current.clone());
        return;
    }
    for index in start..candidates.len() {
        current.push(candidates[index]);
        combinations(candidates, count, index + 1, current, result);
        current.pop();
    }
}

fn add_chi(
    result: &mut Vec<ActionCandidate>,
    player: &PlayerState,
    source: u8,
    tile: u8,
    tile_type: usize,
) {
    let suit_start = tile_type / 9 * 9;
    for sequence_start in suit_start..=suit_start + 6 {
        if !(sequence_start..sequence_start + 3).contains(&tile_type) {
            continue;
        }
        let needed = (sequence_start..sequence_start + 3)
            .filter(|&candidate| candidate != tile_type)
            .collect::<Vec<_>>();
        let left = combinations_of_type(&player.concealed_tiles, needed[0], 1);
        let right = combinations_of_type(&player.concealed_tiles, needed[1], 1);
        for a in &left {
            for b in &right {
                result.push(call(ActionKind::Chi, source, tile, &[a[0], b[0]]));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::{
        phase::HandPhase,
        rules::hand::wall_from_tiles,
        state::{Meld, RiverEntry},
    };

    fn hand(current_draw: u8) -> HandState {
        HandState {
            phase: HandPhase::SelfTurnDecision,
            wall: wall_from_tiles(std::array::from_fn(|index| index as u8)).unwrap(),
            current_seat: 0,
            current_draw,
            current_draw_is_replacement: false,
            last_discard: None,
            provisional_kan: None,
            decision: None,
        }
    }

    #[test]
    fn pon_enumerates_physical_choices_and_open_kan_is_unique() {
        let mut player = PlayerState::new(1);
        player.concealed_tiles = vec![0, 1, 2, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56];
        let actions = reactions(&player, 1, 0, 3);
        assert_eq!(
            actions
                .iter()
                .filter(|action| action.kind == ActionKind::Pon)
                .count(),
            3
        );
        assert_eq!(
            actions
                .iter()
                .filter(|action| action.kind == ActionKind::OpenKan)
                .count(),
            1
        );
    }

    #[test]
    fn own_river_wait_causes_permanent_furiten() {
        let mut player = PlayerState::new(1);
        player.concealed_tiles = vec![0, 1, 2, 4, 8, 12, 36, 40, 44, 72, 76, 80, 108];
        let ron = reactions(&player, 1, 0, 109);
        assert!(ron.iter().any(|action| action.kind == ActionKind::Ron));
        player.river.push(RiverEntry {
            tile: 111,
            sequence: 0,
            riichi_declaration: false,
            called: false,
            tsumogiri: false,
        });
        let furiten = reactions(&player, 1, 0, 109);
        assert!(!furiten.iter().any(|action| action.kind == ActionKind::Ron));
    }

    #[test]
    fn self_turn_includes_special_win_closed_and_added_kan() {
        let mut seven_pairs = PlayerState::new(0);
        seven_pairs.concealed_tiles = vec![0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24, 25];
        assert!(self_turn(&seven_pairs, &hand(25), 25_000)
            .iter()
            .any(|action| action.kind == ActionKind::Tsumo));

        let mut kans = PlayerState::new(0);
        kans.concealed_tiles = vec![0, 1, 2, 3, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52];
        assert!(self_turn(&kans, &hand(52), 25_000)
            .iter()
            .any(|action| action.kind == ActionKind::ClosedKan));
        kans.concealed_tiles = vec![3, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52];
        kans.melds.push(Meld {
            kind: MeldKind::Pon,
            tiles: [0, 1, 2, ABSENT],
            tile_count: 3,
            called_tile: 2,
            from_seat: 3,
            created_sequence: 0,
        });
        assert!(self_turn(&kans, &hand(52), 25_000)
            .iter()
            .any(|action| action.kind == ActionKind::AddedKan));
    }

    #[test]
    fn shared_winning_context_distinguishes_rinshan_from_haitei() {
        let mut replacement = hand(52);
        replacement.current_draw_is_replacement = true;
        replacement.wall.live_start = replacement.wall.live_end;
        let h = HanchanState {
            round_wind: Wind::East,
            hand_number: 1,
            dealer: 0,
            honba: 0,
            riichi_deposits: 0,
            completed_kyoku: 0,
            scores: [25_000; 4],
            initial_seats: [0, 1, 2, 3],
            players: std::array::from_fn(|seat| PlayerState::new(seat as u8)),
            hand: replacement,
        };

        let context = winning_context(&h, 0, true, false);
        assert!(context.rinshan);
        assert!(!context.haitei);
    }

    #[test]
    fn first_turn_context_supports_chiihou_and_rejects_kan_interruptions() {
        let mut h = HanchanState {
            round_wind: Wind::East,
            hand_number: 0,
            dealer: 0,
            honba: 0,
            riichi_deposits: 0,
            completed_kyoku: 0,
            scores: [25_000; 4],
            initial_seats: [0, 1, 2, 3],
            players: std::array::from_fn(|seat| PlayerState::new(seat as u8)),
            hand: hand(52),
        };
        h.players[0].river.push(RiverEntry {
            tile: 0,
            sequence: 1,
            riichi_declaration: false,
            called: false,
            tsumogiri: true,
        });
        assert!(winning_context(&h, 1, true, false).first_turn_tsumo);

        h.players[0].melds.push(Meld {
            kind: MeldKind::ClosedKan,
            tiles: [4, 5, 6, 7],
            tile_count: 4,
            called_tile: ABSENT,
            from_seat: ABSENT,
            created_sequence: 2,
        });
        assert!(!winning_context(&h, 1, true, false).first_turn_tsumo);
    }

    #[test]
    fn double_riichi_is_derived_from_first_discard_and_prior_melds() {
        let mut h = HanchanState {
            round_wind: Wind::East,
            hand_number: 0,
            dealer: 0,
            honba: 0,
            riichi_deposits: 0,
            completed_kyoku: 0,
            scores: [25_000; 4],
            initial_seats: [0, 1, 2, 3],
            players: std::array::from_fn(|seat| PlayerState::new(seat as u8)),
            hand: hand(52),
        };
        h.players[2].riichi_state = RiichiState::Accepted;
        h.players[2].river.push(RiverEntry {
            tile: 12,
            sequence: 10,
            riichi_declaration: true,
            called: false,
            tsumogiri: false,
        });
        assert!(winning_context(&h, 2, false, false).double_riichi);

        h.players[0].melds.push(Meld {
            kind: MeldKind::Pon,
            tiles: [0, 1, 2, ABSENT],
            tile_count: 3,
            called_tile: 2,
            from_seat: 3,
            created_sequence: 9,
        });
        assert!(!winning_context(&h, 2, false, false).double_riichi);
    }

    #[test]
    fn chi_that_leaves_only_kuikae_tiles_is_not_offered() {
        let mut player = PlayerState::new(0);
        // Call 2m with 3m-4m; only the opposite-side 5m remains.
        player.concealed_tiles = vec![8, 12, 16];

        let actions = reactions(&player, 0, 3, 4);

        assert!(!actions.iter().any(|action| {
            action.kind == ActionKind::Chi && action.tiles[..3].to_vec() == vec![4, 8, 12]
        }));
    }

    #[test]
    fn kuikae_uses_the_opposite_ryanmen_tile_without_crossing_suits() {
        let low_call = call(ActionKind::Chi, 3, 16, &[20, 24]);
        let low_mask = kuikae_mask(&low_call, 16);
        assert_ne!(low_mask & (1 << 4), 0); // called 5m
        assert_ne!(low_mask & (1 << 7), 0); // opposite-side 8m
        assert_eq!(low_mask & (1 << 3), 0); // adjacent 4m remains legal

        let one_pin_call = call(ActionKind::Chi, 3, 36, &[40, 44]);
        let one_pin_mask = kuikae_mask(&one_pin_call, 36);
        assert_ne!(one_pin_mask & (1 << 9), 0); // called 1p
        assert_ne!(one_pin_mask & (1 << 12), 0); // opposite-side 4p
        assert_eq!(one_pin_mask & (1 << 8), 0); // never crosses into 9m
    }
}
