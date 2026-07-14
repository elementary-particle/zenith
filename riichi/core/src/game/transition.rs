use crate::game::{
    action::{ActionDescriptor, ActionKind, DecisionFrame, SeatDecision, ABSENT},
    phase::{EnvironmentLifecycle, HandPhase, MeldKind, RiichiState},
    rules::{
        hand::{draw_live, draw_replacement},
        legal, precedence,
        profile::RIICHILAB_MJSOUL,
        settlement::{ranking, Settlement},
    },
    state::{GameState, HanchanState, Meld, ProvisionalKan, RiverEntry},
};

/// Applies match metadata after a settled hand.
///
/// This is deliberately independent of scoring: settlement decides whether
/// the dealer continues, while this function owns round rotation, agari-yame,
/// West extension, the target score, and tobi termination.
pub fn advance_after_settlement(
    h: &mut HanchanState,
    dealer_continues: bool,
    was_draw: bool,
    is_midway_draw: bool,
) -> bool {
    let busted = RIICHILAB_MJSOUL.tobi_below_zero && h.scores.iter().any(|&score| score < 0);
    let scores = h.scores;
    let dealer_is_first = ranking(scores, h.initial_seats)[h.dealer as usize] == 1;
    let maximum_score = scores.into_iter().max().expect("four scores");
    let target_reached = maximum_score >= RIICHILAB_MJSOUL.return_points;
    let at_south_four = h.round_wind == crate::game::phase::Wind::South && h.dealer == 3;
    let in_extension = h.round_wind as u8 > crate::game::phase::Wind::South as u8;

    let agari_yame = dealer_continues
        && !is_midway_draw
        && dealer_is_first
        && scores[h.dealer as usize] >= RIICHILAB_MJSOUL.return_points
        && (at_south_four || in_extension);
    let rotation_ends_match = !dealer_continues
        && ((at_south_four && target_reached)
            || (in_extension
                && (target_reached
                    || (h.round_wind == RIICHILAB_MJSOUL.maximum_extension_wind
                        && h.dealer == 3))));
    if busted || agari_yame || rotation_ends_match {
        h.hand.phase = HandPhase::HanchanComplete;
        h.hand.decision_frame = None;
        return true;
    }

    h.honba = if dealer_continues || was_draw {
        h.honba.saturating_add(1)
    } else {
        0
    };
    if !dealer_continues {
        h.dealer = (h.dealer + 1) % 4;
        if h.dealer == 0 {
            h.hand_number = 0;
            h.round_wind = match h.round_wind {
                crate::game::phase::Wind::East => crate::game::phase::Wind::South,
                crate::game::phase::Wind::South => crate::game::phase::Wind::West,
                crate::game::phase::Wind::West | crate::game::phase::Wind::North => {
                    crate::game::phase::Wind::West
                }
            };
        } else {
            h.hand_number += 1;
        }
    }
    h.hand.phase = HandPhase::HandComplete;
    h.hand.decision_frame = None;
    false
}

/// Commits a pure settlement result before applying match progression.
pub fn apply_settlement_and_advance(
    h: &mut HanchanState,
    settlement: &Settlement,
    was_draw: bool,
    is_midway_draw: bool,
) -> bool {
    for (score, delta) in h.scores.iter_mut().zip(settlement.deltas) {
        *score = score.saturating_add(delta);
    }
    h.riichi_deposits = settlement.deposits_after;
    advance_after_settlement(h, settlement.dealer_continues, was_draw, is_midway_draw)
}

pub fn offer_self_turn(slot: &mut GameState) {
    let h = slot.hanchan.as_mut().expect("initialized");
    let seat = h.hand.current_seat;
    h.hand.phase = HandPhase::SelfTurnDecision;
    let ops = legal::self_turn_for_hanchan(h, seat);
    let frame_id = slot.next_frame_id;
    slot.next_frame_id += 1;
    h.hand.decision_frame = Some(DecisionFrame {
        environment_id: slot.environment_id,
        episode_generation: slot.episode_generation,
        frame_id,
        phase: HandPhase::SelfTurnDecision,
        eligible_mask: 1 << seat,
        decisions: vec![SeatDecision { seat, actions: ops }],
    });
    slot.lifecycle = EnvironmentLifecycle::Running;
}

pub fn draw_and_offer(slot: &mut GameState) -> bool {
    let h = slot.hanchan.as_mut().expect("initialized");
    let seat = h.hand.current_seat;
    let Some(tile) = draw_live(&mut h.hand.wall) else {
        h.hand.phase = HandPhase::Settlement;
        return false;
    };
    h.players[seat as usize].concealed_tiles.push(tile);
    h.players[seat as usize].concealed_tiles.sort_unstable();
    h.players[seat as usize].temporary_furiten = false;
    h.hand.current_draw = tile;
    h.hand.current_draw_is_replacement = false;
    offer_self_turn(slot);
    true
}

pub fn apply_self_turn(slot: &mut GameState, seat: u8, action: &ActionDescriptor) {
    let h = slot.hanchan.as_mut().expect("initialized");
    h.hand.decision_frame = None;
    match action.kind {
        ActionKind::Discard | ActionKind::RiichiDiscard => {
            let tile = action.tiles[0];
            let player = &mut h.players[seat as usize];
            let index = player
                .concealed_tiles
                .iter()
                .position(|&id| id == tile)
                .expect("validated action");
            player.concealed_tiles.remove(index);
            player.forbidden_discard_mask = 0;
            if action.kind == ActionKind::RiichiDiscard {
                player.riichi_state = RiichiState::Declared;
                player.ippatsu_eligible = false;
            } else if player.ippatsu_eligible {
                player.ippatsu_eligible = false;
            }
            let seq =
                slot.next_event_sequence + u64::from(action.kind == ActionKind::RiichiDiscard);
            player.river.push(RiverEntry {
                tile,
                sequence: seq,
                riichi_declaration: action.kind == ActionKind::RiichiDiscard,
                called: false,
                tsumogiri: tile == h.hand.current_draw,
            });
            h.hand.last_discard = Some((seat, tile));
            h.hand.phase = HandPhase::DiscardReactionFrame;
            let mut decisions = Vec::new();
            let mut mask = 0;
            for target in 0..4_u8 {
                if target != seat {
                    let actions = legal::reactions_for_hanchan(h, target, seat, tile);
                    mask |= 1 << target;
                    decisions.push(SeatDecision {
                        seat: target,
                        actions,
                    });
                }
            }
            let frame_id = slot.next_frame_id;
            slot.next_frame_id += 1;
            h.hand.decision_frame = Some(DecisionFrame {
                environment_id: slot.environment_id,
                episode_generation: slot.episode_generation,
                frame_id,
                phase: HandPhase::DiscardReactionFrame,
                eligible_mask: mask,
                decisions,
            });
        }
        ActionKind::Tsumo => {
            h.hand.phase = HandPhase::Settlement;
        }
        ActionKind::ClosedKan => offer_or_commit_closed_kan(slot, seat, action.clone()),
        ActionKind::AddedKan => offer_kan_rob(slot, seat, action.clone()),
        ActionKind::AbortiveDeclaration => {
            h.hand.phase = HandPhase::Settlement;
        }
        _ => unreachable!("self-turn action generation is authoritative"),
    }
}

pub fn apply_reactions(slot: &mut GameState, selections: &[(u8, ActionDescriptor)]) {
    let h = slot.hanchan.as_mut().expect("initialized");
    let reaction_phase = h.hand.phase;
    let offered_ron = h
        .hand
        .decision_frame
        .as_ref()
        .map(|frame| {
            frame
                .decisions
                .iter()
                .filter(|decision| {
                    decision
                        .actions
                        .iter()
                        .any(|action| action.kind == ActionKind::Ron)
                })
                .map(|decision| decision.seat)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    for &(seat, ref action) in selections {
        if action.kind == ActionKind::Pass && offered_ron.contains(&seat) {
            let player = &mut h.players[seat as usize];
            if player.riichi_state == RiichiState::Accepted {
                player.riichi_furiten = true;
            } else {
                player.temporary_furiten = true;
            }
        }
    }

    if reaction_phase == HandPhase::KanRobReactionFrame {
        let source = h
            .hand
            .provisional_kan
            .as_ref()
            .expect("kan-rob context")
            .seat;
        h.hand.decision_frame = None;
        match precedence::resolve(source, selections) {
            precedence::ReactionResolution::Ron(_) => h.hand.phase = HandPhase::Settlement,
            precedence::ReactionResolution::AllPass => commit_provisional_kan(slot),
            precedence::ReactionResolution::Call(..) => {
                unreachable!("kan-rob frames contain only pass and ron")
            }
        }
        return;
    }

    let (source, _) = h.hand.last_discard.expect("reaction context");
    h.hand.decision_frame = None;
    let resolution = precedence::resolve(source, selections);
    match &resolution {
        precedence::ReactionResolution::Ron(_) => {
            if h.players[source as usize].riichi_state == RiichiState::Declared {
                h.players[source as usize].riichi_state = RiichiState::None;
            }
        }
        _ => accept_pending_riichi(h, source),
    }
    match resolution {
        precedence::ReactionResolution::Ron(_) => h.hand.phase = HandPhase::Settlement,
        precedence::ReactionResolution::Call(seat, action) => match action.kind {
            ActionKind::Chi | ActionKind::Pon | ActionKind::OpenKan => {
                let called_tile = h.hand.last_discard.expect("call context").1;
                let source_river = &mut h.players[source as usize].river;
                source_river
                    .iter_mut()
                    .rev()
                    .find(|entry| entry.tile == called_tile && !entry.called)
                    .expect("discard remains in source river")
                    .called = true;

                {
                    let player = &mut h.players[seat as usize];
                    for &tile in action
                        .tiles
                        .iter()
                        .take(action.tile_count as usize)
                        .filter(|&&tile| tile != called_tile)
                    {
                        let index = player
                            .concealed_tiles
                            .iter()
                            .position(|&owned| owned == tile)
                            .expect("validated call owns every consumed tile");
                        player.concealed_tiles.remove(index);
                    }
                    player.melds.push(Meld {
                        kind: match action.kind {
                            ActionKind::Chi => MeldKind::Chi,
                            ActionKind::Pon => MeldKind::Pon,
                            ActionKind::OpenKan => MeldKind::OpenKan,
                            _ => unreachable!(),
                        },
                        tiles: action.tiles,
                        tile_count: action.tile_count,
                        called_tile,
                        from_seat: source,
                        created_sequence: slot.next_event_sequence,
                    });
                    set_kuikae_mask(player, &action, called_tile);
                }
                h.hand.current_seat = seat;
                h.hand.current_draw = ABSENT;
                h.hand.current_draw_is_replacement = false;
                for player in &mut h.players {
                    player.ippatsu_eligible = false;
                }

                if action.kind == ActionKind::OpenKan {
                    replacement_draw_and_offer(slot, seat);
                } else {
                    offer_self_turn(slot);
                }
            }
            _ => unreachable!(),
        },
        precedence::ReactionResolution::AllPass => {
            h.hand.current_seat = (source + 1) % 4;
            let _ = draw_and_offer(slot);
        }
    }
}

fn accept_pending_riichi(h: &mut HanchanState, seat: u8) {
    let player = &mut h.players[seat as usize];
    if player.riichi_state != RiichiState::Declared {
        return;
    }
    h.scores[seat as usize] -= 1_000;
    h.riichi_deposits += 1;
    player.riichi_state = RiichiState::Accepted;
    player.ippatsu_eligible = true;
}

fn offer_kan_rob(slot: &mut GameState, seat: u8, mut action: ActionDescriptor) {
    let h = slot.hanchan.as_mut().expect("initialized");
    action.source_seat = seat;
    let tile = action.tiles[0];
    let concealed_kan = action.kind == ActionKind::ClosedKan;
    h.hand.provisional_kan = Some(ProvisionalKan { seat, action });
    h.hand.phase = HandPhase::KanRobReactionFrame;
    let mut decisions = Vec::with_capacity(3);
    let mut eligible_mask = 0;
    for target in 0..4_u8 {
        if target == seat {
            continue;
        }
        eligible_mask |= 1 << target;
        decisions.push(SeatDecision {
            seat: target,
            actions: legal::kan_rob_reactions_for_hanchan(h, target, seat, tile, concealed_kan),
        });
    }
    let frame_id = slot.next_frame_id;
    slot.next_frame_id += 1;
    h.hand.decision_frame = Some(DecisionFrame {
        environment_id: slot.environment_id,
        episode_generation: slot.episode_generation,
        frame_id,
        phase: HandPhase::KanRobReactionFrame,
        eligible_mask,
        decisions,
    });
}

fn offer_or_commit_closed_kan(slot: &mut GameState, seat: u8, action: ActionDescriptor) {
    let tile = action.tiles[0];
    let has_kokushi_rob = slot
        .hanchan
        .as_ref()
        .expect("initialized")
        .players
        .iter()
        .filter(|player| player.seat != seat)
        .any(|player| {
            let h = slot.hanchan.as_ref().expect("initialized");
            legal::kan_rob_reactions_for_hanchan(h, player.seat, seat, tile, true)
                .iter()
                .any(|candidate| candidate.kind == ActionKind::Ron)
        });
    if has_kokushi_rob {
        offer_kan_rob(slot, seat, action);
    } else {
        commit_closed_kan(slot, seat, &action);
    }
}

fn commit_closed_kan(slot: &mut GameState, seat: u8, action: &ActionDescriptor) {
    let h = slot.hanchan.as_mut().expect("initialized");
    let player = &mut h.players[seat as usize];
    remove_owned_tiles(player, &action.tiles, action.tile_count);
    player.melds.push(Meld {
        kind: MeldKind::ClosedKan,
        tiles: action.tiles,
        tile_count: 4,
        called_tile: ABSENT,
        from_seat: ABSENT,
        created_sequence: slot.next_event_sequence,
    });
    for player in &mut h.players {
        player.ippatsu_eligible = false;
    }
    replacement_draw_and_offer(slot, seat);
}

fn commit_provisional_kan(slot: &mut GameState) {
    let provisional = slot
        .hanchan
        .as_mut()
        .expect("initialized")
        .hand
        .provisional_kan
        .take()
        .expect("provisional kan");
    if provisional.action.kind == ActionKind::ClosedKan {
        commit_closed_kan(slot, provisional.seat, &provisional.action);
        return;
    }
    let h = slot.hanchan.as_mut().expect("initialized");
    let player = &mut h.players[provisional.seat as usize];
    let tile = provisional.action.tiles[0];
    let index = player
        .concealed_tiles
        .iter()
        .position(|&owned| owned == tile)
        .expect("added tile remains concealed until commit");
    player.concealed_tiles.remove(index);
    let meld = &mut player.melds[provisional.action.aux as usize];
    meld.kind = MeldKind::AddedKan;
    meld.tiles[3] = tile;
    meld.tiles.sort_unstable();
    meld.tile_count = 4;
    for player in &mut h.players {
        player.ippatsu_eligible = false;
    }
    replacement_draw_and_offer(slot, provisional.seat);
}

fn replacement_draw_and_offer(slot: &mut GameState, seat: u8) {
    let h = slot.hanchan.as_mut().expect("initialized");
    h.hand.current_seat = seat;
    h.hand.phase = HandPhase::ReplacementTurn;
    if let Some(tile) = draw_replacement(&mut h.hand.wall) {
        let player = &mut h.players[seat as usize];
        player.concealed_tiles.push(tile);
        player.concealed_tiles.sort_unstable();
        player.temporary_furiten = false;
        h.hand.current_draw = tile;
        h.hand.current_draw_is_replacement = true;
        offer_self_turn(slot);
    } else {
        h.hand.phase = HandPhase::Settlement;
    }
}

fn remove_owned_tiles(player: &mut crate::game::state::PlayerState, tiles: &[u8; 4], count: u8) {
    for &tile in tiles.iter().take(count as usize) {
        let index = player
            .concealed_tiles
            .iter()
            .position(|&owned| owned == tile)
            .expect("validated action owns tile");
        player.concealed_tiles.remove(index);
    }
}

fn set_kuikae_mask(
    player: &mut crate::game::state::PlayerState,
    action: &ActionDescriptor,
    called_tile: u8,
) {
    player.forbidden_discard_mask = legal::kuikae_mask(action, called_tile);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::{
        rules::hand::wall_from_tiles,
        state::{GameState, RngState},
    };

    fn reset_slot() -> GameState {
        let mut slot = GameState::uninitialized(0);
        slot.reset(RngState {
            state: 1,
            stream: 3,
        });
        slot.take_events();
        slot.hanchan.as_mut().unwrap().hand.wall =
            wall_from_tiles(std::array::from_fn(|index| index as u8)).unwrap();
        slot
    }

    #[test]
    fn closed_kan_commits_and_draws_replacement() {
        let mut slot = reset_slot();
        {
            let h = slot.hanchan.as_mut().unwrap();
            h.players[0].concealed_tiles = vec![0, 1, 2, 3, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52];
            h.hand.current_seat = 0;
            h.hand.current_draw = 52;
        }
        offer_self_turn(&mut slot);
        let frame = slot
            .hanchan
            .as_ref()
            .unwrap()
            .hand
            .decision_frame
            .as_ref()
            .unwrap();
        let action = frame.decisions[0]
            .actions
            .iter()
            .find(|action| action.kind == ActionKind::ClosedKan)
            .unwrap()
            .clone();
        slot.apply_actions(vec![(0, action)]);

        let h = slot.hanchan.as_ref().unwrap();
        assert_eq!(h.players[0].melds[0].kind, MeldKind::ClosedKan);
        assert_eq!(h.hand.current_draw, 135);
        assert_eq!(h.hand.wall.dora_indicator_count, 2);
        assert_eq!(h.hand.phase, HandPhase::SelfTurnDecision);
    }

    #[test]
    fn added_kan_is_provisional_until_joint_rob_frame_passes() {
        let mut slot = reset_slot();
        {
            let h = slot.hanchan.as_mut().unwrap();
            h.players[0].concealed_tiles = vec![3, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52];
            h.players[0].melds.push(Meld {
                kind: MeldKind::Pon,
                tiles: [0, 1, 2, ABSENT],
                tile_count: 3,
                called_tile: 2,
                from_seat: 3,
                created_sequence: 0,
            });
            h.hand.current_seat = 0;
            h.hand.current_draw = 52;
        }
        offer_self_turn(&mut slot);
        let frame = slot
            .hanchan
            .as_ref()
            .unwrap()
            .hand
            .decision_frame
            .as_ref()
            .unwrap();
        let action = frame.decisions[0]
            .actions
            .iter()
            .find(|action| action.kind == ActionKind::AddedKan)
            .unwrap()
            .clone();
        slot.apply_actions(vec![(0, action)]);
        let rob_frame = slot
            .hanchan
            .as_ref()
            .unwrap()
            .hand
            .decision_frame
            .as_ref()
            .unwrap()
            .clone();
        assert_eq!(rob_frame.phase, HandPhase::KanRobReactionFrame);
        assert_eq!(
            slot.hanchan.as_ref().unwrap().players[0].melds[0].kind,
            MeldKind::Pon
        );

        let passes = rob_frame
            .decisions
            .iter()
            .map(|decision| (decision.seat, ActionDescriptor::pass()))
            .collect();
        slot.apply_actions(passes);
        let h = slot.hanchan.as_ref().unwrap();
        assert_eq!(h.players[0].melds[0].kind, MeldKind::AddedKan);
        assert_eq!(h.hand.current_draw, 135);
        assert!(h.hand.provisional_kan.is_none());
    }

    #[test]
    fn kokushi_may_rob_a_concealed_kan_without_committing_it() {
        let mut slot = reset_slot();
        {
            let h = slot.hanchan.as_mut().unwrap();
            h.players[0].concealed_tiles = vec![0, 1, 2, 3, 16, 20, 24, 28, 36, 40, 44, 48, 52, 56];
            h.players[1].concealed_tiles =
                vec![32, 33, 36, 68, 72, 104, 108, 112, 116, 120, 124, 128, 132];
            h.hand.current_seat = 0;
            h.hand.current_draw = 56;
        }
        offer_self_turn(&mut slot);
        let frame = slot
            .hanchan
            .as_ref()
            .unwrap()
            .hand
            .decision_frame
            .as_ref()
            .unwrap();
        let action = frame.decisions[0]
            .actions
            .iter()
            .find(|action| action.kind == ActionKind::ClosedKan)
            .unwrap()
            .clone();
        slot.apply_actions(vec![(0, action)]);

        let rob_frame = slot
            .hanchan
            .as_ref()
            .unwrap()
            .hand
            .decision_frame
            .as_ref()
            .unwrap()
            .clone();
        assert_eq!(rob_frame.phase, HandPhase::KanRobReactionFrame);
        let selections = rob_frame
            .decisions
            .iter()
            .map(|decision| {
                let action = decision
                    .actions
                    .iter()
                    .find(|action| decision.seat != 1 || action.kind == ActionKind::Ron)
                    .unwrap()
                    .clone();
                (decision.seat, action)
            })
            .collect();
        slot.apply_actions(selections);
        let h = slot.hanchan.as_ref().unwrap();
        assert_eq!(h.hand.phase, HandPhase::HanchanComplete);
        assert_eq!(slot.lifecycle, EnvironmentLifecycle::Complete);
        assert!(h.players[0].melds.is_empty());
        assert_eq!(h.hand.wall.dora_indicator_count, 1);
        let kinds = slot
            .take_events()
            .into_iter()
            .map(|event| event.kind)
            .collect::<Vec<_>>();
        assert!(kinds.contains(&crate::game::event::EventKind::Hora));
        assert!(kinds.contains(&crate::game::event::EventKind::EndKyoku));
        assert!(kinds.contains(&crate::game::event::EventKind::EndGame));
        assert!(!kinds.contains(&crate::game::event::EventKind::Ankan));
        assert!(!kinds.contains(&crate::game::event::EventKind::Dora));
    }

    #[test]
    fn riichi_is_accepted_only_after_the_discard_survives_reactions() {
        let mut slot = reset_slot();
        {
            let h = slot.hanchan.as_mut().unwrap();
            h.players[0].concealed_tiles =
                vec![0, 1, 2, 4, 8, 12, 36, 40, 44, 72, 76, 80, 108, 112];
            h.hand.current_seat = 0;
            h.hand.current_draw = 112;
            h.hand.wall.live_start = 53;
        }
        offer_self_turn(&mut slot);
        let frame = slot
            .hanchan
            .as_ref()
            .unwrap()
            .hand
            .decision_frame
            .as_ref()
            .unwrap();
        let action = frame.decisions[0]
            .actions
            .iter()
            .find(|action| action.kind == ActionKind::RiichiDiscard && action.tiles[0] == 112)
            .unwrap()
            .clone();
        slot.apply_actions(vec![(0, action)]);
        let h = slot.hanchan.as_ref().unwrap();
        assert_eq!(h.scores[0], 25_000);
        assert_eq!(h.riichi_deposits, 0);
        assert_eq!(h.players[0].riichi_state, RiichiState::Declared);
        assert!(!h.players[0].ippatsu_eligible);
        let declaration_events = slot.take_events();
        assert!(declaration_events
            .iter()
            .any(|event| event.kind == crate::game::event::EventKind::Reach));
        assert!(!declaration_events
            .iter()
            .any(|event| event.kind == crate::game::event::EventKind::ReachAccepted));

        let reaction = slot
            .hanchan
            .as_ref()
            .unwrap()
            .hand
            .decision_frame
            .as_ref()
            .unwrap()
            .clone();
        let passes = reaction
            .decisions
            .iter()
            .map(|decision| (decision.seat, ActionDescriptor::pass()))
            .collect();
        slot.apply_actions(passes);
        let h = slot.hanchan.as_ref().unwrap();
        assert_eq!(h.scores[0], 24_000);
        assert_eq!(h.riichi_deposits, 1);
        assert_eq!(h.players[0].riichi_state, RiichiState::Accepted);
        assert!(h.players[0].ippatsu_eligible);
        assert!(slot
            .take_events()
            .iter()
            .any(|event| event.kind == crate::game::event::EventKind::ReachAccepted));
    }

    #[test]
    fn passing_a_ron_sets_temporary_or_riichi_furiten() {
        for riichi in [false, true] {
            let mut slot = reset_slot();
            {
                let h = slot.hanchan.as_mut().unwrap();
                h.players[2].concealed_tiles = vec![0, 1, 2, 4, 8, 12, 36, 40, 44, 72, 76, 80, 108];
                if riichi {
                    h.players[2].riichi_state = RiichiState::Accepted;
                }
                h.players[0].river = vec![RiverEntry {
                    tile: 109,
                    sequence: 0,
                    riichi_declaration: false,
                    called: false,
                    tsumogiri: false,
                }];
                h.hand.last_discard = Some((0, 109));
                h.hand.phase = HandPhase::DiscardReactionFrame;
                h.hand.decision_frame = Some(DecisionFrame {
                    environment_id: 0,
                    episode_generation: slot.episode_generation,
                    frame_id: 77,
                    phase: HandPhase::DiscardReactionFrame,
                    eligible_mask: 0b1110,
                    decisions: (1..4)
                        .map(|seat| SeatDecision {
                            seat,
                            actions: legal::reactions(&h.players[seat as usize], seat, 0, 109),
                        })
                        .collect(),
                });
            }
            apply_reactions(
                &mut slot,
                &[
                    (1, ActionDescriptor::pass()),
                    (2, ActionDescriptor::pass()),
                    (3, ActionDescriptor::pass()),
                ],
            );
            let player = &slot.hanchan.as_ref().unwrap().players[2];
            assert_eq!(player.temporary_furiten, !riichi);
            assert_eq!(player.riichi_furiten, riichi);
        }
    }
}
