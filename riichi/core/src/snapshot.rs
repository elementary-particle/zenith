mod body;

use crate::{
    error::CoreError,
    game::{
        action::{semantic_representatives, ActionCandidate, ActionKind, ABSENT},
        phase::{EnvironmentLifecycle, HandPhase, MeldKind},
        rules::hand::recompute_live_wall_counts,
        rules::legal,
        rules::profile::{self, RNG_PROFILE_ID, SNAPSHOT_SCHEMA_VERSION},
        state::{GameState, HanchanState},
    },
};

pub const SNAPSHOT_FORMAT_VERSION: u16 = 1;
pub const SNAPSHOT_HEADER_BYTES: usize = 64;
const MAX_BODY_BYTES: usize = 1 << 20;

pub fn encode(slot: &GameState) -> Result<Vec<u8>, CoreError> {
    let body = body::encode(slot);
    if body.len() > MAX_BODY_BYTES {
        return Err(CoreError::Snapshot("body exceeds contract maximum".into()));
    }
    let mut out = vec![0; SNAPSHOT_HEADER_BYTES];
    out[0..4].copy_from_slice(b"RCHI");
    put_u16(&mut out, 4, SNAPSHOT_FORMAT_VERSION);
    put_u16(&mut out, 6, SNAPSHOT_HEADER_BYTES as u16);
    put_u32(&mut out, 8, SNAPSHOT_SCHEMA_VERSION);
    put_u32(&mut out, 12, slot.rules_profile_id);
    put_u32(&mut out, 16, RNG_PROFILE_ID);
    put_u32(&mut out, 24, slot.environment_id);
    put_u64(&mut out, 32, slot.episode_generation);
    put_u64(&mut out, 40, body.len() as u64);
    put_u64(&mut out, 48, slot.next_frame_id);
    put_u64(&mut out, 56, slot.next_event_sequence);
    out.extend_from_slice(&body);
    Ok(out)
}

pub fn decode(bytes: &[u8]) -> Result<GameState, CoreError> {
    if bytes.len() < SNAPSHOT_HEADER_BYTES || &bytes[0..4] != b"RCHI" {
        return Err(CoreError::Snapshot("bad magic or truncated header".into()));
    }
    exact_u16(bytes, 4, SNAPSHOT_FORMAT_VERSION, "snapshot format")?;
    exact_u16(bytes, 6, SNAPSHOT_HEADER_BYTES as u16, "header length")?;
    exact_u32(bytes, 8, SNAPSHOT_SCHEMA_VERSION, "snapshot schema")?;
    let rules_profile_id = read_u32(bytes, 12)?;
    if profile::by_id(rules_profile_id).is_none() {
        return Err(CoreError::Snapshot("unsupported rules profile".into()));
    }
    exact_u32(bytes, 16, RNG_PROFILE_ID, "rng profile")?;
    if read_u32(bytes, 20)? != 0 || read_u32(bytes, 28)? != 0 {
        return Err(CoreError::Snapshot(
            "reserved header bits are nonzero".into(),
        ));
    }
    let body_len = usize::try_from(read_u64(bytes, 40)?)
        .map_err(|_| CoreError::Snapshot("body length overflow".into()))?;
    if body_len > MAX_BODY_BYTES || SNAPSHOT_HEADER_BYTES.checked_add(body_len) != Some(bytes.len())
    {
        return Err(CoreError::Snapshot("invalid body length".into()));
    }
    let slot = body::decode(
        &bytes[SNAPSHOT_HEADER_BYTES..],
        read_u32(bytes, 24)?,
        read_u64(bytes, 32)?,
        read_u64(bytes, 48)?,
        read_u64(bytes, 56)?,
        rules_profile_id,
    )?;
    validate(&slot)?;
    Ok(slot)
}

fn validate(slot: &GameState) -> Result<(), CoreError> {
    if slot.next_frame_id == 0 || slot.rng.stream & 1 == 0 {
        return Err(invalid("invalid frame counter or PCG stream"));
    }
    if (slot.lifecycle == EnvironmentLifecycle::Failed) != slot.failure.is_some() {
        return Err(invalid("lifecycle/failure mismatch"));
    }
    match (&slot.lifecycle, &slot.hanchan) {
        (EnvironmentLifecycle::Uninitialized, None) => {}
        (EnvironmentLifecycle::Uninitialized, Some(_)) => {
            return Err(invalid("uninitialized slot contains a match"));
        }
        (_, None) => return Err(invalid("initialized slot is missing its match")),
        (_, Some(h)) => validate_hanchan(slot, h)?,
    }
    Ok(())
}

fn validate_hanchan(slot: &GameState, h: &HanchanState) -> Result<(), CoreError> {
    if h.scores.iter().sum::<i32>() + i32::from(h.riichi_deposits) * 1000
        != slot.rules_profile().starting_points * 4
    {
        return Err(invalid("score conservation failed"));
    }
    if h.dealer >= 4 || h.hand_number >= 4 || !is_seat_permutation(&h.initial_seats) {
        return Err(invalid("invalid match seat metadata"));
    }
    if !h
        .players
        .iter()
        .enumerate()
        .all(|(seat, player)| usize::from(player.seat) == seat)
    {
        return Err(invalid("player seat binding mismatch"));
    }

    let wall = &h.hand.wall;
    let mut wall_seen = [false; 136];
    for &tile in &wall.tiles {
        insert_tile(&mut wall_seen, tile, "wall is not a tile permutation")?;
    }
    // Replacement draws can move the dead-wall boundary behind an already
    // exhausted live cursor. `draw_live` and the maintained count cache both
    // deliberately treat that reversed interval as an empty live wall.
    if wall.live_start > 122
        || wall.live_end > 122
        || !(131..=135).contains(&wall.rinshan_index)
        || !(1..=5).contains(&wall.dora_indicator_count)
    {
        return Err(invalid("invalid wall cursor"));
    }
    if wall.live_wall_counts != recompute_live_wall_counts(wall) {
        return Err(invalid("live wall count cache does not match wall cursor"));
    }
    for (index, &tile) in wall.revealed_dora_indicators.iter().enumerate() {
        let revealed = index < usize::from(wall.dora_indicator_count);
        if (revealed && tile >= 136) || (!revealed && tile != ABSENT) {
            return Err(invalid("invalid revealed dora indicators"));
        }
    }
    if wall.ura_indicators.iter().any(|&tile| tile >= 136) {
        return Err(invalid("invalid ura indicators"));
    }

    let mut located = [false; 136];
    for player in &h.players {
        if player.concealed_tiles.len() > 14 || player.melds.len() > 4 || player.river.len() > 96 {
            return Err(invalid("player collection exceeds its bound"));
        }
        for &tile in &player.concealed_tiles {
            insert_tile(&mut located, tile, "duplicate located tile")?;
        }
        for meld in &player.melds {
            let expected_count = match meld.kind {
                MeldKind::Chi | MeldKind::Pon => 3,
                MeldKind::OpenKan | MeldKind::ClosedKan | MeldKind::AddedKan => 4,
            };
            let source_is_valid = if meld.kind == MeldKind::ClosedKan {
                meld.from_seat == ABSENT && meld.called_tile == ABSENT
            } else {
                meld.from_seat < 4 && meld.called_tile < 136
            };
            if usize::from(meld.tile_count) != expected_count || !source_is_valid {
                return Err(invalid("invalid meld metadata"));
            }
            for (index, &tile) in meld.tiles.iter().enumerate() {
                if index < expected_count {
                    insert_tile(&mut located, tile, "duplicate located tile")?;
                } else if tile != ABSENT {
                    return Err(invalid("invalid meld padding"));
                }
            }
        }
        for entry in &player.river {
            if entry.tile >= 136 {
                return Err(invalid("invalid river tile"));
            }
            // A called discard remains in the river as history and is also
            // referenced by the resulting meld. Uncalled discards have one
            // physical location and therefore participate in uniqueness.
            if !entry.called {
                insert_tile(&mut located, entry.tile, "duplicate located tile")?;
            }
        }
    }

    if h.hand.current_seat >= 4 || (h.hand.current_draw >= 136 && h.hand.current_draw != ABSENT) {
        return Err(invalid("invalid current-turn metadata"));
    }
    if let Some((seat, tile)) = h.hand.last_discard {
        if seat >= 4 || tile >= 136 {
            return Err(invalid("invalid last discard"));
        }
    }
    if let Some(kan) = &h.hand.provisional_kan {
        if kan.seat >= 4 {
            return Err(invalid("invalid provisional kan seat"));
        }
        validate_action(&kan.action)?;
    }

    if let Some(frame) = &h.hand.decision {
        if frame.environment_id != slot.environment_id
            || frame.episode_generation != slot.episode_generation
            || frame.frame_id == 0
            || frame.frame_id >= slot.next_frame_id
            || frame.phase != h.hand.phase
            || frame.eligible_mask() & !0b1111 != 0
            || frame.action_spaces.is_empty()
            || frame.action_spaces.len() > 4
            || !matches!(
                frame.phase,
                HandPhase::SelfTurnDecision
                    | HandPhase::DiscardReactionFrame
                    | HandPhase::KanRobReactionFrame
            )
        {
            return Err(invalid("invalid decision binding"));
        }
        let mut seats = 0_u8;
        for decision in &frame.action_spaces {
            if decision.seat >= 4
                || seats & (1 << decision.seat) != 0
                || decision.candidates.len() < 2
                || decision.candidates.len() > 256
            {
                return Err(invalid("invalid seat decision"));
            }
            seats |= 1 << decision.seat;
            for action in &decision.candidates {
                validate_action(action)?;
            }
        }
        if seats != frame.eligible_mask() {
            return Err(invalid("eligible mask does not match queryable seats"));
        }
    } else if slot.lifecycle == EnvironmentLifecycle::Running {
        // Frame-free automatic phases are valid transition boundaries. Restore
        // stabilizes native games and preserves replay settlement boundaries.
        validate_frame_free_automatic(h)?;
    }

    if slot.lifecycle == EnvironmentLifecycle::Complete
        && h.hand.phase != HandPhase::HanchanComplete
    {
        return Err(invalid("complete slot has incomplete match phase"));
    }
    Ok(())
}

fn validate_frame_free_automatic(h: &HanchanState) -> Result<(), CoreError> {
    match h.hand.phase {
        HandPhase::SelfTurnDecision => {
            let seat = h.hand.current_seat;
            if semantic_representatives(seat, legal::self_turn_for_hanchan(h, seat)).len() != 1 {
                return Err(invalid("frame-free self turn is not automatic"));
            }
        }
        HandPhase::DiscardReactionFrame => {
            let Some((source, tile)) = h.hand.last_discard else {
                return Err(invalid("frame-free reaction is missing its discard"));
            };
            for seat in 0..4_u8 {
                if seat != source
                    && semantic_representatives(
                        seat,
                        legal::reactions_for_hanchan(h, seat, source, tile),
                    )
                    .len()
                        != 1
                {
                    return Err(invalid("frame-free discard reaction is queryable"));
                }
            }
        }
        HandPhase::KanRobReactionFrame => {
            let Some(kan) = &h.hand.provisional_kan else {
                return Err(invalid("frame-free kan reaction is missing its proposal"));
            };
            let tile = kan.action.tiles[0];
            let concealed = kan.action.kind == ActionKind::ClosedKan;
            for seat in 0..4_u8 {
                if seat != kan.seat
                    && semantic_representatives(
                        seat,
                        legal::kan_rob_reactions_for_hanchan(h, seat, kan.seat, tile, concealed),
                    )
                    .len()
                        != 1
                {
                    return Err(invalid("frame-free kan reaction is queryable"));
                }
            }
        }
        HandPhase::Settlement => {}
        _ => return Err(invalid("running slot stopped outside an automatic phase")),
    }
    Ok(())
}

fn validate_action(action: &ActionCandidate) -> Result<(), CoreError> {
    if action.tile_count > 4
        || (action.primary_tile_type >= 34 && action.primary_tile_type != ABSENT)
        || (action.source_seat >= 4 && action.source_seat != ABSENT)
    {
        return Err(invalid("invalid action metadata"));
    }
    for (index, &tile) in action.tiles.iter().enumerate() {
        if index < usize::from(action.tile_count) {
            if tile >= 136 {
                return Err(invalid("invalid action tile"));
            }
        } else if tile != ABSENT {
            return Err(invalid("invalid action padding"));
        }
    }
    Ok(())
}

fn insert_tile(seen: &mut [bool; 136], tile: u8, message: &str) -> Result<(), CoreError> {
    let Some(entry) = seen.get_mut(usize::from(tile)) else {
        return Err(invalid(message));
    };
    if std::mem::replace(entry, true) {
        return Err(invalid(message));
    }
    Ok(())
}

fn is_seat_permutation(seats: &[u8; 4]) -> bool {
    let mut mask = 0_u8;
    for &seat in seats {
        if seat >= 4 || mask & (1 << seat) != 0 {
            return false;
        }
        mask |= 1 << seat;
    }
    mask == 0b1111
}

fn invalid(message: &str) -> CoreError {
    CoreError::Snapshot(message.into())
}
fn put_u16(out: &mut [u8], offset: usize, value: u16) {
    out[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}
fn put_u32(out: &mut [u8], offset: usize, value: u32) {
    out[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}
fn put_u64(out: &mut [u8], offset: usize, value: u64) {
    out[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}
fn read_u16(input: &[u8], offset: usize) -> Result<u16, CoreError> {
    Ok(u16::from_le_bytes(
        input
            .get(offset..offset + 2)
            .ok_or_else(|| CoreError::Snapshot("truncated integer".into()))?
            .try_into()
            .expect("length checked"),
    ))
}
fn read_u32(input: &[u8], offset: usize) -> Result<u32, CoreError> {
    Ok(u32::from_le_bytes(
        input
            .get(offset..offset + 4)
            .ok_or_else(|| CoreError::Snapshot("truncated integer".into()))?
            .try_into()
            .expect("length checked"),
    ))
}
fn read_u64(input: &[u8], offset: usize) -> Result<u64, CoreError> {
    Ok(u64::from_le_bytes(
        input
            .get(offset..offset + 8)
            .ok_or_else(|| CoreError::Snapshot("truncated integer".into()))?
            .try_into()
            .expect("length checked"),
    ))
}
fn exact_u16(input: &[u8], offset: usize, expected: u16, name: &str) -> Result<(), CoreError> {
    if read_u16(input, offset)? != expected {
        Err(CoreError::Snapshot(format!("{name} version mismatch")))
    } else {
        Ok(())
    }
}
fn exact_u32(input: &[u8], offset: usize, expected: u32, name: &str) -> Result<(), CoreError> {
    if read_u32(input, offset)? != expected {
        Err(CoreError::Snapshot(format!("{name} version mismatch")))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_accepts_exhausted_wall_after_replacement_draws() {
        let mut state = GameState::new(0);
        state.reset_from_seed(17);
        let wall = &mut state.hanchan.as_mut().unwrap().hand.wall;
        wall.live_start = 122;
        wall.live_end = 118;
        wall.live_wall_counts = [0; 34];

        let encoded = encode(&state).unwrap();
        let decoded = decode(&encoded).unwrap();
        let restored = &decoded.hanchan.unwrap().hand.wall;
        assert_eq!((restored.live_start, restored.live_end), (122, 118));
    }
}
