//! Search-only hidden-wall determinization.
//!
//! A particle is a complete, immutable position-to-physical-tile assignment.
//! Search branches which share a particle therefore see the same tile at the
//! same physical wall position even when calls or kans make them request wall
//! positions in a different order.  The production environment never samples
//! from this module implicitly; callers must apply a particle to a cloned
//! [`GameState`].

use std::fmt;

use super::{
    phase::{HandPhase, RiichiState},
    rules::hand::recompute_live_wall_counts,
    rules::{hand::tile_type_counts, shanten},
    state::{GameState, RngState, WallState},
};

const WALL_SIZE: usize = 136;
const LIVE_WALL_END: usize = 122;
const DORA_INDICATOR_START: usize = 130;
const URA_INDICATOR_START: usize = 131;

/// Why a privileged wall particle could not be constructed or applied.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SearchError {
    MissingHanchan,
    InvalidWallCursor,
    ParticleRootMismatch,
    UnsupportedPublicRoot,
    PublicParticleExhausted,
}

impl fmt::Display for SearchError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::MissingHanchan => "search root has no active hanchan",
            Self::InvalidWallCursor => "search root has an invalid wall cursor",
            Self::ParticleRootMismatch => "wall particle does not match the search root",
            Self::UnsupportedPublicRoot => {
                "public-information search requires the observer's isolated self-turn decision"
            }
            Self::PublicParticleExhausted => {
                "could not sample public hidden state consistent with accepted riichi"
            }
        };
        formatter.write_str(message)
    }
}

/// One public-information determinization for an isolated self-turn decision.
///
/// The observer's concealed hand and every visible tile stay fixed. Opponent
/// concealed tiles and unresolved physical wall positions are shuffled as one
/// pool without replacement. Accepted-riichi hands are rejection-sampled to
/// remain tenpai. This deliberately excludes reaction frames: changing an
/// opponent hand after its call candidates have been materialized would make
/// the root decision stale.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PublicInformationParticle {
    root_fingerprint: u64,
    observer_seat: u8,
    particle_key: u64,
    tiles: [u8; WALL_SIZE],
    concealed_tiles: [Vec<u8>; 4],
}

impl PublicInformationParticle {
    pub fn sample(
        root: &GameState,
        observer_seat: u8,
        particle_key: u64,
    ) -> Result<Self, SearchError> {
        let hanchan = root.hanchan.as_ref().ok_or(SearchError::MissingHanchan)?;
        let wall = &hanchan.hand.wall;
        validate_wall_cursor(wall)?;
        validate_public_root(root, observer_seat)?;

        let unresolved = unresolved_positions(wall);
        let positions = unresolved
            .iter()
            .enumerate()
            .filter_map(|(position, &value)| value.then_some(position))
            .collect::<Vec<_>>();
        let mut hidden_pool = positions
            .iter()
            .map(|&position| wall.tiles[position])
            .collect::<Vec<_>>();
        for seat in 0..4_u8 {
            if seat != observer_seat {
                hidden_pool.extend_from_slice(&hanchan.players[usize::from(seat)].concealed_tiles);
            }
        }
        let mut rng = public_particle_rng(root, observer_seat, particle_key);
        for _ in 0..1024 {
            for index in (1..hidden_pool.len()).rev() {
                let other = rng.bounded((index + 1) as u32) as usize;
                hidden_pool.swap(index, other);
            }
            let mut cursor = 0;
            let mut tiles = wall.tiles;
            for &position in &positions {
                tiles[position] = hidden_pool[cursor];
                cursor += 1;
            }
            let mut concealed_tiles =
                std::array::from_fn(|seat| hanchan.players[seat].concealed_tiles.clone());
            for seat in 0..4_u8 {
                if seat == observer_seat {
                    continue;
                }
                let length = concealed_tiles[usize::from(seat)].len();
                concealed_tiles[usize::from(seat)] = hidden_pool[cursor..cursor + length].to_vec();
                concealed_tiles[usize::from(seat)].sort_unstable();
                cursor += length;
            }
            debug_assert_eq!(cursor, hidden_pool.len());
            if public_hands_are_consistent(hanchan, &concealed_tiles, observer_seat) {
                return Ok(Self {
                    root_fingerprint: public_root_fingerprint(root, observer_seat),
                    observer_seat,
                    particle_key,
                    tiles,
                    concealed_tiles,
                });
            }
        }
        Err(SearchError::PublicParticleExhausted)
    }

    pub fn particle_key(&self) -> u64 {
        self.particle_key
    }

    pub fn apply(&self, state: &mut GameState) -> Result<(), SearchError> {
        validate_public_root(state, self.observer_seat)?;
        if public_root_fingerprint(state, self.observer_seat) != self.root_fingerprint {
            return Err(SearchError::ParticleRootMismatch);
        }
        let hanchan = state.hanchan.as_mut().ok_or(SearchError::MissingHanchan)?;
        hanchan.hand.wall.tiles = self.tiles;
        refresh_wall_caches(&mut hanchan.hand.wall);
        for seat in 0..4_usize {
            hanchan.players[seat].concealed_tiles = self.concealed_tiles[seat].clone();
        }
        Ok(())
    }
}

impl std::error::Error for SearchError {}

/// One determinization of information hidden only in the remaining wall.
///
/// Opponent concealed hands and every already observed tile stay fixed.  The
/// particle permutes all unresolved physical wall positions, including tiles
/// shifted behind the live-wall boundary by kans, future rinshan tiles, hidden
/// dora indicators, and ura indicators.  Revealed dora and already drawn
/// rinshan positions are fixed.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PrivilegedWallParticle {
    root_fingerprint: u64,
    particle_key: u64,
    tiles: [u8; WALL_SIZE],
    unresolved: [bool; WALL_SIZE],
}

impl PrivilegedWallParticle {
    pub fn sample(root: &GameState, particle_key: u64) -> Result<Self, SearchError> {
        let wall = &root
            .hanchan
            .as_ref()
            .ok_or(SearchError::MissingHanchan)?
            .hand
            .wall;
        validate_wall_cursor(wall)?;

        let unresolved = unresolved_positions(wall);
        let mut positions = unresolved
            .iter()
            .enumerate()
            .filter_map(|(position, &value)| value.then_some(position))
            .collect::<Vec<_>>();
        let mut tiles = wall.tiles;
        let mut rng = particle_rng(root, particle_key, wall);

        // Fisher-Yates over positions is equivalent to sampling physical
        // tiles without replacement.  Materializing the 136-byte chance table
        // once also makes resolution independent of traversal order.
        for index in (1..positions.len()).rev() {
            let other = rng.bounded((index + 1) as u32) as usize;
            tiles.swap(positions[index], positions[other]);
        }
        positions.clear();

        Ok(Self {
            root_fingerprint: wall_fingerprint(wall),
            particle_key,
            tiles,
            unresolved,
        })
    }

    pub fn particle_key(&self) -> u64 {
        self.particle_key
    }

    /// Resolve one physical wall position from the immutable chance table.
    pub fn tile_at(&self, position: usize) -> Option<u8> {
        self.tiles.get(position).copied()
    }

    pub fn is_unresolved_position(&self, position: usize) -> bool {
        self.unresolved.get(position).copied().unwrap_or(false)
    }

    pub fn unresolved_position_count(&self) -> usize {
        self.unresolved.iter().filter(|&&value| value).count()
    }

    /// Apply this determinization to a cloned search state.
    pub fn apply(&self, state: &mut GameState) -> Result<(), SearchError> {
        let wall = &mut state
            .hanchan
            .as_mut()
            .ok_or(SearchError::MissingHanchan)?
            .hand
            .wall;
        validate_wall_cursor(wall)?;
        if wall_fingerprint(wall) != self.root_fingerprint {
            return Err(SearchError::ParticleRootMismatch);
        }
        wall.tiles = self.tiles;
        refresh_wall_caches(wall);
        Ok(())
    }
}

/// Clone a root, bind it to a target batch slot, and apply a wall particle.
pub fn fork_with_privileged_wall(
    root: &GameState,
    target_environment_id: u32,
    particle_key: u64,
) -> Result<GameState, SearchError> {
    let particle = PrivilegedWallParticle::sample(root, particle_key)?;
    let mut branch = root.clone();
    rebind_environment(&mut branch, target_environment_id);
    // The current kyoku reads only its explicit wall.  Giving each particle a
    // target-independent future RNG makes optional full-match calibration
    // sample later kyoku as well, while paired action branches retain common
    // future chance. Production environment RNG is never modified.
    branch.rng = future_hand_rng(root, particle_key);
    particle.apply(&mut branch)?;
    branch.pending_events.clear();
    Ok(branch)
}

/// Clone a root and jointly resample public-hidden hands and wall tiles.
pub fn fork_with_public_information(
    root: &GameState,
    target_environment_id: u32,
    observer_seat: u8,
    particle_key: u64,
) -> Result<GameState, SearchError> {
    let particle = PublicInformationParticle::sample(root, observer_seat, particle_key)?;
    let mut branch = root.clone();
    rebind_environment(&mut branch, target_environment_id);
    branch.rng = future_hand_rng(root, particle_key);
    particle.apply(&mut branch)?;
    branch.pending_events.clear();
    Ok(branch)
}

fn future_hand_rng(root: &GameState, particle_key: u64) -> RngState {
    RngState {
        state: mix64(
            root.rng.state
                ^ particle_key.rotate_left(7)
                ^ root.episode_generation.rotate_left(23)
                ^ 0x4655_5455_5245_5741,
        ),
        stream: mix64(root.rng.stream ^ particle_key.rotate_right(13) ^ 0x4c4c_5f43_4841_4e43) | 1,
    }
}

fn validate_public_root(root: &GameState, observer_seat: u8) -> Result<(), SearchError> {
    let hanchan = root.hanchan.as_ref().ok_or(SearchError::MissingHanchan)?;
    let decision = hanchan
        .hand
        .decision
        .as_ref()
        .ok_or(SearchError::UnsupportedPublicRoot)?;
    if observer_seat >= 4
        || hanchan.hand.phase != HandPhase::SelfTurnDecision
        || hanchan.hand.current_seat != observer_seat
        || decision.action_spaces.len() != 1
        || decision.action_spaces[0].seat != observer_seat
    {
        return Err(SearchError::UnsupportedPublicRoot);
    }
    Ok(())
}

fn public_hands_are_consistent(
    hanchan: &super::state::HanchanState,
    concealed_tiles: &[Vec<u8>; 4],
    observer_seat: u8,
) -> bool {
    (0..4_usize).all(|seat| {
        seat == usize::from(observer_seat)
            || hanchan.players[seat].riichi_state != RiichiState::Accepted
            || shanten::calculate(
                &tile_type_counts(&concealed_tiles[seat]),
                hanchan.players[seat].melds.len() as u8,
            )
            .overall
                == 0
    })
}

/// Bind cloned search state to a different batched-environment slot.
pub fn rebind_environment(state: &mut GameState, environment_id: u32) {
    state.environment_id = environment_id;
    if let Some(decision) = state
        .hanchan
        .as_mut()
        .and_then(|hanchan| hanchan.hand.decision.as_mut())
    {
        decision.environment_id = environment_id;
    }
    for event in &mut state.pending_events {
        event.environment_id = environment_id;
    }
}

fn validate_wall_cursor(wall: &WallState) -> Result<(), SearchError> {
    if wall.live_start as usize > LIVE_WALL_END
        || wall.live_end as usize > LIVE_WALL_END
        || !(131..=135).contains(&wall.rinshan_index)
        || !(1..=5).contains(&wall.dora_indicator_count)
    {
        return Err(SearchError::InvalidWallCursor);
    }
    Ok(())
}

fn unresolved_positions(wall: &WallState) -> [bool; WALL_SIZE] {
    let mut unresolved = [false; WALL_SIZE];

    // All not-yet-consumed live-wall positions are hidden.  Positions shifted
    // behind live_end by a kan remain in the hidden pool even though no future
    // draw reaches them; otherwise the sampler would condition on their actual
    // (privileged) tile identities.
    unresolved[usize::from(wall.live_start)..LIVE_WALL_END].fill(true);

    // Unrevealed dead-wall positions are hidden.  Already consumed rinshan
    // positions are represented in a player's hand/river and must stay fixed.
    unresolved[132..=usize::from(wall.rinshan_index)].fill(true);
    for indicator in 0..5 {
        let dora_position = DORA_INDICATOR_START - indicator * 2;
        if indicator >= usize::from(wall.dora_indicator_count) {
            unresolved[dora_position] = true;
        }
        unresolved[URA_INDICATOR_START - indicator * 2] = true;
    }
    unresolved
}

fn refresh_wall_caches(wall: &mut WallState) {
    wall.live_wall_counts = recompute_live_wall_counts(wall);
    for indicator in 0..usize::from(wall.dora_indicator_count) {
        wall.revealed_dora_indicators[indicator] = wall.tiles[DORA_INDICATOR_START - indicator * 2];
    }
    for indicator in usize::from(wall.dora_indicator_count)..5 {
        wall.revealed_dora_indicators[indicator] = 255;
    }
    for indicator in 0..5 {
        wall.ura_indicators[indicator] = wall.tiles[URA_INDICATOR_START - indicator * 2];
    }
}

fn particle_rng(root: &GameState, particle_key: u64, wall: &WallState) -> RngState {
    let fingerprint = wall_fingerprint(wall);
    RngState {
        state: mix64(
            fingerprint
                ^ particle_key
                ^ root.episode_generation.rotate_left(19)
                ^ 0x5345_4152_4348_5741,
        ),
        stream: mix64(
            particle_key.rotate_left(29) ^ fingerprint.rotate_right(11) ^ 0x4c4c_5f50_4152_5449,
        ) | 1,
    }
}

fn public_particle_rng(root: &GameState, observer_seat: u8, particle_key: u64) -> RngState {
    let fingerprint = public_root_fingerprint(root, observer_seat);
    RngState {
        state: mix64(
            fingerprint
                ^ particle_key
                ^ u64::from(observer_seat).rotate_left(37)
                ^ 0x5055_424c_4943_494e,
        ),
        stream: mix64(
            particle_key.rotate_left(29) ^ fingerprint.rotate_right(11) ^ 0x464f_5f50_4152_5449,
        ) | 1,
    }
}

fn public_root_fingerprint(root: &GameState, observer_seat: u8) -> u64 {
    let Some(hanchan) = root.hanchan.as_ref() else {
        return 0;
    };
    let mut value = wall_fingerprint(&hanchan.hand.wall) ^ u64::from(observer_seat).rotate_left(43);
    for player in &hanchan.players {
        value = mix64(value ^ u64::from(player.seat).rotate_left(7));
        for &tile in &player.concealed_tiles {
            value = mix64(value ^ u64::from(tile).rotate_left(17));
        }
    }
    mix64(value)
}

fn wall_fingerprint(wall: &WallState) -> u64 {
    let mut value = 0x9e37_79b9_7f4a_7c15_u64;
    for (position, &tile) in wall.tiles.iter().enumerate() {
        // Past and otherwise fixed positions identify the root.  Unresolved
        // contents deliberately remain part of the fingerprint so a particle
        // cannot accidentally be applied to another state with equal cursors.
        value = mix64(value ^ u64::from(tile) ^ (position as u64).rotate_left(17));
    }
    value ^= u64::from(wall.live_start)
        | (u64::from(wall.live_end) << 8)
        | (u64::from(wall.rinshan_index) << 16)
        | (u64::from(wall.dora_indicator_count) << 24);
    mix64(value)
}

fn mix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{game::action::ActionSelection, snapshot};

    fn root(seed: u64) -> GameState {
        let mut state = GameState::new(0);
        state.reset_from_seed(seed);
        state.take_events();
        state
    }

    #[test]
    fn particle_preserves_fixed_positions_and_hidden_multiset() {
        let state = root(17);
        let wall = &state.hanchan.as_ref().unwrap().hand.wall;
        let particle = PrivilegedWallParticle::sample(&state, 91).unwrap();
        let mut original_hidden = Vec::new();
        let mut sampled_hidden = Vec::new();
        for position in 0..WALL_SIZE {
            if particle.is_unresolved_position(position) {
                original_hidden.push(wall.tiles[position]);
                sampled_hidden.push(particle.tile_at(position).unwrap());
            } else {
                assert_eq!(particle.tile_at(position), Some(wall.tiles[position]));
            }
        }
        original_hidden.sort_unstable();
        sampled_hidden.sort_unstable();
        assert_eq!(sampled_hidden, original_hidden);
        assert!(particle.unresolved_position_count() > 60);
    }

    #[test]
    fn public_particle_preserves_observer_and_joint_hidden_multiset() {
        let state = root(19);
        let hanchan = state.hanchan.as_ref().unwrap();
        let observer = hanchan.hand.current_seat;
        let unresolved = unresolved_positions(&hanchan.hand.wall);
        let mut before = unresolved
            .iter()
            .enumerate()
            .filter_map(|(position, &hidden)| hidden.then_some(hanchan.hand.wall.tiles[position]))
            .collect::<Vec<_>>();
        for seat in 0..4_usize {
            if seat != usize::from(observer) {
                before.extend_from_slice(&hanchan.players[seat].concealed_tiles);
            }
        }
        let observer_hand = hanchan.players[usize::from(observer)]
            .concealed_tiles
            .clone();
        let particle = PublicInformationParticle::sample(&state, observer, 23).unwrap();
        let mut branch = state.clone();
        particle.apply(&mut branch).unwrap();
        let sampled = branch.hanchan.as_ref().unwrap();
        let mut after = unresolved
            .iter()
            .enumerate()
            .filter_map(|(position, &hidden)| hidden.then_some(sampled.hand.wall.tiles[position]))
            .collect::<Vec<_>>();
        for seat in 0..4_usize {
            if seat != usize::from(observer) {
                after.extend_from_slice(&sampled.players[seat].concealed_tiles);
            }
        }
        before.sort_unstable();
        after.sort_unstable();
        assert_eq!(after, before);
        assert_eq!(
            sampled.players[usize::from(observer)].concealed_tiles,
            observer_hand,
        );
        assert_ne!(
            sampled.players[(usize::from(observer) + 1) % 4].concealed_tiles,
            hanchan.players[(usize::from(observer) + 1) % 4].concealed_tiles,
        );
    }

    #[test]
    fn public_particle_is_reproducible_and_rejects_wrong_observer() {
        let state = root(21);
        let observer = state.hanchan.as_ref().unwrap().hand.current_seat;
        assert_eq!(
            PublicInformationParticle::sample(&state, observer, 29).unwrap(),
            PublicInformationParticle::sample(&state, observer, 29).unwrap(),
        );
        assert_eq!(
            PublicInformationParticle::sample(&state, (observer + 1) % 4, 29),
            Err(SearchError::UnsupportedPublicRoot),
        );
    }

    #[test]
    fn same_particle_is_order_independent_and_reproducible() {
        let state = root(23);
        let first = PrivilegedWallParticle::sample(&state, 5).unwrap();
        let second = PrivilegedWallParticle::sample(&state, 5).unwrap();
        assert_eq!(first, second);
        for position in (0..WALL_SIZE).rev() {
            assert_eq!(first.tile_at(position), second.tile_at(position));
        }
    }

    #[test]
    fn future_hand_rng_is_paired_by_particle_not_target_slot() {
        let state = root(29);
        let first = fork_with_privileged_wall(&state, 1, 17).unwrap();
        let second = fork_with_privileged_wall(&state, 2, 17).unwrap();
        let third = fork_with_privileged_wall(&state, 3, 18).unwrap();
        assert_eq!(first.rng, second.rng);
        assert_ne!(first.rng, third.rng);
    }

    #[test]
    fn applying_particle_produces_a_valid_snapshot() {
        let state = root(31);
        let mut branch = state.clone();
        PrivilegedWallParticle::sample(&state, 7)
            .unwrap()
            .apply(&mut branch)
            .unwrap();
        let payload = snapshot::encode(&branch).unwrap();
        assert_eq!(snapshot::decode(&payload).unwrap(), branch);
    }

    #[test]
    fn paired_forks_rebind_actions_and_follow_identical_trace() {
        let state = root(41);
        let mut first = fork_with_privileged_wall(&state, 3, 101).unwrap();
        let mut second = fork_with_privileged_wall(&state, 9, 101).unwrap();
        for _ in 0..24 {
            let Some(first_frame) = first
                .hanchan
                .as_ref()
                .and_then(|hanchan| hanchan.hand.decision.as_ref())
                .cloned()
            else {
                break;
            };
            let second_frame = second
                .hanchan
                .as_ref()
                .and_then(|hanchan| hanchan.hand.decision.as_ref())
                .cloned()
                .unwrap();
            let first_actions = first_frame
                .action_spaces
                .iter()
                .map(|space| ActionSelection::bind(&first_frame, space.seat, 0))
                .collect::<Vec<_>>();
            let second_actions = second_frame
                .action_spaces
                .iter()
                .map(|space| ActionSelection::bind(&second_frame, space.seat, 0))
                .collect::<Vec<_>>();
            first.step(&first_actions).unwrap();
            second.step(&second_actions).unwrap();
            rebind_environment(&mut second, 3);
            assert_eq!(
                snapshot::encode(&first).unwrap(),
                snapshot::encode(&second).unwrap()
            );
            rebind_environment(&mut second, 9);
        }
    }

    #[test]
    fn particle_rejects_a_different_root() {
        let first = root(3);
        let mut second = root(4);
        let particle = PrivilegedWallParticle::sample(&first, 0).unwrap();
        assert_eq!(
            particle.apply(&mut second),
            Err(SearchError::ParticleRootMismatch),
        );
    }
}
