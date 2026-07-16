use crate::{
    error::{CoreError, ErrorCode, FailureRecord},
    game::{
        action::{ActionDescriptor, ActionKind, DecisionFrame, SeatDecision},
        phase::{EnvironmentLifecycle, HandPhase, MeldKind, RiichiState, Wind},
        rules::hand::recompute_live_wall_counts,
        state::{
            GameState, HanchanState, HandState, Meld, PlayerState, ProvisionalKan, RiverEntry,
            RngState, WallState,
        },
    },
};

const MAX_CONCEALED: usize = 14;
const MAX_MELDS: usize = 4;
const MAX_RIVER: usize = 96;
const MAX_DECISIONS: usize = 4;
const MAX_ACTIONS: usize = 256;

pub fn encode(slot: &GameState) -> Vec<u8> {
    let mut w = Writer::default();
    w.u8(slot.lifecycle as u8);
    match &slot.failure {
        Some(failure) => {
            w.bool(true);
            w.u16(failure.code as u16);
            w.i64(failure.arg0);
            w.i64(failure.arg1);
        }
        None => w.bool(false),
    }
    w.u64(slot.rng.state);
    w.u64(slot.rng.stream);
    match &slot.hanchan {
        Some(hanchan) => {
            w.bool(true);
            encode_hanchan(&mut w, hanchan);
        }
        None => w.bool(false),
    }
    // Additive schema-v1 extension. Older bodies end immediately after the
    // hanchan; new readers accept both forms so active training checkpoints
    // remain resumable.
    w.u32(slot.hanchan.as_ref().map_or(0, |h| h.completed_kyoku));
    w.bytes
}

pub fn decode(
    bytes: &[u8],
    environment_id: u32,
    episode_generation: u64,
    next_frame_id: u64,
    next_event_sequence: u64,
) -> Result<GameState, CoreError> {
    let mut r = Reader { bytes, at: 0 };
    let lifecycle = lifecycle(r.u8()?)?;
    let failure = if r.bool()? {
        Some(FailureRecord {
            code: error_code(r.u16()?)?,
            arg0: r.i64()?,
            arg1: r.i64()?,
        })
    } else {
        None
    };
    let rng = RngState {
        state: r.u64()?,
        stream: r.u64()?,
    };
    let mut hanchan = if r.bool()? {
        Some(decode_hanchan(&mut r, environment_id, episode_generation)?)
    } else {
        None
    };
    match bytes.len().saturating_sub(r.at) {
        0 => {}
        4 => {
            let completed_kyoku = r.u32()?;
            if let Some(hanchan) = &mut hanchan {
                hanchan.completed_kyoku = completed_kyoku;
            }
        }
        _ => return Err(invalid("trailing body bytes")),
    }
    if r.at != bytes.len() {
        return Err(invalid("trailing body bytes"));
    }
    Ok(GameState {
        environment_id,
        episode_generation,
        lifecycle,
        rng,
        hanchan,
        next_frame_id,
        next_event_sequence,
        failure,
        pending_events: Vec::new(),
        automatic_decisions: 0,
    })
}

fn encode_hanchan(w: &mut Writer, h: &HanchanState) {
    w.u8(h.round_wind as u8);
    w.u8(h.hand_number);
    w.u8(h.dealer);
    w.u16(h.honba);
    w.u16(h.riichi_deposits);
    for value in h.scores {
        w.i32(value);
    }
    w.raw(&h.initial_seats);
    for player in &h.players {
        encode_player(w, player);
    }
    encode_hand(w, &h.hand);
}

fn decode_hanchan(
    r: &mut Reader<'_>,
    environment_id: u32,
    episode_generation: u64,
) -> Result<HanchanState, CoreError> {
    let round_wind = wind(r.u8()?)?;
    let hand_number = r.u8()?;
    let dealer = r.u8()?;
    let honba = r.u16()?;
    let riichi_deposits = r.u16()?;
    let completed_kyoku = u32::from(round_wind as u8)
        .saturating_mul(4)
        .saturating_add(u32::from(hand_number))
        .saturating_add(u32::from(honba));
    let scores = [r.i32()?, r.i32()?, r.i32()?, r.i32()?];
    let initial_seats = r.array::<4>()?;
    let players = [
        decode_player(r)?,
        decode_player(r)?,
        decode_player(r)?,
        decode_player(r)?,
    ];
    let hand = decode_hand(r, environment_id, episode_generation)?;
    Ok(HanchanState {
        round_wind,
        hand_number,
        dealer,
        honba,
        riichi_deposits,
        completed_kyoku,
        scores,
        initial_seats,
        players,
        hand,
    })
}

fn encode_player(w: &mut Writer, p: &PlayerState) {
    w.u8(p.seat);
    w.count(p.concealed_tiles.len());
    w.raw(&p.concealed_tiles);
    w.count(p.melds.len());
    for meld in &p.melds {
        w.u8(meld.kind as u8);
        w.raw(&meld.tiles);
        w.u8(meld.tile_count);
        w.u8(meld.called_tile);
        w.u8(meld.from_seat);
        w.u64(meld.created_sequence);
    }
    w.count(p.river.len());
    for river in &p.river {
        w.u8(river.tile);
        w.u64(river.sequence);
        w.bool(river.riichi_declaration);
        w.bool(river.called);
        w.bool(river.tsumogiri);
    }
    w.u8(p.riichi_state as u8);
    w.bool(p.ippatsu_eligible);
    w.bool(p.permanent_furiten);
    w.bool(p.temporary_furiten);
    w.bool(p.riichi_furiten);
    w.u64(p.forbidden_discard_mask);
}

fn decode_player(r: &mut Reader<'_>) -> Result<PlayerState, CoreError> {
    let seat = r.u8()?;
    let concealed_tiles = r.vec_u8(MAX_CONCEALED, "concealed tiles")?;
    let meld_count = r.count(MAX_MELDS, "melds")?;
    let mut melds = Vec::with_capacity(meld_count);
    for _ in 0..meld_count {
        melds.push(Meld {
            kind: meld_kind(r.u8()?)?,
            tiles: r.array::<4>()?,
            tile_count: r.u8()?,
            called_tile: r.u8()?,
            from_seat: r.u8()?,
            created_sequence: r.u64()?,
        });
    }
    let river_count = r.count(MAX_RIVER, "river")?;
    let mut river = Vec::with_capacity(river_count);
    for _ in 0..river_count {
        river.push(RiverEntry {
            tile: r.u8()?,
            sequence: r.u64()?,
            riichi_declaration: r.bool()?,
            called: r.bool()?,
            tsumogiri: r.bool()?,
        });
    }
    Ok(PlayerState {
        seat,
        concealed_tiles,
        melds,
        river,
        riichi_state: riichi_state(r.u8()?)?,
        ippatsu_eligible: r.bool()?,
        permanent_furiten: r.bool()?,
        temporary_furiten: r.bool()?,
        riichi_furiten: r.bool()?,
        forbidden_discard_mask: r.u64()?,
    })
}

fn encode_hand(w: &mut Writer, h: &HandState) {
    w.u8(h.phase as u8);
    w.raw(&h.wall.tiles);
    w.u8(h.wall.live_start);
    w.u8(h.wall.live_end);
    w.u8(h.wall.rinshan_index);
    w.u8(h.wall.dora_indicator_count);
    w.raw(&h.wall.revealed_dora_indicators);
    w.raw(&h.wall.ura_indicators);
    w.u8(h.current_seat);
    w.u8(h.current_draw);
    w.bool(h.current_draw_is_replacement);
    encode_pair(w, h.last_discard);
    match &h.provisional_kan {
        Some(kan) => {
            w.bool(true);
            w.u8(kan.seat);
            encode_action(w, &kan.action);
        }
        None => w.bool(false),
    }
    match &h.decision_frame {
        Some(frame) => {
            w.bool(true);
            encode_frame(w, frame);
        }
        None => w.bool(false),
    }
}

fn decode_hand(
    r: &mut Reader<'_>,
    environment_id: u32,
    episode_generation: u64,
) -> Result<HandState, CoreError> {
    let phase = hand_phase(r.u8()?)?;
    let mut wall = WallState {
        tiles: r.array::<136>()?,
        live_wall_counts: [0; 34],
        live_start: r.u8()?,
        live_end: r.u8()?,
        rinshan_index: r.u8()?,
        dora_indicator_count: r.u8()?,
        revealed_dora_indicators: r.array::<5>()?,
        ura_indicators: r.array::<5>()?,
    };
    wall.live_wall_counts = recompute_live_wall_counts(&wall);
    let current_seat = r.u8()?;
    let current_draw = r.u8()?;
    let current_draw_is_replacement = r.bool()?;
    let last_discard = decode_pair(r)?;
    let provisional_kan = if r.bool()? {
        Some(ProvisionalKan {
            seat: r.u8()?,
            action: decode_action(r)?,
        })
    } else {
        None
    };
    let decision_frame = if r.bool()? {
        Some(decode_frame(r, environment_id, episode_generation)?)
    } else {
        None
    };
    Ok(HandState {
        phase,
        wall,
        current_seat,
        current_draw,
        current_draw_is_replacement,
        last_discard,
        provisional_kan,
        decision_frame,
    })
}

fn encode_frame(w: &mut Writer, frame: &DecisionFrame) {
    w.u32(frame.environment_id);
    w.u64(frame.episode_generation);
    w.u64(frame.frame_id);
    w.u8(frame.phase as u8);
    w.u8(frame.eligible_mask);
    w.count(frame.decisions.len());
    for decision in &frame.decisions {
        w.u8(decision.seat);
        w.count(decision.actions.len());
        for action in &decision.actions {
            encode_action(w, action);
        }
    }
}

fn decode_frame(
    r: &mut Reader<'_>,
    expected_environment: u32,
    expected_generation: u64,
) -> Result<DecisionFrame, CoreError> {
    let environment_id = r.u32()?;
    let episode_generation = r.u64()?;
    if environment_id != expected_environment || episode_generation != expected_generation {
        return Err(invalid("decision frame binding mismatch"));
    }
    let frame_id = r.u64()?;
    let phase = hand_phase(r.u8()?)?;
    let eligible_mask = r.u8()?;
    let count = r.count(MAX_DECISIONS, "decisions")?;
    let mut decisions = Vec::with_capacity(count);
    for _ in 0..count {
        let seat = r.u8()?;
        let action_count = r.count(MAX_ACTIONS, "actions")?;
        let mut actions = Vec::with_capacity(action_count);
        for _ in 0..action_count {
            actions.push(decode_action(r)?);
        }
        decisions.push(SeatDecision { seat, actions });
    }
    Ok(DecisionFrame {
        environment_id,
        episode_generation,
        frame_id,
        phase,
        eligible_mask,
        decisions,
    })
}

fn encode_action(w: &mut Writer, op: &ActionDescriptor) {
    w.u8(op.kind as u8);
    w.u8(op.primary_tile_type);
    w.u8(op.source_seat);
    w.u8(op.tile_count);
    w.raw(&op.tiles);
    w.u16(op.aux);
    w.u16(op.flags);
}

fn decode_action(r: &mut Reader<'_>) -> Result<ActionDescriptor, CoreError> {
    Ok(ActionDescriptor {
        kind: action_kind(r.u8()?)?,
        primary_tile_type: r.u8()?,
        source_seat: r.u8()?,
        tile_count: r.u8()?,
        tiles: r.array::<4>()?,
        aux: r.u16()?,
        flags: r.u16()?,
    })
}

fn encode_pair(w: &mut Writer, value: Option<(u8, u8)>) {
    match value {
        Some((a, b)) => {
            w.bool(true);
            w.u8(a);
            w.u8(b);
        }
        None => w.bool(false),
    }
}

fn decode_pair(r: &mut Reader<'_>) -> Result<Option<(u8, u8)>, CoreError> {
    if r.bool()? {
        Ok(Some((r.u8()?, r.u8()?)))
    } else {
        Ok(None)
    }
}

#[derive(Default)]
struct Writer {
    bytes: Vec<u8>,
}

impl Writer {
    fn raw(&mut self, value: &[u8]) {
        self.bytes.extend_from_slice(value);
    }
    fn u8(&mut self, value: u8) {
        self.bytes.push(value);
    }
    fn bool(&mut self, value: bool) {
        self.u8(u8::from(value));
    }
    fn u16(&mut self, value: u16) {
        self.raw(&value.to_le_bytes());
    }
    fn u32(&mut self, value: u32) {
        self.raw(&value.to_le_bytes());
    }
    fn i32(&mut self, value: i32) {
        self.raw(&value.to_le_bytes());
    }
    fn u64(&mut self, value: u64) {
        self.raw(&value.to_le_bytes());
    }
    fn i64(&mut self, value: i64) {
        self.raw(&value.to_le_bytes());
    }
    fn count(&mut self, value: usize) {
        self.u16(value as u16);
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl Reader<'_> {
    fn take(&mut self, count: usize) -> Result<&[u8], CoreError> {
        let end = self
            .at
            .checked_add(count)
            .ok_or_else(|| invalid("body offset overflow"))?;
        let result = self
            .bytes
            .get(self.at..end)
            .ok_or_else(|| invalid("truncated body"))?;
        self.at = end;
        Ok(result)
    }
    fn array<const N: usize>(&mut self) -> Result<[u8; N], CoreError> {
        self.take(N)?
            .try_into()
            .map_err(|_| invalid("invalid fixed array"))
    }
    fn u8(&mut self) -> Result<u8, CoreError> {
        Ok(self.take(1)?[0])
    }
    fn bool(&mut self) -> Result<bool, CoreError> {
        match self.u8()? {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(invalid("invalid boolean")),
        }
    }
    fn u16(&mut self) -> Result<u16, CoreError> {
        Ok(u16::from_le_bytes(self.array()?))
    }
    fn u32(&mut self) -> Result<u32, CoreError> {
        Ok(u32::from_le_bytes(self.array()?))
    }
    fn i32(&mut self) -> Result<i32, CoreError> {
        Ok(i32::from_le_bytes(self.array()?))
    }
    fn u64(&mut self) -> Result<u64, CoreError> {
        Ok(u64::from_le_bytes(self.array()?))
    }
    fn i64(&mut self) -> Result<i64, CoreError> {
        Ok(i64::from_le_bytes(self.array()?))
    }
    fn count(&mut self, maximum: usize, name: &str) -> Result<usize, CoreError> {
        let count = usize::from(self.u16()?);
        if count > maximum {
            Err(invalid(&format!("{name} count exceeds maximum")))
        } else {
            Ok(count)
        }
    }
    fn vec_u8(&mut self, maximum: usize, name: &str) -> Result<Vec<u8>, CoreError> {
        let count = self.count(maximum, name)?;
        Ok(self.take(count)?.to_vec())
    }
}

fn lifecycle(value: u8) -> Result<EnvironmentLifecycle, CoreError> {
    match value {
        0 => Ok(EnvironmentLifecycle::Uninitialized),
        1 => Ok(EnvironmentLifecycle::Ready),
        2 => Ok(EnvironmentLifecycle::Running),
        3 => Ok(EnvironmentLifecycle::Complete),
        4 => Ok(EnvironmentLifecycle::Failed),
        _ => Err(invalid("invalid lifecycle")),
    }
}
fn hand_phase(value: u8) -> Result<HandPhase, CoreError> {
    match value {
        0 => Ok(HandPhase::Setup),
        1 => Ok(HandPhase::SelfTurnDecision),
        2 => Ok(HandPhase::DiscardReactionFrame),
        3 => Ok(HandPhase::KanRobReactionFrame),
        4 => Ok(HandPhase::ReplacementTurn),
        5 => Ok(HandPhase::Settlement),
        6 => Ok(HandPhase::HandComplete),
        7 => Ok(HandPhase::HanchanComplete),
        _ => Err(invalid("invalid hand phase")),
    }
}
fn wind(value: u8) -> Result<Wind, CoreError> {
    match value {
        0 => Ok(Wind::East),
        1 => Ok(Wind::South),
        2 => Ok(Wind::West),
        3 => Ok(Wind::North),
        _ => Err(invalid("invalid wind")),
    }
}
fn meld_kind(value: u8) -> Result<MeldKind, CoreError> {
    match value {
        1 => Ok(MeldKind::Chi),
        2 => Ok(MeldKind::Pon),
        3 => Ok(MeldKind::OpenKan),
        4 => Ok(MeldKind::ClosedKan),
        5 => Ok(MeldKind::AddedKan),
        _ => Err(invalid("invalid meld kind")),
    }
}
fn riichi_state(value: u8) -> Result<RiichiState, CoreError> {
    match value {
        0 => Ok(RiichiState::None),
        1 => Ok(RiichiState::Declared),
        2 => Ok(RiichiState::Accepted),
        _ => Err(invalid("invalid riichi state")),
    }
}
fn action_kind(value: u8) -> Result<ActionKind, CoreError> {
    match value {
        0 => Ok(ActionKind::Pass),
        1 => Ok(ActionKind::Discard),
        2 => Ok(ActionKind::RiichiDiscard),
        3 => Ok(ActionKind::Chi),
        4 => Ok(ActionKind::Pon),
        5 => Ok(ActionKind::OpenKan),
        6 => Ok(ActionKind::ClosedKan),
        7 => Ok(ActionKind::AddedKan),
        8 => Ok(ActionKind::Ron),
        9 => Ok(ActionKind::Tsumo),
        10 => Ok(ActionKind::AbortiveDeclaration),
        _ => Err(invalid("invalid action kind")),
    }
}
fn error_code(value: u16) -> Result<ErrorCode, CoreError> {
    match value {
        0 => Ok(ErrorCode::None),
        1 => Ok(ErrorCode::EnvironmentOutOfRange),
        2 => Ok(ErrorCode::EpisodeGenerationMismatch),
        3 => Ok(ErrorCode::FrameIdMismatch),
        4 => Ok(ErrorCode::MissingSeat),
        5 => Ok(ErrorCode::DuplicateSeat),
        6 => Ok(ErrorCode::IneligibleSeat),
        7 => Ok(ErrorCode::ActionOutOfRange),
        8 => Ok(ErrorCode::EnvironmentNotReady),
        9 => Ok(ErrorCode::InternalPanic),
        10 => Ok(ErrorCode::SnapshotInvalid),
        11 => Ok(ErrorCode::SnapshotVersionMismatch),
        12 => Ok(ErrorCode::EnvClosed),
        _ => Err(invalid("invalid error code")),
    }
}

fn invalid(message: &str) -> CoreError {
    CoreError::Snapshot(message.into())
}
