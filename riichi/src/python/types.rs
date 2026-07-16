use pyo3::{
    prelude::*,
    sync::PyOnceLock,
    types::{PyBytes, PyDict, PyTuple},
};
use riichi_core::{
    game::{phase::RiichiState, rules::hand::tile_type_counts, state::GameState},
    Action as CoreAction, ActionKind as CoreActionKind,
};

use crate::batch_env::BatchTransition;

#[pyclass(name = "ActionKind", frozen, eq, eq_int)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum PyActionKind {
    Pass = 0,
    Discard = 1,
    RiichiDiscard = 2,
    Chi = 3,
    Pon = 4,
    OpenKan = 5,
    ClosedKan = 6,
    AddedKan = 7,
    Ron = 8,
    Tsumo = 9,
    AbortiveDeclaration = 10,
}

impl From<CoreActionKind> for PyActionKind {
    fn from(value: CoreActionKind) -> Self {
        match value {
            CoreActionKind::Pass => Self::Pass,
            CoreActionKind::Discard => Self::Discard,
            CoreActionKind::RiichiDiscard => Self::RiichiDiscard,
            CoreActionKind::Chi => Self::Chi,
            CoreActionKind::Pon => Self::Pon,
            CoreActionKind::OpenKan => Self::OpenKan,
            CoreActionKind::ClosedKan => Self::ClosedKan,
            CoreActionKind::AddedKan => Self::AddedKan,
            CoreActionKind::Ron => Self::Ron,
            CoreActionKind::Tsumo => Self::Tsumo,
            CoreActionKind::AbortiveDeclaration => Self::AbortiveDeclaration,
        }
    }
}

#[pyclass(name = "Action", frozen)]
#[derive(Clone, Debug)]
pub struct PyAction {
    pub(crate) inner: CoreAction,
}

#[pymethods]
impl PyAction {
    #[getter]
    fn environment_id(&self) -> u32 {
        self.inner.environment_id
    }
    #[getter]
    fn episode_generation(&self) -> u64 {
        self.inner.episode_generation
    }
    #[getter]
    fn frame_id(&self) -> u64 {
        self.inner.frame_id
    }
    #[getter]
    fn seat(&self) -> u8 {
        self.inner.seat
    }
    #[getter]
    fn action_index(&self) -> u32 {
        self.inner.action_index
    }
    #[getter]
    fn kind(&self) -> PyActionKind {
        self.inner.kind.into()
    }
    #[getter]
    fn primary_tile_type(&self) -> Option<u8> {
        (self.inner.primary_tile_type != 255).then_some(self.inner.primary_tile_type)
    }
    #[getter]
    fn source_seat(&self) -> Option<u8> {
        (self.inner.source_seat != 255).then_some(self.inner.source_seat)
    }
    #[getter]
    fn tiles(&self) -> Vec<u8> {
        self.inner.tiles[..usize::from(self.inner.tile_count)].to_vec()
    }
    #[getter]
    fn aux(&self) -> u16 {
        self.inner.aux
    }
    #[getter]
    fn flags(&self) -> u16 {
        self.inner.flags
    }
    fn __repr__(&self) -> String {
        format!(
            "Action(env={}, generation={}, frame={}, seat={}, index={}, kind={:?})",
            self.inner.environment_id,
            self.inner.episode_generation,
            self.inner.frame_id,
            self.inner.seat,
            self.inner.action_index,
            self.inner.kind
        )
    }
}

#[pyclass(name = "Decision", frozen)]
#[derive(Clone, Debug)]
pub struct PyDecision {
    #[pyo3(get)]
    pub environment_id: u32,
    #[pyo3(get)]
    pub episode_generation: u64,
    #[pyo3(get)]
    pub frame_id: u64,
    #[pyo3(get)]
    pub seat: u8,
    #[pyo3(get)]
    pub flags: u32,
    #[pyo3(get)]
    pub current_draw: Option<u8>,
    pub(crate) concealed_counts: [u8; 34],
    pub(crate) actions: Vec<PyAction>,
}

#[pymethods]
impl PyDecision {
    #[getter]
    fn concealed_counts(&self) -> Vec<u8> {
        self.concealed_counts.to_vec()
    }
    #[getter]
    fn actions<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let values = self
            .actions
            .iter()
            .cloned()
            .map(|value| Py::new(py, value))
            .collect::<PyResult<Vec<_>>>()?;
        PyTuple::new(py, values)
    }
}

#[pyclass(name = "Meld", frozen)]
#[derive(Clone, Debug)]
pub struct PyMeld {
    #[pyo3(get)]
    pub seat: u8,
    #[pyo3(get)]
    pub kind: u8,
    #[pyo3(get)]
    pub from_seat: Option<u8>,
    #[pyo3(get)]
    pub called_tile: Option<u8>,
    #[pyo3(get)]
    pub tiles: Vec<u8>,
    #[pyo3(get)]
    pub created_sequence: u64,
}

#[pyclass(name = "RiverTile", frozen)]
#[derive(Clone, Debug)]
pub struct PyRiverTile {
    #[pyo3(get)]
    pub seat: u8,
    #[pyo3(get)]
    pub tile: u8,
    #[pyo3(get)]
    pub sequence: u64,
    #[pyo3(get)]
    pub riichi_declaration: bool,
    #[pyo3(get)]
    pub called: bool,
    #[pyo3(get)]
    pub tsumogiri: bool,
}

#[pyclass(name = "HiddenState", frozen)]
#[derive(Clone, Debug)]
pub struct PyHiddenState {
    pub(crate) concealed_counts: Vec<[u8; 34]>,
    pub(crate) concealed_tile_ids: Vec<Vec<u8>>,
    pub(crate) live_wall_counts: [u8; 34],
    pub(crate) wall: [u8; 136],
    pub(crate) wall_indices: [u8; 4],
    pub(crate) ura_indicators: Vec<u8>,
}

#[pymethods]
impl PyHiddenState {
    #[getter]
    fn concealed_counts(&self) -> Vec<Vec<u8>> {
        self.concealed_counts
            .iter()
            .map(|counts| counts.to_vec())
            .collect()
    }
    #[getter]
    fn concealed_tile_ids(&self) -> Vec<Vec<u8>> {
        self.concealed_tile_ids.clone()
    }
    #[getter]
    fn live_wall_counts(&self) -> Vec<u8> {
        self.live_wall_counts.to_vec()
    }
    #[getter]
    fn wall(&self) -> Vec<u8> {
        self.wall.to_vec()
    }
    #[getter]
    fn wall_indices(&self) -> (u8, u8, u8, u8) {
        self.wall_indices.into()
    }
    #[getter]
    fn ura_indicators(&self) -> Vec<u8> {
        self.ura_indicators.clone()
    }
}

#[pyclass(name = "State", frozen)]
#[derive(Clone, Debug)]
pub struct PyState {
    #[pyo3(get)]
    pub environment_id: u32,
    #[pyo3(get)]
    pub episode_generation: u64,
    #[pyo3(get)]
    pub frame_id: u64,
    #[pyo3(get)]
    pub phase: u8,
    #[pyo3(get)]
    pub eligible_mask: u8,
    #[pyo3(get)]
    pub lifecycle: u8,
    #[pyo3(get)]
    pub status: u16,
    #[pyo3(get)]
    pub error_code: u16,
    #[pyo3(get)]
    pub error_args: (i64, i64),
    #[pyo3(get)]
    pub round_wind: u8,
    #[pyo3(get)]
    pub hand_number: u8,
    #[pyo3(get)]
    pub dealer: u8,
    #[pyo3(get)]
    pub honba: u16,
    #[pyo3(get)]
    pub riichi_deposits: u16,
    #[pyo3(get)]
    pub live_wall_remaining: u8,
    #[pyo3(get)]
    pub scores: (i32, i32, i32, i32),
    #[pyo3(get)]
    pub seat_flags: (u32, u32, u32, u32),
    #[pyo3(get)]
    pub dora_indicators: Vec<u8>,
    pub(crate) decisions: Vec<PyDecision>,
    pub(crate) melds: Vec<PyMeld>,
    pub(crate) rivers: Vec<PyRiverTile>,
    pub(crate) hidden: Option<PyHiddenState>,
}

#[pymethods]
impl PyState {
    #[getter]
    fn decisions<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let values = self
            .decisions
            .iter()
            .cloned()
            .map(|value| Py::new(py, value))
            .collect::<PyResult<Vec<_>>>()?;
        PyTuple::new(py, values)
    }
    #[getter]
    fn melds<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let values = self
            .melds
            .iter()
            .cloned()
            .map(|value| Py::new(py, value))
            .collect::<PyResult<Vec<_>>>()?;
        PyTuple::new(py, values)
    }
    #[getter]
    fn rivers<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let values = self
            .rivers
            .iter()
            .cloned()
            .map(|value| Py::new(py, value))
            .collect::<PyResult<Vec<_>>>()?;
        PyTuple::new(py, values)
    }
    #[getter]
    fn hidden(&self) -> Option<PyHiddenState> {
        self.hidden.clone()
    }
}

#[pyclass(name = "Event", frozen)]
#[derive(Clone, Debug)]
pub struct PyEvent {
    #[pyo3(get)]
    pub environment_id: u32,
    #[pyo3(get)]
    pub episode_generation: u64,
    #[pyo3(get)]
    pub sequence: u64,
    #[pyo3(get)]
    pub kind: u16,
    #[pyo3(get)]
    pub kind_name: &'static str,
    #[pyo3(get)]
    pub actor_seat: Option<u8>,
    #[pyo3(get)]
    pub target_seat: Option<u8>,
    #[pyo3(get)]
    pub visibility_mask: u8,
    #[pyo3(get)]
    pub args: (i64, i64, i64, i64),
    pub(crate) payload: Vec<u8>,
}

#[pymethods]
impl PyEvent {
    #[getter]
    fn payload<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyBytes>> {
        (!self.payload.is_empty()).then(|| PyBytes::new(py, &self.payload))
    }
}

#[pyclass(name = "Transition", frozen)]
pub struct PyTransition {
    #[pyo3(get)]
    pub transition_id: u64,
    #[pyo3(get)]
    pub validation_ns: u64,
    #[pyo3(get)]
    pub env_step_ns: u64,
    #[pyo3(get)]
    pub materialization_ns: u64,
    #[pyo3(get)]
    pub exchange_ns: u64,
    pub(crate) states: Vec<PyState>,
    pub(crate) events: Vec<PyEvent>,
    pub(crate) projection: PyOnceLock<Py<PyDict>>,
}

#[pymethods]
impl PyTransition {
    #[getter]
    fn states<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let values = self
            .states
            .iter()
            .cloned()
            .map(|value| Py::new(py, value))
            .collect::<PyResult<Vec<_>>>()?;
        PyTuple::new(py, values)
    }
    #[getter]
    fn events<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let values = self
            .events
            .iter()
            .cloned()
            .map(|value| Py::new(py, value))
            .collect::<PyResult<Vec<_>>>()?;
        PyTuple::new(py, values)
    }
    fn as_numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        crate::python::projection::as_numpy(self, py)
    }
}

pub fn materialize_transition(value: BatchTransition, privileged: bool) -> PyTransition {
    let states = value
        .states
        .iter()
        .map(|state| materialize_state(state, privileged))
        .collect();
    let events = value
        .events
        .into_iter()
        .map(|event| PyEvent {
            environment_id: event.environment_id,
            episode_generation: event.episode_generation,
            sequence: event.sequence,
            kind: event.kind as u16,
            kind_name: event.kind.mjai_name(),
            actor_seat: (event.actor_seat != 255).then_some(event.actor_seat),
            target_seat: (event.target_seat != 255).then_some(event.target_seat),
            visibility_mask: event.visibility_mask,
            args: event.args.into(),
            payload: event.payload,
        })
        .collect();
    PyTransition {
        transition_id: value.transition_id,
        states,
        events,
        validation_ns: value.validation_ns,
        env_step_ns: value.env_step_ns,
        materialization_ns: value.materialization_ns,
        exchange_ns: 0,
        projection: PyOnceLock::new(),
    }
}

fn materialize_state(state: &GameState, privileged: bool) -> PyState {
    let game = state.hanchan.as_ref();
    let frame = game.and_then(|value| value.hand.decision_frame.as_ref());
    let decisions = frame
        .into_iter()
        .flat_map(|value| value.decisions.iter())
        .map(|decision| {
            let frame = frame.expect("decision requires frame");
            let player = &game.expect("frame requires game").players[decision.seat as usize];
            PyDecision {
                environment_id: state.environment_id,
                episode_generation: state.episode_generation,
                frame_id: frame.frame_id,
                seat: decision.seat,
                flags: decision_flags(player),
                current_draw: (game.expect("decision requires game").hand.current_seat
                    == decision.seat)
                    .then_some(game.expect("decision requires game").hand.current_draw)
                    .filter(|tile| *tile != 255),
                concealed_counts: tile_type_counts(&player.concealed_tiles),
                actions: decision
                    .actions
                    .iter()
                    .enumerate()
                    .map(|(index, descriptor)| PyAction {
                        inner: CoreAction::bind(frame, decision.seat, index as u32, descriptor),
                    })
                    .collect(),
            }
        })
        .collect();
    let melds = game
        .into_iter()
        .flat_map(|value| value.players.iter())
        .flat_map(|player| {
            player.melds.iter().map(move |meld| PyMeld {
                seat: player.seat,
                kind: meld.kind as u8,
                from_seat: (meld.from_seat != 255).then_some(meld.from_seat),
                called_tile: (meld.called_tile != 255).then_some(meld.called_tile),
                tiles: meld.tiles[..usize::from(meld.tile_count)].to_vec(),
                created_sequence: meld.created_sequence,
            })
        })
        .collect();
    let rivers = game
        .into_iter()
        .flat_map(|value| value.players.iter())
        .flat_map(|player| {
            player.river.iter().map(move |tile| PyRiverTile {
                seat: player.seat,
                tile: tile.tile,
                sequence: tile.sequence,
                riichi_declaration: tile.riichi_declaration,
                called: tile.called,
                tsumogiri: tile.tsumogiri,
            })
        })
        .collect();
    let scores = game.map_or([0; 4], |value| value.scores);
    let flags = game.map_or([0; 4], |value| {
        std::array::from_fn(|seat| decision_flags(&value.players[seat]))
    });
    let hidden = privileged.then(|| {
        let game = game.expect("privileged initialized state");
        PyHiddenState {
            concealed_counts: game
                .players
                .iter()
                .map(|player| tile_type_counts(&player.concealed_tiles))
                .collect(),
            concealed_tile_ids: game
                .players
                .iter()
                .map(|player| player.concealed_tiles.clone())
                .collect(),
            live_wall_counts: game.hand.wall.live_wall_counts,
            wall: game.hand.wall.tiles,
            wall_indices: [
                game.hand.wall.live_start,
                game.hand.wall.live_end,
                game.hand.wall.rinshan_index,
                game.hand.wall.dora_indicator_count,
            ],
            ura_indicators: game.hand.wall.ura_indicators
                [..usize::from(game.hand.wall.dora_indicator_count)]
                .to_vec(),
        }
    });
    PyState {
        environment_id: state.environment_id,
        episode_generation: state.episode_generation,
        frame_id: frame.map_or(0, |value| value.frame_id),
        phase: game.map_or(0, |value| value.hand.phase as u8),
        eligible_mask: frame.map_or(0, |value| value.eligible_mask),
        lifecycle: state.lifecycle as u8,
        status: 0,
        error_code: 0,
        error_args: (0, 0),
        round_wind: game.map_or(0, |value| value.round_wind as u8),
        hand_number: game.map_or(0, |value| value.hand_number),
        dealer: game.map_or(255, |value| value.dealer),
        honba: game.map_or(0, |value| value.honba),
        riichi_deposits: game.map_or(0, |value| value.riichi_deposits),
        live_wall_remaining: game.map_or(0, |value| {
            value
                .hand
                .wall
                .live_end
                .saturating_sub(value.hand.wall.live_start)
        }),
        scores: scores.into(),
        seat_flags: flags.into(),
        dora_indicators: game.map_or_else(Vec::new, |value| {
            value.hand.wall.revealed_dora_indicators
                [..usize::from(value.hand.wall.dora_indicator_count)]
                .to_vec()
        }),
        decisions,
        melds,
        rivers,
        hidden,
    }
}

fn decision_flags(player: &riichi_core::game::state::PlayerState) -> u32 {
    u32::from(player.riichi_state == RiichiState::Declared)
        | (u32::from(player.riichi_state == RiichiState::Accepted) << 1)
        | (u32::from(player.ippatsu_eligible) << 2)
        | (u32::from(player.permanent_furiten) << 3)
        | (u32::from(player.temporary_furiten) << 4)
        | (u32::from(player.riichi_furiten) << 5)
}
