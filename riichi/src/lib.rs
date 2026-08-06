pub mod batch_env;
mod python;
pub mod rollout;
pub use batch_env::{BatchEnv, BatchTransition, EnvError};
pub use riichi_core::{
    ActionCandidate, ActionKind, ActionSelection, EventKind, GameState, ReplayEvent, ReplayHanchan,
    DECISION_SCHEMA_VERSION, EVENT_SCHEMA_VERSION, HAND_ANALYSIS_VERSION, MJAI_EVENT_NAMES,
    RNG_PROFILE, RNG_PROFILE_ID, RULES_PROFILE, RULES_PROFILE_ID, SNAPSHOT_SCHEMA_VERSION,
    STATE_SCHEMA_VERSION, TENHOU_RULES_PROFILE, TENHOU_RULES_PROFILE_ID,
};
pub use rollout::{InferenceRequest, RolloutChunk, RolloutEngine};

pub const ABSENT_SENTINEL: u8 = 255;
pub const SHANTEN_UNAVAILABLE: i8 = 127;

#[pyo3::pymodule]
fn _riichi(module: &pyo3::Bound<'_, pyo3::types::PyModule>) -> pyo3::PyResult<()> {
    python::register(module)
}
