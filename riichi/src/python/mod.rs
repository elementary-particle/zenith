mod analysis;
mod env;
pub(crate) mod projection;
mod rollout;
mod types;

use pyo3::exceptions::PyOSError;
use pyo3::prelude::*;

use crate::{
    ABSENT_SENTINEL, DECISION_SCHEMA_VERSION, EVENT_SCHEMA_VERSION, HAND_ANALYSIS_VERSION,
    MJAI_EVENT_NAMES, RNG_PROFILE, RNG_PROFILE_ID, RULES_PROFILE, RULES_PROFILE_ID,
    SHANTEN_UNAVAILABLE, SNAPSHOT_SCHEMA_VERSION, STATE_SCHEMA_VERSION, TENHOU_RULES_PROFILE,
    TENHOU_RULES_PROFILE_ID,
};

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    analysis::register(module)?;
    rollout::register(module)?;
    module.add_class::<types::PyActionKind>()?;
    module.add_class::<types::PyActionSelection>()?;
    module.add_class::<types::PyActionCandidate>()?;
    module.add_class::<types::PyReplayHanchan>()?;
    module.add_class::<types::PyReplayEvent>()?;
    module.add_class::<types::PyActionSpace>()?;
    module.add_class::<types::PyMeld>()?;
    module.add_class::<types::PyRiverTile>()?;
    module.add_class::<types::PyHiddenState>()?;
    module.add_class::<types::PyState>()?;
    module.add_class::<types::PyEvent>()?;
    module.add_class::<types::PyTransition>()?;
    module.add_class::<env::PyEnv>()?;
    module.add_function(wrap_pyfunction!(ensure_shanten_cache, module)?)?;
    module.add("STATE_SCHEMA_VERSION", STATE_SCHEMA_VERSION)?;
    module.add("EVENT_SCHEMA_VERSION", EVENT_SCHEMA_VERSION)?;
    module.add("DECISION_SCHEMA_VERSION", DECISION_SCHEMA_VERSION)?;
    module.add("HAND_ANALYSIS_VERSION", HAND_ANALYSIS_VERSION)?;
    module.add("SNAPSHOT_SCHEMA_VERSION", SNAPSHOT_SCHEMA_VERSION)?;
    module.add("RULES_PROFILE", RULES_PROFILE)?;
    module.add("RULES_PROFILE_ID", RULES_PROFILE_ID)?;
    module.add("TENHOU_RULES_PROFILE", TENHOU_RULES_PROFILE)?;
    module.add("TENHOU_RULES_PROFILE_ID", TENHOU_RULES_PROFILE_ID)?;
    module.add("RNG_PROFILE", RNG_PROFILE)?;
    module.add("RNG_PROFILE_ID", RNG_PROFILE_ID)?;
    module.add("ABSENT_SENTINEL", ABSENT_SENTINEL)?;
    module.add("SHANTEN_UNAVAILABLE", SHANTEN_UNAVAILABLE)?;
    module.add(
        "NATIVE_BUILD_PROFILE",
        if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        },
    )?;
    module.add("MJAI_EVENT_NAMES", MJAI_EVENT_NAMES)?;
    module.add(
        "FRAME_STATUS_CODES",
        [
            ("OK", 0_u16),
            ("INVALID_ACTION", 1),
            ("STALE_FRAME", 2),
            ("INCOMPLETE_ACTION_SET", 3),
            ("DUPLICATE_SEAT", 4),
            ("INELIGIBLE_SEAT", 5),
            ("ENVIRONMENT_FAILED", 6),
            ("INTERNAL_FAILURE", 7),
        ],
    )?;
    module.add(
        "ACTION_KIND_CODES",
        [
            ("PASS", 0_u8),
            ("DISCARD", 1),
            ("RIICHI_DISCARD", 2),
            ("CHI", 3),
            ("PON", 4),
            ("OPEN_KAN", 5),
            ("CLOSED_KAN", 6),
            ("ADDED_KAN", 7),
            ("RON", 8),
            ("TSUMO", 9),
            ("ABORTIVE_DECLARATION", 10),
        ],
    )?;
    Ok(())
}

/// Prewarms the shared shanten cache before rollout workers are created.
#[pyfunction]
fn ensure_shanten_cache() -> PyResult<String> {
    riichi_core::game::rules::shanten::ensure_cache()
        .map(|path| path.to_string_lossy().into_owned())
        .map_err(|error| PyOSError::new_err(error.to_string()))
}
