//! State-only riichi rules boundary.
//!
//! It deliberately has no Python, NumPy, batching, threading, or training dependency.

pub mod error;
mod evaluator;
pub mod game;
pub mod snapshot;

pub use error::{ErrorCode, FrameStatus};
pub use game::state::GameState;
pub use game::state::{derive_rng, RngState};
pub use game::{
    action::{ActionCandidate, ActionKind, ActionSelection, ActionSpace, Decision},
    event::{EventKind, MJAI_EVENT_NAMES},
    replay::{ReplayEvent, ReplayHanchan},
    rules::profile::{
        DECISION_SCHEMA_VERSION, EVENT_SCHEMA_VERSION, HAND_ANALYSIS_VERSION, RNG_PROFILE,
        RNG_PROFILE_ID, RULES_PROFILE, RULES_PROFILE_ID, SNAPSHOT_SCHEMA_VERSION,
        STATE_SCHEMA_VERSION, TENHOU_RULES_PROFILE, TENHOU_RULES_PROFILE_ID,
    },
    search::{
        fork_with_privileged_wall, fork_with_public_information, PrivilegedWallParticle,
        PublicInformationParticle, SearchError,
    },
};

pub const ABSENT_SENTINEL: u8 = 255;
pub const SHANTEN_UNAVAILABLE: i8 = 127;
