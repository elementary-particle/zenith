use serde::{Deserialize, Serialize};

#[repr(u16)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub enum FrameStatus {
    #[default]
    Ok = 0,
    InvalidAction = 1,
    StaleFrame = 2,
    IncompleteActionSet = 3,
    DuplicateSeat = 4,
    IneligibleSeat = 5,
    EnvironmentFailed = 6,
    InternalFailure = 7,
}

#[repr(u16)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum ErrorCode {
    None = 0,
    EnvironmentOutOfRange = 1,
    EpisodeGenerationMismatch = 2,
    FrameIdMismatch = 3,
    MissingSeat = 4,
    DuplicateSeat = 5,
    IneligibleSeat = 6,
    ActionOutOfRange = 7,
    EnvironmentNotReady = 8,
    InternalPanic = 9,
    SnapshotInvalid = 10,
    SnapshotVersionMismatch = 11,
    EnvClosed = 12,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct FailureRecord {
    pub code: ErrorCode,
    pub arg0: i64,
    pub arg1: i64,
}

#[derive(Debug, thiserror::Error)]
pub enum CoreError {
    #[error("env is closed")]
    Closed,
    #[error("environment id {0} is out of range")]
    EnvironmentOutOfRange(u32),
    #[error("duplicate environment id {0}")]
    DuplicateEnvironment(u32),
    #[error("invalid rules profile: {0}")]
    InvalidProfile(String),
    #[error("invalid argument: {0}")]
    InvalidArgument(String),
    #[error("invalid action set: {status:?}/{code:?} ({arg0}, {arg1})")]
    InvalidActions {
        status: FrameStatus,
        code: ErrorCode,
        arg0: i64,
        arg1: i64,
    },
    #[error("snapshot is invalid: {0}")]
    Snapshot(String),
}
