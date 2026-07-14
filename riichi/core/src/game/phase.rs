use serde::{Deserialize, Serialize};

#[repr(u8)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub enum EnvironmentLifecycle {
    #[default]
    Uninitialized = 0,
    Ready = 1,
    Running = 2,
    Complete = 3,
    Failed = 4,
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub enum HandPhase {
    #[default]
    Setup = 0,
    SelfTurnDecision = 1,
    DiscardReactionFrame = 2,
    KanRobReactionFrame = 3,
    ReplacementTurn = 4,
    Settlement = 5,
    HandComplete = 6,
    HanchanComplete = 7,
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub enum Wind {
    #[default]
    East = 0,
    South = 1,
    West = 2,
    North = 3,
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum TerminalReason {
    Ron = 1,
    Tsumo = 2,
    ExhaustiveDraw = 3,
    AbortiveDraw = 4,
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum MeldKind {
    Chi = 1,
    Pon = 2,
    OpenKan = 3,
    ClosedKan = 4,
    AddedKan = 5,
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub enum RiichiState {
    #[default]
    None = 0,
    Declared = 1,
    Accepted = 2,
}
