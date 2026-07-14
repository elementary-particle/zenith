use riichi::{
    ActionKind, EventKind, ABSENT_SENTINEL, EVENT_SCHEMA_VERSION, HAND_ANALYSIS_VERSION,
    RNG_PROFILE_ID, RULES_PROFILE_ID, SHANTEN_UNAVAILABLE, SNAPSHOT_SCHEMA_VERSION,
    STATE_SCHEMA_VERSION,
};

#[test]
fn stable_numeric_contract() {
    assert_eq!(STATE_SCHEMA_VERSION, 2);
    assert_eq!(EVENT_SCHEMA_VERSION, 2);
    assert_eq!(HAND_ANALYSIS_VERSION, 2);
    assert_eq!(SNAPSHOT_SCHEMA_VERSION, 1);
    assert_eq!(RULES_PROFILE_ID, 2);
    assert_eq!(RNG_PROFILE_ID, 1);
    assert_eq!(ABSENT_SENTINEL, 255);
    assert_eq!(SHANTEN_UNAVAILABLE, 127);
    assert_eq!(ActionKind::Pass as u8, 0);
    assert_eq!(ActionKind::AbortiveDeclaration as u8, 10);
    assert_eq!(EventKind::StartGame as u16, 1);
    assert_eq!(EventKind::StartKyoku as u16, 2);
    assert_eq!(EventKind::Tsumo as u16, 3);
    assert_eq!(EventKind::Dahai as u16, 4);
    assert_eq!(EventKind::EndGame as u16, 16);
}
