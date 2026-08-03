use riichi_core::{snapshot, ActionSelection, ErrorCode, FrameStatus, GameState};

fn first_per_seat(state: &GameState) -> Vec<ActionSelection> {
    let mut actions = state.legal_selections();
    actions.sort_by_key(|selection| (selection.seat, selection.candidate_index));
    actions.dedup_by_key(|action| action.seat);
    actions
}

fn advance_once(state: &mut GameState) {
    let actions = first_per_seat(state);
    if actions.is_empty() {
        assert!(state.advance_automatic_once());
    } else {
        state.step(&actions).unwrap();
    }
}

#[test]
fn seeded_single_game_transitions_are_deterministic_and_legal() {
    let mut left = GameState::new(7);
    let mut right = GameState::new(7);
    left.reset_from_seed(0x5eed);
    right.reset_from_seed(0x5eed);
    assert_eq!(
        snapshot::encode(&left).unwrap(),
        snapshot::encode(&right).unwrap()
    );

    for _ in 0..32 {
        let left_actions = first_per_seat(&left);
        let right_actions = first_per_seat(&right);
        assert_eq!(left_actions, right_actions);
        if left_actions.is_empty() {
            assert!(left.advance_automatic_once());
            assert!(right.advance_automatic_once());
        } else {
            left.step(&left_actions).unwrap();
            right.step(&right_actions).unwrap();
        }
        assert_eq!(
            snapshot::encode(&left).unwrap(),
            snapshot::encode(&right).unwrap()
        );
        assert!(left.episode_generation > 0);
        assert!(left
            .hanchan
            .as_ref()
            .unwrap()
            .scores
            .iter()
            .all(|score| score.abs() < 1_000_000));
    }
}

#[test]
fn invalid_bound_action_is_rejected_without_mutation() {
    let mut state = GameState::new(0);
    state.reset_from_seed(1);
    let before = snapshot::encode(&state).unwrap();
    let mut invalid = first_per_seat(&state);
    invalid[0].frame_id = invalid[0].frame_id.wrapping_add(1);
    let error = state.step(&invalid).unwrap_err();
    match error {
        riichi_core::error::CoreError::InvalidActions { status, code, .. } => {
            assert_eq!(status, FrameStatus::StaleFrame);
            assert_eq!(code, ErrorCode::FrameIdMismatch);
        }
        other => panic!("unexpected error: {other}"),
    }
    assert_eq!(snapshot::encode(&state).unwrap(), before);
}

#[test]
fn end_game_reports_the_rust_maintained_kyoku_count() {
    let mut state = GameState::new(0);
    state.reset_from_seed(37);
    state.take_events();
    let mut observed_end_kyoku = 0_u64;

    for _ in 0..10_000 {
        advance_once(&mut state);
        for event in state.take_events() {
            if event.kind == riichi_core::EventKind::EndKyoku {
                observed_end_kyoku += 1;
            } else if event.kind == riichi_core::EventKind::EndGame {
                assert!(observed_end_kyoku > 0);
                assert_eq!(event.args[0], observed_end_kyoku as i64);
                assert_eq!(
                    state.hanchan.as_ref().unwrap().completed_kyoku,
                    observed_end_kyoku as u32,
                );
                return;
            }
        }
    }
    panic!("deterministic match did not complete");
}

#[test]
fn snapshot_extensions_are_backward_compatible() {
    let mut state = GameState::new(1);
    state.reset_from_seed(41);
    state.hanchan.as_mut().unwrap().completed_kyoku = 7;

    let bytes = snapshot::encode(&state).unwrap();
    let restored = snapshot::decode(&bytes).unwrap();
    assert_eq!(restored.hanchan.unwrap().completed_kyoku, 7);

    let mut previous = bytes.clone();
    previous.truncate(previous.len() - 1);
    let previous_body_len = (previous.len() - snapshot::SNAPSHOT_HEADER_BYTES) as u64;
    previous[40..48].copy_from_slice(&previous_body_len.to_le_bytes());
    let restored_previous = snapshot::decode(&previous).unwrap();
    assert_eq!(restored_previous.hanchan.unwrap().completed_kyoku, 7);

    let mut legacy = bytes;
    legacy.truncate(legacy.len() - 6);
    let legacy_body_len = (legacy.len() - snapshot::SNAPSHOT_HEADER_BYTES) as u64;
    legacy[40..48].copy_from_slice(&legacy_body_len.to_le_bytes());
    let restored_legacy = snapshot::decode(&legacy).unwrap();
    assert_eq!(restored_legacy.hanchan.unwrap().completed_kyoku, 0);
}
