use riichi_core::{snapshot, Action, ErrorCode, FrameStatus, GameState};

fn first_per_seat(state: &GameState) -> Vec<Action> {
    let mut actions = state.legal_actions();
    actions.sort_by_key(|action| (action.seat, action.action_index));
    actions.dedup_by_key(|action| action.seat);
    actions
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
        assert!(!left_actions.is_empty());
        left.step(&left_actions).unwrap();
        right.step(&right_actions).unwrap();
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
