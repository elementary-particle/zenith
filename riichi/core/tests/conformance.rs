use riichi_core::{
    snapshot, Action, EventKind, GameState, EVENT_SCHEMA_VERSION, STATE_SCHEMA_VERSION,
};

fn first_per_seat(state: &GameState) -> Vec<Action> {
    let mut actions = state.legal_actions();
    actions.sort_by_key(|action| (action.seat, action.action_index));
    actions.dedup_by_key(|action| action.seat);
    actions
}

#[test]
fn reset_and_step_emit_gap_free_mjai_events_and_restore_exactly() {
    assert_eq!((STATE_SCHEMA_VERSION, EVENT_SCHEMA_VERSION), (2, 2));
    let mut state = GameState::new(3);
    state.reset_from_seed(11);
    let initial = state.take_events();
    assert_eq!(initial.first().unwrap().kind, EventKind::StartGame);
    assert!(initial
        .windows(2)
        .all(|pair| pair[1].sequence == pair[0].sequence + 1));

    state.step(&first_per_seat(&state)).unwrap();
    let delta = state.take_events();
    assert!(delta.iter().all(|event| !event.kind.mjai_name().is_empty()));
    let bytes = snapshot::encode(&state).unwrap();
    let restored = snapshot::decode(&bytes).unwrap();
    assert_eq!(snapshot::encode(&restored).unwrap(), bytes);
}

#[test]
fn simultaneous_reaction_submission_is_order_independent() {
    let mut ordered = GameState::new(0);
    ordered.reset_from_seed(19);
    ordered.take_events();
    ordered.step(&first_per_seat(&ordered)).unwrap();
    let checkpoint = snapshot::encode(&ordered).unwrap();
    let mut reversed = snapshot::decode(&checkpoint).unwrap();
    let actions = first_per_seat(&ordered);
    let mut reverse_actions = actions.clone();
    reverse_actions.reverse();
    ordered.step(&actions).unwrap();
    reversed.step(&reverse_actions).unwrap();
    assert_eq!(
        snapshot::encode(&ordered).unwrap(),
        snapshot::encode(&reversed).unwrap()
    );
}
