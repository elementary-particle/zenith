use riichi_core::{
    snapshot, ActionSelection, EventKind, GameState, EVENT_SCHEMA_VERSION, STATE_SCHEMA_VERSION,
};

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
fn reset_and_step_emit_gap_free_mjai_events_and_restore_exactly() {
    assert_eq!((STATE_SCHEMA_VERSION, EVENT_SCHEMA_VERSION), (5, 2));
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
    for _ in 0..10_000 {
        let actions = first_per_seat(&ordered);
        if actions.len() >= 2 {
            let checkpoint = snapshot::encode(&ordered).unwrap();
            let mut reversed = snapshot::decode(&checkpoint).unwrap();
            let mut reverse_actions = actions.clone();
            reverse_actions.reverse();
            ordered.step(&actions).unwrap();
            reversed.step(&reverse_actions).unwrap();
            assert_eq!(
                snapshot::encode(&ordered).unwrap(),
                snapshot::encode(&reversed).unwrap()
            );
            return;
        }
        advance_once(&mut ordered);
    }
    panic!("deterministic match did not expose a simultaneous reaction");
}
