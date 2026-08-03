use numpy::{
    ndarray::{Array2, Array3},
    Element, IntoPyArray, PyArray1,
};
use pyo3::{prelude::*, types::PyDict};

use super::types::PyTransition;

pub fn as_numpy<'py>(transition: &PyTransition, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
    transition
        .projection
        .get_or_try_init(py, || build(transition, py).map(Bound::unbind))
        .map(|value| value.bind(py).clone())
}

fn build<'py>(transition: &PyTransition, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
    let result = PyDict::new(py);
    let states = &transition.states;

    insert_1(
        &result,
        py,
        "state_environment_id",
        states.iter().map(|state| state.environment_id).collect(),
    )?;
    insert_1(
        &result,
        py,
        "state_episode_generation",
        states
            .iter()
            .map(|state| state.episode_generation)
            .collect(),
    )?;
    insert_1(
        &result,
        py,
        "state_frame_id",
        states.iter().map(|state| state.frame_id).collect(),
    )?;
    insert_1(
        &result,
        py,
        "state_phase",
        states.iter().map(|state| state.phase).collect(),
    )?;
    insert_1(
        &result,
        py,
        "state_eligible_mask",
        states.iter().map(|state| state.eligible_mask).collect(),
    )?;
    insert_1(
        &result,
        py,
        "state_lifecycle",
        states.iter().map(|state| state.lifecycle).collect(),
    )?;
    insert_1(
        &result,
        py,
        "state_status",
        states.iter().map(|state| state.status).collect(),
    )?;
    insert_1(
        &result,
        py,
        "state_error_code",
        states.iter().map(|state| state.error_code).collect(),
    )?;
    insert_2(
        &result,
        py,
        "state_error_args",
        states.len(),
        2,
        states
            .iter()
            .flat_map(|state| [state.error_args.0, state.error_args.1])
            .collect(),
    )?;
    for (name, values) in [
        (
            "state_round_wind",
            states.iter().map(|state| state.round_wind).collect(),
        ),
        (
            "state_hand_number",
            states.iter().map(|state| state.hand_number).collect(),
        ),
        (
            "state_dealer",
            states.iter().map(|state| state.dealer).collect(),
        ),
        (
            "state_live_wall_remaining",
            states
                .iter()
                .map(|state| state.live_wall_remaining)
                .collect(),
        ),
    ] {
        insert_1(&result, py, name, values)?;
    }
    insert_1(
        &result,
        py,
        "state_honba",
        states.iter().map(|state| state.honba).collect(),
    )?;
    insert_1(
        &result,
        py,
        "state_riichi_deposits",
        states.iter().map(|state| state.riichi_deposits).collect(),
    )?;
    insert_2(
        &result,
        py,
        "state_scores",
        states.len(),
        4,
        states
            .iter()
            .flat_map(|state| {
                [
                    state.scores.0,
                    state.scores.1,
                    state.scores.2,
                    state.scores.3,
                ]
            })
            .collect(),
    )?;
    insert_2(
        &result,
        py,
        "state_seat_flags",
        states.len(),
        4,
        states
            .iter()
            .flat_map(|state| {
                [
                    state.seat_flags.0,
                    state.seat_flags.1,
                    state.seat_flags.2,
                    state.seat_flags.3,
                ]
            })
            .collect(),
    )?;

    let (action_space_offsets, action_spaces) =
        flatten_with_offsets(states.iter().map(|state| &state.action_spaces));
    let (meld_offsets, melds) = flatten_with_offsets(states.iter().map(|state| &state.melds));
    let (river_offsets, rivers) = flatten_with_offsets(states.iter().map(|state| &state.rivers));
    let (dora_offsets, dora) =
        flatten_copy_with_offsets(states.iter().map(|state| &state.dora_indicators));
    insert_1(
        &result,
        py,
        "state_action_space_offsets",
        action_space_offsets,
    )?;
    insert_1(&result, py, "state_meld_offsets", meld_offsets)?;
    insert_1(&result, py, "state_river_offsets", river_offsets)?;
    insert_1(&result, py, "state_dora_offsets", dora_offsets)?;
    insert_1(&result, py, "state_dora_indicators", dora)?;

    insert_1(
        &result,
        py,
        "action_space_environment_id",
        action_spaces
            .iter()
            .map(|value| value.environment_id)
            .collect(),
    )?;
    insert_1(
        &result,
        py,
        "action_space_episode_generation",
        action_spaces
            .iter()
            .map(|value| value.episode_generation)
            .collect(),
    )?;
    insert_1(
        &result,
        py,
        "action_space_frame_id",
        action_spaces.iter().map(|value| value.frame_id).collect(),
    )?;
    insert_1(
        &result,
        py,
        "action_space_seat",
        action_spaces.iter().map(|value| value.seat).collect(),
    )?;
    insert_1(
        &result,
        py,
        "action_space_flags",
        action_spaces.iter().map(|value| value.flags).collect(),
    )?;
    insert_1(
        &result,
        py,
        "action_space_current_draw",
        action_spaces
            .iter()
            .map(|value| value.current_draw.unwrap_or(255))
            .collect(),
    )?;
    insert_2(
        &result,
        py,
        "action_space_concealed_counts",
        action_spaces.len(),
        34,
        action_spaces
            .iter()
            .flat_map(|value| value.concealed_counts)
            .collect(),
    )?;
    let (candidate_offsets, candidates) =
        flatten_with_offsets(action_spaces.iter().map(|space| &space.candidates));
    insert_1(
        &result,
        py,
        "action_space_candidate_offsets",
        candidate_offsets,
    )?;

    insert_1(
        &result,
        py,
        "candidate_environment_id",
        candidates
            .iter()
            .map(|value| value.selection.environment_id)
            .collect(),
    )?;
    insert_1(
        &result,
        py,
        "candidate_episode_generation",
        candidates
            .iter()
            .map(|value| value.selection.episode_generation)
            .collect(),
    )?;
    insert_1(
        &result,
        py,
        "candidate_frame_id",
        candidates
            .iter()
            .map(|value| value.selection.frame_id)
            .collect(),
    )?;
    insert_1(
        &result,
        py,
        "candidate_seat",
        candidates
            .iter()
            .map(|value| value.selection.seat)
            .collect(),
    )?;
    insert_1(
        &result,
        py,
        "candidate_index",
        candidates
            .iter()
            .map(|value| value.selection.candidate_index)
            .collect(),
    )?;
    for (name, values) in [
        (
            "candidate_kind",
            candidates
                .iter()
                .map(|value| value.candidate.kind as u8)
                .collect(),
        ),
        (
            "candidate_primary_tile_type",
            candidates
                .iter()
                .map(|value| value.candidate.primary_tile_type)
                .collect(),
        ),
        (
            "candidate_source_seat",
            candidates
                .iter()
                .map(|value| value.candidate.source_seat)
                .collect(),
        ),
        (
            "candidate_tile_count",
            candidates
                .iter()
                .map(|value| value.candidate.tile_count)
                .collect(),
        ),
    ] {
        insert_1(&result, py, name, values)?;
    }
    insert_2(
        &result,
        py,
        "candidate_tiles",
        candidates.len(),
        4,
        candidates
            .iter()
            .flat_map(|value| value.candidate.tiles)
            .collect(),
    )?;
    insert_1(
        &result,
        py,
        "candidate_aux",
        candidates.iter().map(|value| value.candidate.aux).collect(),
    )?;
    insert_1(
        &result,
        py,
        "candidate_flags",
        candidates
            .iter()
            .map(|value| value.candidate.flags)
            .collect(),
    )?;

    let applied = &transition.applied_selections;
    insert_1(
        &result,
        py,
        "applied_selection_environment_id",
        applied
            .iter()
            .map(|value| value.inner.environment_id)
            .collect(),
    )?;
    insert_1(
        &result,
        py,
        "applied_selection_episode_generation",
        applied
            .iter()
            .map(|value| value.inner.episode_generation)
            .collect(),
    )?;
    insert_1(
        &result,
        py,
        "applied_selection_frame_id",
        applied.iter().map(|value| value.inner.frame_id).collect(),
    )?;
    insert_1(
        &result,
        py,
        "applied_selection_seat",
        applied.iter().map(|value| value.inner.seat).collect(),
    )?;
    insert_1(
        &result,
        py,
        "applied_candidate_index",
        applied
            .iter()
            .map(|value| value.inner.candidate_index)
            .collect(),
    )?;

    insert_1(
        &result,
        py,
        "meld_seat",
        melds.iter().map(|value| value.seat).collect(),
    )?;
    insert_1(
        &result,
        py,
        "meld_kind",
        melds.iter().map(|value| value.kind).collect(),
    )?;
    insert_1(
        &result,
        py,
        "meld_from_seat",
        melds
            .iter()
            .map(|value| value.from_seat.unwrap_or(255))
            .collect(),
    )?;
    insert_1(
        &result,
        py,
        "meld_called_tile",
        melds
            .iter()
            .map(|value| value.called_tile.unwrap_or(255))
            .collect(),
    )?;
    insert_1(
        &result,
        py,
        "meld_created_sequence",
        melds.iter().map(|value| value.created_sequence).collect(),
    )?;
    insert_1(
        &result,
        py,
        "meld_tile_count",
        melds.iter().map(|value| value.tiles.len() as u8).collect(),
    )?;
    insert_2(
        &result,
        py,
        "meld_tiles",
        melds.len(),
        4,
        melds
            .iter()
            .flat_map(|value| padded::<4>(&value.tiles))
            .collect(),
    )?;

    insert_1(
        &result,
        py,
        "river_seat",
        rivers.iter().map(|value| value.seat).collect(),
    )?;
    insert_1(
        &result,
        py,
        "river_tile",
        rivers.iter().map(|value| value.tile).collect(),
    )?;
    insert_1(
        &result,
        py,
        "river_sequence",
        rivers.iter().map(|value| value.sequence).collect(),
    )?;
    insert_1(
        &result,
        py,
        "river_flags",
        rivers
            .iter()
            .map(|value| {
                u8::from(value.riichi_declaration)
                    | (u8::from(value.called) << 1)
                    | (u8::from(value.tsumogiri) << 2)
            })
            .collect(),
    )?;

    let events = &transition.events;
    insert_1(
        &result,
        py,
        "event_environment_id",
        events.iter().map(|value| value.environment_id).collect(),
    )?;
    insert_1(
        &result,
        py,
        "event_episode_generation",
        events
            .iter()
            .map(|value| value.episode_generation)
            .collect(),
    )?;
    insert_1(
        &result,
        py,
        "event_sequence",
        events.iter().map(|value| value.sequence).collect(),
    )?;
    insert_1(
        &result,
        py,
        "event_kind",
        events.iter().map(|value| value.kind).collect(),
    )?;
    insert_1(
        &result,
        py,
        "event_actor_seat",
        events
            .iter()
            .map(|value| value.actor_seat.unwrap_or(255))
            .collect(),
    )?;
    insert_1(
        &result,
        py,
        "event_target_seat",
        events
            .iter()
            .map(|value| value.target_seat.unwrap_or(255))
            .collect(),
    )?;
    insert_1(
        &result,
        py,
        "event_visibility_mask",
        events.iter().map(|value| value.visibility_mask).collect(),
    )?;
    insert_2(
        &result,
        py,
        "event_args",
        events.len(),
        4,
        events
            .iter()
            .flat_map(|value| [value.args.0, value.args.1, value.args.2, value.args.3])
            .collect(),
    )?;
    let (payload_offsets, payload) =
        flatten_copy_with_offsets(events.iter().map(|event| &event.payload));
    let payload_lengths = payload_offsets
        .windows(2)
        .map(|pair| (pair[1] - pair[0]) as u32)
        .collect();
    insert_1(&result, py, "event_payload_offsets", payload_offsets)?;
    insert_1(&result, py, "event_payload_lengths", payload_lengths)?;
    insert_1(&result, py, "event_payload", payload)?;

    if states.iter().all(|state| state.hidden.is_some()) {
        let hidden = states
            .iter()
            .map(|state| state.hidden.as_ref().expect("checked"))
            .collect::<Vec<_>>();
        insert_3(
            &result,
            py,
            "hidden_concealed_counts",
            hidden.len(),
            4,
            34,
            hidden
                .iter()
                .flat_map(|value| value.concealed_counts.iter().flatten().copied())
                .collect(),
        )?;
        insert_3(
            &result,
            py,
            "hidden_concealed_tile_ids",
            hidden.len(),
            4,
            14,
            hidden
                .iter()
                .flat_map(|value| {
                    value
                        .concealed_tile_ids
                        .iter()
                        .flat_map(|tiles| padded::<14>(tiles))
                })
                .collect(),
        )?;
        insert_2(
            &result,
            py,
            "hidden_live_wall_counts",
            hidden.len(),
            34,
            hidden
                .iter()
                .flat_map(|value| value.live_wall_counts)
                .collect(),
        )?;
        insert_2(
            &result,
            py,
            "hidden_wall",
            hidden.len(),
            136,
            hidden.iter().flat_map(|value| value.wall).collect(),
        )?;
        insert_2(
            &result,
            py,
            "hidden_wall_indices",
            hidden.len(),
            4,
            hidden.iter().flat_map(|value| value.wall_indices).collect(),
        )?;
        insert_2(
            &result,
            py,
            "hidden_ura_indicators",
            hidden.len(),
            5,
            hidden
                .iter()
                .flat_map(|value| padded::<5>(&value.ura_indicators))
                .collect(),
        )?;
    }
    Ok(result)
}

fn insert_1<T: Element>(
    dict: &Bound<'_, PyDict>,
    py: Python<'_>,
    name: &str,
    values: Vec<T>,
) -> PyResult<()> {
    let array = PyArray1::from_vec(py, values);
    array.call_method1("setflags", (false,))?;
    dict.set_item(name, array)
}

fn insert_2<T: Element>(
    dict: &Bound<'_, PyDict>,
    py: Python<'_>,
    name: &str,
    rows: usize,
    columns: usize,
    values: Vec<T>,
) -> PyResult<()> {
    let array = Array2::from_shape_vec((rows, columns), values)
        .expect("projection shape")
        .into_pyarray(py);
    array.call_method1("setflags", (false,))?;
    dict.set_item(name, array)
}

#[allow(clippy::too_many_arguments)]
fn insert_3<T: Element>(
    dict: &Bound<'_, PyDict>,
    py: Python<'_>,
    name: &str,
    first: usize,
    second: usize,
    third: usize,
    values: Vec<T>,
) -> PyResult<()> {
    let array = Array3::from_shape_vec((first, second, third), values)
        .expect("projection shape")
        .into_pyarray(py);
    array.call_method1("setflags", (false,))?;
    dict.set_item(name, array)
}

fn flatten_with_offsets<'a, T: 'a>(
    values: impl Iterator<Item = &'a Vec<T>>,
) -> (Vec<u64>, Vec<&'a T>) {
    let mut offsets = vec![0];
    let mut flattened = Vec::new();
    for value in values {
        flattened.extend(value);
        offsets.push(flattened.len() as u64);
    }
    (offsets, flattened)
}

fn flatten_copy_with_offsets<'a, T: Copy + 'a>(
    values: impl Iterator<Item = &'a Vec<T>>,
) -> (Vec<u64>, Vec<T>) {
    let mut offsets = vec![0];
    let mut flattened = Vec::new();
    for value in values {
        flattened.extend_from_slice(value);
        offsets.push(flattened.len() as u64);
    }
    (offsets, flattened)
}

fn padded<const N: usize>(values: &[u8]) -> [u8; N] {
    let mut result = [255; N];
    result[..values.len().min(N)].copy_from_slice(&values[..values.len().min(N)]);
    result
}
