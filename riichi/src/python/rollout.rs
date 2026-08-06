use std::collections::BTreeMap;

use numpy::{
    ndarray::{Array2, Array3},
    Element, IntoPyArray, PyArray1, PyArray2, PyArray3,
};
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
    types::PyList,
    types::{PyBytes, PyDict},
};

use crate::{
    rollout::{InferenceRequest, RolloutChunk, RolloutEngine, RolloutMetrics},
    EnvError,
};

#[pyclass(name = "InferenceRequest", frozen)]
pub struct PyInferenceRequest {
    inner: InferenceRequest,
}

#[pymethods]
impl PyInferenceRequest {
    #[getter]
    fn request_id(&self) -> u64 {
        self.inner.request_id
    }

    #[getter]
    fn policy_slot(&self) -> u32 {
        self.inner.policy_slot
    }

    #[getter]
    fn sequence_bucket(&self) -> usize {
        self.inner.sequence_bucket
    }

    #[getter]
    fn action_bucket(&self) -> usize {
        self.inner.action_bucket
    }

    #[getter]
    fn row_count(&self) -> usize {
        self.inner.rows()
    }

    #[getter]
    fn row_ids<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u64>> {
        array1(py, self.inner.row_ids.clone())
    }

    #[getter]
    fn environment_ids<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        array1(py, self.inner.environment_ids.clone())
    }

    #[getter]
    fn episode_generations<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u64>> {
        array1(py, self.inner.episode_generations.clone())
    }

    #[getter]
    fn frame_ids<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u64>> {
        array1(py, self.inner.frame_ids.clone())
    }

    #[getter]
    fn seats<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u8>> {
        array1(py, self.inner.seats.clone())
    }

    #[getter]
    fn token_factors<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<i32>> {
        array3(
            py,
            self.inner.rows(),
            self.inner.sequence_bucket,
            10,
            self.inner.token_factors.clone(),
        )
    }

    #[getter]
    fn token_numeric<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f32>> {
        array3(
            py,
            self.inner.rows(),
            self.inner.sequence_bucket,
            8,
            self.inner.token_numeric.clone(),
        )
    }

    #[getter]
    fn lengths<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        array1(py, self.inner.lengths.clone())
    }

    #[getter]
    fn query_offsets<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        array1(py, self.inner.query_offsets.clone())
    }

    #[getter]
    fn decision_seats<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        array1(py, self.inner.decision_seats.clone())
    }

    #[getter]
    fn rank_boundary_features<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> {
        array2(
            py,
            self.inner.rows(),
            28,
            self.inner.rank_boundary_features.clone(),
        )
    }

    #[getter]
    fn action_factors<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<i32>> {
        array3(
            py,
            self.inner.rows(),
            self.inner.action_bucket,
            15,
            self.inner.action_factors.clone(),
        )
    }

    #[getter]
    fn action_lengths<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        array1(py, self.inner.action_lengths.clone())
    }

    #[getter]
    fn action_offsets<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        array1(py, self.inner.action_offsets.clone())
    }

    #[pyo3(signature = (*, backend="sdpa"))]
    fn model_inputs<'py>(&self, py: Python<'py>, backend: &str) -> PyResult<Bound<'py, PyDict>> {
        let result = PyDict::new(py);
        result.set_item("token_factors", self.token_factors(py))?;
        result.set_item("token_numeric", self.token_numeric(py))?;
        result.set_item("actor_query_indices", self.query_offsets(py))?;
        result.set_item("lengths", self.lengths(py))?;
        result.set_item("decision_seats", self.decision_seats(py))?;
        result.set_item("rank_boundary_features", self.rank_boundary_features(py))?;
        result.set_item("action_factors", self.action_factors(py))?;
        result.set_item("action_lengths", self.action_lengths(py))?;
        result.set_item("action_offsets", self.action_offsets(py))?;
        result.set_item("backend", backend)?;
        Ok(result)
    }

    fn __repr__(&self) -> String {
        format!(
            "InferenceRequest(id={}, policy_slot={}, rows={}, sequence_bucket={}, action_bucket={})",
            self.inner.request_id,
            self.inner.policy_slot,
            self.inner.rows(),
            self.inner.sequence_bucket,
            self.inner.action_bucket,
        )
    }
}

#[pyclass(name = "RolloutChunk")]
#[derive(Clone)]
pub struct PyRolloutChunk {
    inner: RolloutChunk,
}

#[pymethods]
impl PyRolloutChunk {
    #[getter]
    fn row_count(&self) -> usize {
        self.inner.rows()
    }

    #[getter]
    fn kyoku_completions(&self) -> u64 {
        self.inner.kyoku_completions
    }

    #[getter]
    fn match_completions(&self) -> u64 {
        self.inner.match_completions
    }

    fn as_numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let result = PyDict::new(py);
        insert1(&result, py, "row_ids", self.inner.row_ids.clone())?;
        insert1(
            &result,
            py,
            "environment_ids",
            self.inner.environment_ids.clone(),
        )?;
        insert1(
            &result,
            py,
            "episode_generations",
            self.inner.episode_generations.clone(),
        )?;
        insert1(&result, py, "frame_ids", self.inner.frame_ids.clone())?;
        insert1(&result, py, "seats", self.inner.seats.clone())?;
        insert1(&result, py, "policy_slots", self.inner.policy_slots.clone())?;
        insert1(&result, py, "eligibility", self.inner.eligibility.clone())?;
        insert1(&result, py, "phases", self.inner.phases.clone())?;
        insert1(
            &result,
            py,
            "selected_groups",
            self.inner.selected_groups.clone(),
        )?;
        insert1(
            &result,
            py,
            "selected_native",
            self.inner.selected_native.clone(),
        )?;
        insert1(&result, py, "old_logp", self.inner.old_logp.clone())?;
        insert1(
            &result,
            py,
            "token_offsets",
            self.inner.token_offsets.clone(),
        )?;
        insert2(
            &result,
            py,
            "token_factors",
            self.inner.token_factors.len() / 10,
            10,
            self.inner.token_factors.clone(),
        )?;
        insert2(
            &result,
            py,
            "token_numeric",
            self.inner.token_numeric.len() / 8,
            8,
            self.inner.token_numeric.clone(),
        )?;
        insert1(
            &result,
            py,
            "query_offsets",
            self.inner.query_offsets.clone(),
        )?;
        insert1(
            &result,
            py,
            "decision_seats",
            self.inner.decision_seats.clone(),
        )?;
        insert2(
            &result,
            py,
            "rank_boundary_features",
            self.inner.rows(),
            28,
            self.inner.rank_boundary_features.clone(),
        )?;
        insert1(
            &result,
            py,
            "boundary_group_ids",
            self.inner.boundary_group_ids.clone(),
        )?;
        insert1(
            &result,
            py,
            "old_boundary_values",
            self.inner.old_boundary_values.clone(),
        )?;
        insert1(
            &result,
            py,
            "terminal_placements",
            self.inner.terminal_placements.clone(),
        )?;
        insert1(
            &result,
            py,
            "rank_order_targets",
            self.inner.rank_order_targets.clone(),
        )?;
        insert1(
            &result,
            py,
            "kyoku_boundary",
            self.inner.kyoku_boundary.clone(),
        )?;
        insert1(
            &result,
            py,
            "match_boundary",
            self.inner.match_boundary.clone(),
        )?;
        insert1(
            &result,
            py,
            "rank_boundary_supervision",
            self.inner.rank_boundary_supervision.clone(),
        )?;
        insert1(&result, py, "advantages", self.inner.advantages.clone())?;
        insert1(
            &result,
            py,
            "normalized_advantages",
            self.inner.normalized_advantages.clone(),
        )?;
        insert1(
            &result,
            py,
            "action_offsets",
            self.inner.action_offsets.clone(),
        )?;
        insert2(
            &result,
            py,
            "action_factors",
            self.inner.action_factors.len() / 15,
            15,
            self.inner.action_factors.clone(),
        )?;
        insert1(
            &result,
            py,
            "terminal_environment_ids",
            self.inner.terminal_environment_ids.clone(),
        )?;
        insert1(
            &result,
            py,
            "terminal_episode_generations",
            self.inner.terminal_episode_generations.clone(),
        )?;
        insert2(
            &result,
            py,
            "terminal_scores",
            self.inner.terminal_scores.len() / 4,
            4,
            self.inner.terminal_scores.clone(),
        )?;
        insert2(
            &result,
            py,
            "terminal_ranks",
            self.inner.terminal_ranks.len() / 4,
            4,
            self.inner.terminal_ranks.clone(),
        )?;
        insert1(
            &result,
            py,
            "terminal_completed_kyoku",
            self.inner.terminal_completed_kyoku.clone(),
        )?;
        insert1(
            &result,
            py,
            "terminal_exhaustive_ryukyoku",
            self.inner.terminal_exhaustive_ryukyoku.clone(),
        )?;
        insert2(
            &result,
            py,
            "terminal_wins",
            self.inner.terminal_wins.len() / 4,
            4,
            self.inner.terminal_wins.clone(),
        )?;
        insert2(
            &result,
            py,
            "terminal_deal_ins",
            self.inner.terminal_deal_ins.len() / 4,
            4,
            self.inner.terminal_deal_ins.clone(),
        )?;
        insert2(
            &result,
            py,
            "terminal_riichi_hands",
            self.inner.terminal_riichi_hands.len() / 4,
            4,
            self.inner.terminal_riichi_hands.clone(),
        )?;
        insert2(
            &result,
            py,
            "terminal_calling_hands",
            self.inner.terminal_calling_hands.len() / 4,
            4,
            self.inner.terminal_calling_hands.clone(),
        )?;
        insert2(
            &result,
            py,
            "terminal_tsumo_wins",
            self.inner.terminal_tsumo_wins.len() / 4,
            4,
            self.inner.terminal_tsumo_wins.clone(),
        )?;
        insert2(
            &result,
            py,
            "terminal_dama_wins",
            self.inner.terminal_dama_wins.len() / 4,
            4,
            self.inner.terminal_dama_wins.clone(),
        )?;
        insert2(
            &result,
            py,
            "terminal_winning_points",
            self.inner.terminal_winning_points.len() / 4,
            4,
            self.inner.terminal_winning_points.clone(),
        )?;
        insert2(
            &result,
            py,
            "terminal_winning_point_events",
            self.inner.terminal_winning_point_events.len() / 4,
            4,
            self.inner.terminal_winning_point_events.clone(),
        )?;
        insert2(
            &result,
            py,
            "terminal_deal_in_points",
            self.inner.terminal_deal_in_points.len() / 4,
            4,
            self.inner.terminal_deal_in_points.clone(),
        )?;
        insert2(
            &result,
            py,
            "terminal_deal_in_point_events",
            self.inner.terminal_deal_in_point_events.len() / 4,
            4,
            self.inner.terminal_deal_in_point_events.clone(),
        )?;
        insert2(
            &result,
            py,
            "terminal_winning_turns",
            self.inner.terminal_winning_turns.len() / 4,
            4,
            self.inner.terminal_winning_turns.clone(),
        )?;
        insert2(
            &result,
            py,
            "terminal_winning_turn_events",
            self.inner.terminal_winning_turn_events.len() / 4,
            4,
            self.inner.terminal_winning_turn_events.clone(),
        )?;
        Ok(result)
    }

    /// Compact typed columns without copying the potentially large ragged
    /// token and action arenas.
    fn columns<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let result = PyDict::new(py);
        insert1(&result, py, "row_ids", self.inner.row_ids.clone())?;
        insert1(
            &result,
            py,
            "environment_ids",
            self.inner.environment_ids.clone(),
        )?;
        insert1(
            &result,
            py,
            "episode_generations",
            self.inner.episode_generations.clone(),
        )?;
        insert1(&result, py, "frame_ids", self.inner.frame_ids.clone())?;
        insert1(&result, py, "seats", self.inner.seats.clone())?;
        insert1(&result, py, "policy_slots", self.inner.policy_slots.clone())?;
        insert1(&result, py, "eligibility", self.inner.eligibility.clone())?;
        insert1(
            &result,
            py,
            "selected_groups",
            self.inner.selected_groups.clone(),
        )?;
        insert1(&result, py, "old_logp", self.inner.old_logp.clone())?;
        insert1(
            &result,
            py,
            "token_offsets",
            self.inner.token_offsets.clone(),
        )?;
        insert1(
            &result,
            py,
            "action_offsets",
            self.inner.action_offsets.clone(),
        )?;
        insert1(
            &result,
            py,
            "boundary_group_ids",
            self.inner.boundary_group_ids.clone(),
        )?;
        insert1(
            &result,
            py,
            "old_boundary_values",
            self.inner.old_boundary_values.clone(),
        )?;
        insert1(
            &result,
            py,
            "terminal_placements",
            self.inner.terminal_placements.clone(),
        )?;
        insert1(
            &result,
            py,
            "rank_order_targets",
            self.inner.rank_order_targets.clone(),
        )?;
        insert1(
            &result,
            py,
            "rank_boundary_supervision",
            self.inner.rank_boundary_supervision.clone(),
        )?;
        insert1(&result, py, "advantages", self.inner.advantages.clone())?;
        insert1(
            &result,
            py,
            "normalized_advantages",
            self.inner.normalized_advantages.clone(),
        )?;
        insert1(
            &result,
            py,
            "terminal_environment_ids",
            self.inner.terminal_environment_ids.clone(),
        )?;
        insert1(
            &result,
            py,
            "terminal_episode_generations",
            self.inner.terminal_episode_generations.clone(),
        )?;
        insert2(
            &result,
            py,
            "terminal_scores",
            self.inner.terminal_scores.len() / 4,
            4,
            self.inner.terminal_scores.clone(),
        )?;
        insert2(
            &result,
            py,
            "terminal_ranks",
            self.inner.terminal_ranks.len() / 4,
            4,
            self.inner.terminal_ranks.clone(),
        )?;
        insert1(
            &result,
            py,
            "terminal_completed_kyoku",
            self.inner.terminal_completed_kyoku.clone(),
        )?;
        insert1(
            &result,
            py,
            "terminal_exhaustive_ryukyoku",
            self.inner.terminal_exhaustive_ryukyoku.clone(),
        )?;
        insert2(
            &result,
            py,
            "terminal_wins",
            self.inner.terminal_wins.len() / 4,
            4,
            self.inner.terminal_wins.clone(),
        )?;
        insert2(
            &result,
            py,
            "terminal_deal_ins",
            self.inner.terminal_deal_ins.len() / 4,
            4,
            self.inner.terminal_deal_ins.clone(),
        )?;
        insert2(
            &result,
            py,
            "terminal_riichi_hands",
            self.inner.terminal_riichi_hands.len() / 4,
            4,
            self.inner.terminal_riichi_hands.clone(),
        )?;
        insert2(
            &result,
            py,
            "terminal_calling_hands",
            self.inner.terminal_calling_hands.len() / 4,
            4,
            self.inner.terminal_calling_hands.clone(),
        )?;
        insert2(
            &result,
            py,
            "terminal_tsumo_wins",
            self.inner.terminal_tsumo_wins.len() / 4,
            4,
            self.inner.terminal_tsumo_wins.clone(),
        )?;
        insert2(
            &result,
            py,
            "terminal_dama_wins",
            self.inner.terminal_dama_wins.len() / 4,
            4,
            self.inner.terminal_dama_wins.clone(),
        )?;
        insert2(
            &result,
            py,
            "terminal_winning_points",
            self.inner.terminal_winning_points.len() / 4,
            4,
            self.inner.terminal_winning_points.clone(),
        )?;
        insert2(
            &result,
            py,
            "terminal_winning_point_events",
            self.inner.terminal_winning_point_events.len() / 4,
            4,
            self.inner.terminal_winning_point_events.clone(),
        )?;
        insert2(
            &result,
            py,
            "terminal_deal_in_points",
            self.inner.terminal_deal_in_points.len() / 4,
            4,
            self.inner.terminal_deal_in_points.clone(),
        )?;
        insert2(
            &result,
            py,
            "terminal_deal_in_point_events",
            self.inner.terminal_deal_in_point_events.len() / 4,
            4,
            self.inner.terminal_deal_in_point_events.clone(),
        )?;
        insert2(
            &result,
            py,
            "terminal_winning_turns",
            self.inner.terminal_winning_turns.len() / 4,
            4,
            self.inner.terminal_winning_turns.clone(),
        )?;
        insert2(
            &result,
            py,
            "terminal_winning_turn_events",
            self.inner.terminal_winning_turn_events.len() / 4,
            4,
            self.inner.terminal_winning_turn_events.clone(),
        )?;
        Ok(result)
    }

    #[pyo3(signature = (row_indices=None, *, backend="sdpa"))]
    fn actor_batch<'py>(
        &self,
        py: Python<'py>,
        row_indices: Option<Vec<usize>>,
        backend: &str,
    ) -> PyResult<Bound<'py, PyDict>> {
        let indices = row_indices.unwrap_or_else(|| (0..self.inner.rows()).collect());
        if indices.is_empty() || indices.iter().any(|&index| index >= self.inner.rows()) {
            return Err(PyValueError::new_err(
                "row_indices must select at least one valid rollout row",
            ));
        }
        let lengths = indices
            .iter()
            .map(|&index| {
                (self.inner.token_offsets[index + 1] - self.inner.token_offsets[index]) as usize
            })
            .collect::<Vec<_>>();
        let action_lengths = indices
            .iter()
            .map(|&index| {
                (self.inner.action_offsets[index + 1] - self.inner.action_offsets[index]) as usize
            })
            .collect::<Vec<_>>();
        let maximum = *lengths.iter().max().expect("nonempty");
        let action_maximum = *action_lengths.iter().max().expect("nonempty");
        let mut token_factors = vec![0_i32; indices.len() * maximum * 10];
        let mut token_numeric = vec![0.0_f32; indices.len() * maximum * 8];
        let mut action_factors = vec![0_i32; indices.len() * action_maximum * 15];
        let mut action_offsets = vec![0_i64];
        for (batch, &row) in indices.iter().enumerate() {
            let token_start = self.inner.token_offsets[row] as usize;
            for token in 0..lengths[batch] {
                let source = (token_start + token) * 10;
                let target = (batch * maximum + token) * 10;
                token_factors[target..target + 10]
                    .iter_mut()
                    .zip(&self.inner.token_factors[source..source + 10])
                    .for_each(|(target, source)| *target = i32::from(*source));
                let source = (token_start + token) * 8;
                let target = (batch * maximum + token) * 8;
                token_numeric[target..target + 8]
                    .copy_from_slice(&self.inner.token_numeric[source..source + 8]);
            }
            let action_start = self.inner.action_offsets[row] as usize;
            for action in 0..action_lengths[batch] {
                let source = (action_start + action) * 15;
                let target = (batch * action_maximum + action) * 15;
                action_factors[target..target + 15]
                    .iter_mut()
                    .zip(&self.inner.action_factors[source..source + 15])
                    .for_each(|(target, source)| *target = i32::from(*source));
            }
            action_offsets
                .push(action_offsets.last().copied().unwrap_or(0) + action_lengths[batch] as i64);
        }
        let result = PyDict::new(py);
        let model_inputs = PyDict::new(py);
        model_inputs.set_item(
            "token_factors",
            array3(py, indices.len(), maximum, 10, token_factors),
        )?;
        model_inputs.set_item(
            "token_numeric",
            array3(py, indices.len(), maximum, 8, token_numeric),
        )?;
        model_inputs.set_item(
            "actor_query_indices",
            array1(
                py,
                indices
                    .iter()
                    .map(|&index| i64::from(self.inner.query_offsets[index]))
                    .collect(),
            ),
        )?;
        model_inputs.set_item(
            "lengths",
            array1(py, lengths.iter().map(|&value| value as i64).collect()),
        )?;
        model_inputs.set_item(
            "decision_seats",
            array1(
                py,
                indices
                    .iter()
                    .map(|&index| i64::from(self.inner.decision_seats[index]))
                    .collect(),
            ),
        )?;
        model_inputs.set_item(
            "rank_boundary_features",
            array2(
                py,
                indices.len(),
                28,
                indices
                    .iter()
                    .flat_map(|&index| {
                        self.inner.rank_boundary_features[index * 28..index * 28 + 28]
                            .iter()
                            .copied()
                    })
                    .collect(),
            ),
        )?;
        model_inputs.set_item(
            "action_factors",
            array3(py, indices.len(), action_maximum, 15, action_factors),
        )?;
        model_inputs.set_item(
            "action_lengths",
            array1(
                py,
                action_lengths.iter().map(|&value| value as i64).collect(),
            ),
        )?;
        model_inputs.set_item("action_offsets", array1(py, action_offsets.clone()))?;
        result.set_item(
            "selected",
            array1(
                py,
                indices
                    .iter()
                    .enumerate()
                    .map(|(batch, &index)| {
                        action_offsets[batch] + i64::from(self.inner.selected_groups[index])
                    })
                    .collect(),
            ),
        )?;
        result.set_item(
            "old_logp",
            array1(
                py,
                indices
                    .iter()
                    .map(|&index| self.inner.old_logp[index])
                    .collect(),
            ),
        )?;
        result.set_item(
            "advantages",
            array1(
                py,
                indices
                    .iter()
                    .map(|&index| self.inner.normalized_advantages[index])
                    .collect(),
            ),
        )?;
        result.set_item(
            "raw_advantages",
            array1(
                py,
                indices
                    .iter()
                    .map(|&index| self.inner.advantages[index])
                    .collect(),
            ),
        )?;
        result.set_item(
            "ppo_eligible",
            array1(
                py,
                indices
                    .iter()
                    .map(|&index| self.inner.eligibility[index] != 0)
                    .collect(),
            ),
        )?;
        model_inputs.set_item("backend", backend)?;
        result.set_item("model_inputs", model_inputs)?;
        Ok(result)
    }

    #[pyo3(signature = (policy_slot, token_budget, *, max_padding_fraction=0.10, backend="sdpa"))]
    fn actor_minibatches<'py>(
        &self,
        py: Python<'py>,
        policy_slot: u32,
        token_budget: usize,
        max_padding_fraction: f64,
        backend: &str,
    ) -> PyResult<Bound<'py, PyList>> {
        if token_budget == 0 || !(0.0..1.0).contains(&max_padding_fraction) {
            return Err(PyValueError::new_err(
                "token budget must be positive and padding fraction in [0,1)",
            ));
        }
        let mut indices = (0..self.inner.rows())
            .filter(|&index| {
                self.inner.eligibility[index] != 0 && self.inner.policy_slots[index] == policy_slot
            })
            .collect::<Vec<_>>();
        indices.sort_by_key(|&index| {
            (
                std::cmp::Reverse(
                    self.inner.token_offsets[index + 1] - self.inner.token_offsets[index],
                ),
                index,
            )
        });
        let mut batches = Vec::<Vec<usize>>::new();
        let mut current = Vec::new();
        let mut maximum = 0_usize;
        let mut useful = 0_usize;
        for index in indices {
            let length =
                (self.inner.token_offsets[index + 1] - self.inner.token_offsets[index]) as usize;
            if length > token_budget {
                return Err(PyValueError::new_err(format!(
                    "rollout row length {length} exceeds token budget {token_budget}"
                )));
            }
            let next_maximum = maximum.max(length);
            let next_useful = useful + length;
            let next_padded = next_maximum * (current.len() + 1);
            let next_padding = (next_padded - next_useful) as f64 / next_padded as f64;
            if !current.is_empty()
                && (next_padded > token_budget || next_padding > max_padding_fraction)
            {
                batches.push(std::mem::take(&mut current));
                maximum = 0;
                useful = 0;
            }
            current.push(index);
            maximum = maximum.max(length);
            useful += length;
        }
        if !current.is_empty() {
            batches.push(current);
        }
        let result = PyList::empty(py);
        for batch in batches {
            result.append(self.actor_batch(py, Some(batch), backend)?)?;
        }
        Ok(result)
    }

    fn packing_metrics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let result = PyDict::new(py);
        let total_tokens = self.inner.token_offsets.last().copied().unwrap_or(0);
        result.set_item("rows", self.inner.rows())?;
        result.set_item("total_tokens", total_tokens)?;
        result.set_item(
            "total_actions",
            self.inner.action_offsets.last().copied().unwrap_or(0),
        )?;
        Ok(result)
    }

    fn action_statistics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let mut call = 0_u64;
        let mut pass = 0_u64;
        let mut riichi = 0_u64;
        let mut dama = 0_u64;
        for index in 0..self.inner.rows() {
            if self.inner.eligibility[index] == 0 {
                continue;
            }
            let start = self.inner.action_offsets[index] as usize;
            let end = self.inner.action_offsets[index + 1] as usize;
            let kinds = (start..end)
                .map(|action| self.inner.action_factors[action * 15])
                .collect::<Vec<_>>();
            let selected = kinds[self.inner.selected_groups[index] as usize];
            let has_call = kinds.iter().any(|kind| matches!(kind, 3..=5));
            if kinds.contains(&0) && has_call {
                if matches!(selected, 3..=5) {
                    call += 1;
                } else if selected == 0 {
                    pass += 1;
                }
            }
            if kinds.contains(&1) && kinds.contains(&2) {
                if selected == 2 {
                    riichi += 1;
                } else if selected == 1 {
                    dama += 1;
                }
            }
        }
        let result = PyDict::new(py);
        result.set_item("call_count", call)?;
        result.set_item("pass_count", pass)?;
        result.set_item("riichi_count", riichi)?;
        result.set_item("dama_count", dama)?;
        Ok(result)
    }

    #[getter]
    fn trajectory_digest(&self) -> String {
        let mut hashes = [
            0xcbf2_9ce4_8422_2325_u64,
            0x8422_2325_cbf2_9ce4,
            0x9e37_79b9_7f4a_7c15,
            0xd6e8_feb8_6659_fd93,
        ];
        {
            let mut update = |bytes: &[u8]| {
                for &byte in bytes {
                    for (lane, hash) in hashes.iter_mut().enumerate() {
                        *hash ^= u64::from(byte).wrapping_add(lane as u64 * 0x9d);
                        *hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
                    }
                }
            };
            for index in 0..self.inner.rows() {
                update(&self.inner.environment_ids[index].to_le_bytes());
                update(&self.inner.episode_generations[index].to_le_bytes());
                update(&self.inner.frame_ids[index].to_le_bytes());
                update(&[self.inner.seats[index]]);
                update(&self.inner.selected_groups[index].to_le_bytes());
                update(&self.inner.old_logp[index].to_bits().to_le_bytes());
            }
        }
        format!(
            "{:016x}{:016x}{:016x}{:016x}",
            hashes[0], hashes[1], hashes[2], hashes[3]
        )
    }

    fn boundary_batch<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let mut seen = std::collections::BTreeSet::new();
        let indices = (0..self.inner.rows())
            .filter(|&index| {
                self.inner.eligibility[index] != 0
                    && seen.insert(self.inner.boundary_group_ids[index])
            })
            .collect::<Vec<_>>();
        let result = PyDict::new(py);
        result.set_item(
            "boundary_group_ids",
            array1(
                py,
                indices
                    .iter()
                    .map(|&index| self.inner.boundary_group_ids[index])
                    .collect(),
            ),
        )?;
        result.set_item(
            "policy_slots",
            array1(
                py,
                indices
                    .iter()
                    .map(|&index| self.inner.policy_slots[index])
                    .collect(),
            ),
        )?;
        result.set_item(
            "decision_seats",
            array1(
                py,
                indices
                    .iter()
                    .map(|&index| i64::from(self.inner.decision_seats[index]))
                    .collect(),
            ),
        )?;
        result.set_item(
            "rank_boundary_features",
            array2(
                py,
                indices.len(),
                28,
                indices
                    .iter()
                    .flat_map(|&index| {
                        self.inner.rank_boundary_features[index * 28..index * 28 + 28]
                            .iter()
                            .copied()
                    })
                    .collect(),
            ),
        )?;
        Ok(result)
    }

    #[pyo3(signature = (*, policy_slot=None))]
    fn critic_batch<'py>(
        &self,
        py: Python<'py>,
        policy_slot: Option<u32>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let indices = (0..self.inner.rows())
            .filter(|&index| {
                self.inner.rank_boundary_supervision[index] != 0
                    && policy_slot.is_none_or(|slot| self.inner.policy_slots[index] == slot)
            })
            .collect::<Vec<_>>();
        if indices.is_empty() {
            return Err(PyValueError::new_err(
                "critic batch has no supervised boundary rows",
            ));
        }
        let model_inputs = PyDict::new(py);
        model_inputs.set_item(
            "decision_seats",
            array1(
                py,
                indices
                    .iter()
                    .map(|&index| i64::from(self.inner.decision_seats[index]))
                    .collect(),
            ),
        )?;
        model_inputs.set_item(
            "rank_boundary_features",
            array2(
                py,
                indices.len(),
                28,
                indices
                    .iter()
                    .flat_map(|&index| {
                        self.inner.rank_boundary_features[index * 28..index * 28 + 28]
                            .iter()
                            .copied()
                    })
                    .collect(),
            ),
        )?;
        let result = PyDict::new(py);
        result.set_item("model_inputs", model_inputs)?;
        result.set_item(
            "rank_boundary_supervision",
            array1(py, vec![true; indices.len()]),
        )?;
        result.set_item(
            "rank_order_targets",
            array1(
                py,
                indices
                    .iter()
                    .map(|&index| i64::from(self.inner.rank_order_targets[index]))
                    .collect(),
            ),
        )?;
        Ok(result)
    }

    fn set_boundary_values(&mut self, group_ids: Vec<u64>, values: Vec<f32>) -> PyResult<()> {
        self.inner
            .set_boundary_values(&group_ids, &values)
            .map_err(to_py_error)
    }

    fn finish_targets(&mut self) -> PyResult<()> {
        self.inner.finish_targets().map_err(to_py_error)
    }
}

#[pyclass(name = "RolloutEngine")]
pub struct PyRolloutEngine {
    inner: RolloutEngine,
}

#[pymethods]
impl PyRolloutEngine {
    #[new]
    #[pyo3(signature = (num_envs, *, master_seed, num_threads, context_tokens=2048, token_budget=65536, inference_only=false, rules_profile=riichi_core::RULES_PROFILE))]
    fn new(
        num_envs: usize,
        master_seed: u64,
        num_threads: usize,
        context_tokens: usize,
        token_budget: usize,
        inference_only: bool,
        rules_profile: &str,
    ) -> PyResult<Self> {
        let profile =
            riichi_core::game::rules::profile::by_name(rules_profile).ok_or_else(|| {
                PyValueError::new_err(format!("unsupported rules profile: {rules_profile}"))
            })?;
        Ok(Self {
            inner: RolloutEngine::new_with_rules_profile(
                num_envs,
                master_seed,
                num_threads,
                context_tokens,
                token_budget,
                inference_only,
                profile.profile_id,
            )
            .map_err(to_py_error)?,
        })
    }

    #[getter]
    fn num_envs(&self) -> usize {
        self.inner.num_envs()
    }

    #[pyo3(signature = (target_matches=None))]
    fn reset_chunk(
        &mut self,
        py: Python<'_>,
        target_matches: Option<usize>,
    ) -> PyResult<Vec<(u32, u64)>> {
        let target_matches = target_matches.unwrap_or_else(|| self.inner.num_envs());
        py.detach(|| self.inner.reset_chunk(target_matches))
            .map_err(to_py_error)
    }

    fn reset_chunk_seeded(&mut self, py: Python<'_>, seeds: Vec<u64>) -> PyResult<Vec<(u32, u64)>> {
        py.detach(|| self.inner.reset_chunk_seeded(&seeds))
            .map_err(to_py_error)
    }

    #[pyo3(signature = (matches, policy_slots, learner_masks, *, bot_policy_slots=Vec::new()))]
    fn register_lineups(
        &mut self,
        matches: Vec<(u32, u64)>,
        policy_slots: Vec<(u32, u32, u32, u32)>,
        learner_masks: Vec<u8>,
        bot_policy_slots: Vec<u32>,
    ) -> PyResult<()> {
        let policy_slots = policy_slots
            .into_iter()
            .map(Into::into)
            .collect::<Vec<[u32; 4]>>();
        self.inner
            .register_lineups(&matches, &policy_slots, &learner_masks, &bot_policy_slots)
            .map_err(to_py_error)
    }

    fn next_request(&mut self, py: Python<'_>) -> PyResult<Option<PyInferenceRequest>> {
        py.detach(|| self.inner.next_request())
            .map(|value| value.map(|inner| PyInferenceRequest { inner }))
            .map_err(to_py_error)
    }

    fn submit(
        &mut self,
        py: Python<'_>,
        request_id: u64,
        selected_groups: Vec<usize>,
        old_logp: Vec<f32>,
    ) -> PyResult<()> {
        py.detach(|| self.inner.submit(request_id, &selected_groups, &old_logp))
            .map_err(to_py_error)
    }

    #[getter]
    fn complete(&self) -> bool {
        self.inner.is_complete()
    }

    fn chunk(&self) -> PyRolloutChunk {
        PyRolloutChunk {
            inner: self.inner.chunk().clone(),
        }
    }

    fn take_chunk(&mut self) -> PyResult<PyRolloutChunk> {
        self.inner
            .take_chunk()
            .map(|inner| PyRolloutChunk { inner })
            .map_err(to_py_error)
    }

    fn metrics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        metrics_dict(py, self.inner.metrics())
    }

    fn snapshot<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let snapshots = self.inner.snapshots().map_err(to_py_error)?;
        let result = PyDict::new(py);
        for (environment_id, payload) in snapshots {
            result.set_item(environment_id, PyBytes::new(py, &payload))?;
        }
        Ok(result)
    }

    fn restore(&mut self, py: Python<'_>, snapshots: BTreeMap<u32, Vec<u8>>) -> PyResult<()> {
        py.detach(|| self.inner.restore_idle(&snapshots))
            .map_err(to_py_error)
    }
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyInferenceRequest>()?;
    module.add_class::<PyRolloutChunk>()?;
    module.add_class::<PyRolloutEngine>()?;
    Ok(())
}

fn metrics_dict<'py>(py: Python<'py>, metrics: &RolloutMetrics) -> PyResult<Bound<'py, PyDict>> {
    let result = PyDict::new(py);
    result.set_item("inference_requests", metrics.inference_requests)?;
    result.set_item("inference_rows", metrics.inference_rows)?;
    result.set_item("useful_tokens", metrics.useful_tokens)?;
    result.set_item("padded_tokens", metrics.padded_tokens)?;
    result.set_item("useful_actions", metrics.useful_actions)?;
    result.set_item("padded_actions", metrics.padded_actions)?;
    result.set_item("native_bot_rows", metrics.native_bot_rows)?;
    result.set_item("automatic_rows", metrics.automatic_rows)?;
    result.set_item("env_calls", metrics.env_calls)?;
    result.set_item("compile_fallbacks", metrics.compile_fallbacks)?;
    Ok(result)
}

fn array1<T: Element>(py: Python<'_>, values: Vec<T>) -> Bound<'_, PyArray1<T>> {
    let result = PyArray1::from_vec(py, values);
    result
        .call_method1("setflags", (false,))
        .expect("numpy setflags");
    result
}

fn array2<T: Element>(
    py: Python<'_>,
    rows: usize,
    columns: usize,
    values: Vec<T>,
) -> Bound<'_, PyArray2<T>> {
    let result = Array2::from_shape_vec((rows, columns), values)
        .expect("native rollout 2D shape")
        .into_pyarray(py);
    result
        .call_method1("setflags", (false,))
        .expect("numpy setflags");
    result
}

fn array3<T: Element>(
    py: Python<'_>,
    first: usize,
    second: usize,
    third: usize,
    values: Vec<T>,
) -> Bound<'_, PyArray3<T>> {
    let result = Array3::from_shape_vec((first, second, third), values)
        .expect("native rollout 3D shape")
        .into_pyarray(py);
    result
        .call_method1("setflags", (false,))
        .expect("numpy setflags");
    result
}

fn insert1<T: Element>(
    dict: &Bound<'_, PyDict>,
    py: Python<'_>,
    name: &str,
    values: Vec<T>,
) -> PyResult<()> {
    dict.set_item(name, array1(py, values))
}

fn insert2<T: Element>(
    dict: &Bound<'_, PyDict>,
    py: Python<'_>,
    name: &str,
    rows: usize,
    columns: usize,
    values: Vec<T>,
) -> PyResult<()> {
    dict.set_item(name, array2(py, rows, columns, values))
}

fn to_py_error(error: EnvError) -> PyErr {
    match error {
        EnvError::InvalidArgument(_)
        | EnvError::EnvironmentOutOfRange(_)
        | EnvError::DuplicateEnvironment(_)
        | EnvError::SnapshotEnvironmentMismatch { .. }
        | EnvError::Core(riichi_core::error::CoreError::InvalidActions { .. }) => {
            PyValueError::new_err(error.to_string())
        }
        EnvError::Closed | EnvError::Core(_) => PyRuntimeError::new_err(error.to_string()),
    }
}
