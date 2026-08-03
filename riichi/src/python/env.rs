use std::{collections::BTreeMap, time::Instant};

use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
    types::{PyBytes, PyDict},
};

use crate::batch_env::{BatchEnv, EnvError};

use super::types::{
    materialize_transition, PyActionSelection, PyReplayEvent, PyReplayHanchan, PyTransition,
};

#[pyclass(name = "_Env")]
pub struct PyEnv {
    inner: BatchEnv,
    privileged: bool,
    rules_profile: String,
}

#[pymethods]
impl PyEnv {
    #[new]
    #[pyo3(signature = (num_envs, *, master_seed, num_threads, rules_profile=riichi_core::RULES_PROFILE, privileged=false))]
    fn new(
        num_envs: usize,
        master_seed: u64,
        num_threads: usize,
        rules_profile: &str,
        privileged: bool,
    ) -> PyResult<Self> {
        let profile =
            riichi_core::game::rules::profile::by_name(rules_profile).ok_or_else(|| {
                PyValueError::new_err(format!("unsupported rules profile: {rules_profile}"))
            })?;
        Ok(Self {
            inner: BatchEnv::new_with_rules_profile(
                num_envs,
                master_seed,
                num_threads,
                profile.profile_id,
            )
            .map_err(to_py_error)?,
            privileged,
            rules_profile: rules_profile.to_owned(),
        })
    }

    #[getter]
    fn num_envs(&self) -> usize {
        self.inner.num_envs()
    }

    #[getter]
    fn num_threads(&self) -> usize {
        self.inner.num_threads()
    }

    #[getter]
    fn master_seed(&self) -> u64 {
        self.inner.master_seed()
    }

    #[getter]
    fn rules_profile(&self) -> &str {
        &self.rules_profile
    }

    #[getter]
    fn privileged(&self) -> bool {
        self.privileged
    }

    fn reset(&mut self, py: Python<'_>, environment_ids: Vec<u32>) -> PyResult<PyTransition> {
        let value = py
            .detach(|| self.inner.reset(&environment_ids))
            .map_err(to_py_error)?;
        Ok(self.materialize(value))
    }

    fn step(
        &mut self,
        py: Python<'_>,
        selections: Vec<Py<PyActionSelection>>,
    ) -> PyResult<PyTransition> {
        let selections = selections
            .iter()
            .map(|value| value.borrow(py).inner.clone())
            .collect::<Vec<_>>();
        let value = py
            .detach(|| self.inner.step(&selections))
            .map_err(to_py_error)?;
        Ok(self.materialize(value))
    }

    fn advance(&mut self, py: Python<'_>, environment_ids: Vec<u32>) -> PyResult<PyTransition> {
        let value = py
            .detach(|| self.inner.advance(&environment_ids))
            .map_err(to_py_error)?;
        Ok(self.materialize(value))
    }

    fn load_hanchan(
        &mut self,
        py: Python<'_>,
        values: Vec<Py<PyReplayHanchan>>,
    ) -> PyResult<PyTransition> {
        let values = values
            .iter()
            .map(|value| value.borrow(py).inner.clone())
            .collect::<Vec<_>>();
        let result = py
            .detach(|| self.inner.load_hanchan(&values))
            .map_err(to_py_error)?;
        Ok(self.materialize(result))
    }

    fn apply_events(
        &mut self,
        py: Python<'_>,
        values: Vec<Py<PyReplayEvent>>,
    ) -> PyResult<PyTransition> {
        let values = values
            .iter()
            .map(|value| value.borrow(py).inner.clone())
            .collect::<Vec<_>>();
        let result = py
            .detach(|| self.inner.apply_events(&values))
            .map_err(to_py_error)?;
        Ok(self.materialize(result))
    }

    #[pyo3(signature = (environment_ids, *, privileged=None))]
    fn inspect(
        &mut self,
        py: Python<'_>,
        environment_ids: Vec<u32>,
        privileged: Option<bool>,
    ) -> PyResult<PyTransition> {
        let value = py
            .detach(|| self.inner.inspect(&environment_ids))
            .map_err(to_py_error)?;
        Ok(materialize_transition(
            value,
            privileged.unwrap_or(self.privileged),
        ))
    }

    /// Search-only cloning with privileged hidden-wall resampling.
    fn fork_privileged_wall(
        &mut self,
        py: Python<'_>,
        source_environment_id: u32,
        branches: Vec<(u32, u64)>,
    ) -> PyResult<PyTransition> {
        let value = py
            .detach(|| {
                self.inner
                    .fork_privileged_wall(source_environment_id, &branches)
            })
            .map_err(to_py_error)?;
        Ok(self.materialize(value))
    }

    /// Search-only public-information determinization for a self turn.
    fn fork_public_information(
        &mut self,
        py: Python<'_>,
        source_environment_id: u32,
        observer_seat: u8,
        branches: Vec<(u32, u64)>,
    ) -> PyResult<PyTransition> {
        let value = py
            .detach(|| {
                self.inner
                    .fork_public_information(source_environment_id, observer_seat, &branches)
            })
            .map_err(to_py_error)?;
        Ok(self.materialize(value))
    }

    /// Search-only exact state cloning after chance has been determined.
    fn fork_search_state(
        &mut self,
        py: Python<'_>,
        source_environment_id: u32,
        target_environment_ids: Vec<u32>,
    ) -> PyResult<PyTransition> {
        let value = py
            .detach(|| {
                self.inner
                    .fork_search_state(source_environment_id, &target_environment_ids)
            })
            .map_err(to_py_error)?;
        Ok(self.materialize(value))
    }

    fn snapshot<'py>(
        &self,
        py: Python<'py>,
        environment_ids: Vec<u32>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let values = py
            .detach(|| self.inner.snapshots(&environment_ids))
            .map_err(to_py_error)?;
        let result = PyDict::new(py);
        for (environment_id, payload) in values {
            result.set_item(environment_id, PyBytes::new(py, &payload))?;
        }
        Ok(result)
    }

    fn restore(
        &mut self,
        py: Python<'_>,
        snapshots: BTreeMap<u32, Vec<u8>>,
    ) -> PyResult<PyTransition> {
        let value = py
            .detach(|| self.inner.restore(&snapshots))
            .map_err(to_py_error)?;
        Ok(self.materialize(value))
    }

    #[pyo3(signature = (*, reset=false))]
    fn metrics<'py>(&mut self, py: Python<'py>, reset: bool) -> PyResult<Bound<'py, PyDict>> {
        let values = self.inner.metrics(reset);
        let result = PyDict::new(py);
        for (name, value) in values.as_pairs() {
            result.set_item(name, value)?;
        }
        Ok(result)
    }

    fn close(&mut self) {
        self.inner.close();
    }
}

impl PyEnv {
    fn materialize(&mut self, value: crate::batch_env::BatchTransition) -> PyTransition {
        let started = Instant::now();
        let mut result = materialize_transition(value, self.privileged);
        let elapsed = started.elapsed();
        result.exchange_ns = elapsed.as_nanos() as u64;
        self.inner.record_exchange(elapsed);
        result
    }
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
