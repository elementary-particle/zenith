use std::{collections::BTreeMap, time::Instant};

use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
    types::{PyBytes, PyDict},
};

use crate::batch_env::{BatchEnv, EnvError};

use super::types::{materialize_transition, PyAction, PyTransition};

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
        if rules_profile != riichi_core::RULES_PROFILE {
            return Err(PyValueError::new_err(format!(
                "unsupported rules profile: {rules_profile}"
            )));
        }
        Ok(Self {
            inner: BatchEnv::new(num_envs, master_seed, num_threads).map_err(to_py_error)?,
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

    fn step(&mut self, py: Python<'_>, actions: Vec<Py<PyAction>>) -> PyResult<PyTransition> {
        let actions = actions
            .iter()
            .map(|value| value.borrow(py).inner.clone())
            .collect::<Vec<_>>();
        let value = py
            .detach(|| self.inner.step(&actions))
            .map_err(to_py_error)?;
        Ok(self.materialize(value))
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
        EnvError::Closed | EnvError::UnqueryableState { .. } | EnvError::Core(_) => {
            PyRuntimeError::new_err(error.to_string())
        }
    }
}
