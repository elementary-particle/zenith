use pyo3::{exceptions::PyOSError, prelude::*};

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(ensure_shanten_cache, module)?)?;
    Ok(())
}

/// Prewarm the shared shanten cache before rollout workers are created.
#[pyfunction]
fn ensure_shanten_cache() -> PyResult<String> {
    riichi_core::game::rules::shanten::ensure_cache()
        .map(|path| path.to_string_lossy().into_owned())
        .map_err(|error| PyOSError::new_err(error.to_string()))
}
