use numpy::{
    ndarray::Array2, IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2,
    PyUntypedArrayMethods,
};
use pyo3::{exceptions::PyValueError, prelude::*};

use riichi_core::game::rules::hand_efficiency;

#[pyclass(name = "HandEfficiencyBatch", frozen)]
pub struct HandEfficiencyBatch {
    shanten: Py<PyArray2<i8>>,
    improving_tile_mask: Py<PyArray1<u64>>,
    #[pyo3(get)]
    row_count: usize,
    #[pyo3(get)]
    efficiency_version: u32,
}

#[pymethods]
impl HandEfficiencyBatch {
    #[getter]
    fn shanten(&self, py: Python<'_>) -> Py<PyArray2<i8>> {
        self.shanten.clone_ref(py)
    }

    #[getter]
    fn improving_tile_mask(&self, py: Python<'_>) -> Py<PyArray1<u64>> {
        self.improving_tile_mask.clone_ref(py)
    }
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<HandEfficiencyBatch>()?;
    module.add_function(wrap_pyfunction!(evaluate_hand_efficiency, module)?)?;
    Ok(())
}

#[pyfunction]
fn evaluate_hand_efficiency(
    py: Python<'_>,
    counts: PyReadonlyArray2<'_, u8>,
    open_melds: PyReadonlyArray1<'_, u8>,
) -> PyResult<HandEfficiencyBatch> {
    let shape = counts.shape();
    if shape.len() != 2 || shape[1] != 34 {
        return Err(PyValueError::new_err("counts must have shape uint8[N,34]"));
    }
    let rows = shape[0];
    if open_melds.shape() != [rows] {
        return Err(PyValueError::new_err("open_melds must have shape uint8[N]"));
    }
    let counts = counts
        .as_slice()
        .map_err(|_| PyValueError::new_err("counts must be C-contiguous"))?;
    let open_melds = open_melds
        .as_slice()
        .map_err(|_| PyValueError::new_err("open_melds must be C-contiguous"))?;
    let mut hands = Vec::with_capacity(rows);
    for (row, (&melds, values)) in open_melds.iter().zip(counts.chunks_exact(34)).enumerate() {
        if melds > 4 || values.iter().any(|&count| count > 4) {
            return Err(PyValueError::new_err(format!(
                "invalid count or open_melds in row {row}"
            )));
        }
        let total = values
            .iter()
            .map(|&value| usize::from(value))
            .sum::<usize>()
            + 3 * usize::from(melds);
        if total != 13 && total != 14 {
            return Err(PyValueError::new_err(format!(
                "row {row} represents {total} tiles; expected 13 or 14"
            )));
        }
        let mut hand = [0_u8; 34];
        hand.copy_from_slice(values);
        hands.push((hand, melds));
    }

    let values = py.detach(|| {
        hands
            .iter()
            .map(|(hand, melds)| hand_efficiency::evaluate(hand, *melds))
            .collect::<Vec<_>>()
    });
    let shanten_values = values
        .iter()
        .flat_map(|value| {
            [
                value.shanten.overall,
                value.shanten.standard,
                value.shanten.seven_pairs,
                value.shanten.thirteen_orphans,
            ]
        })
        .collect::<Vec<_>>();
    let masks = values
        .into_iter()
        .map(|value| value.improving_tile_mask)
        .collect::<Vec<_>>();
    let shanten = Array2::from_shape_vec((rows, 4), shanten_values)
        .expect("hand efficiency shape")
        .into_pyarray(py);
    let masks = PyArray1::from_vec(py, masks);
    shanten.call_method1("setflags", (false,))?;
    masks.call_method1("setflags", (false,))?;
    Ok(HandEfficiencyBatch {
        shanten: shanten.unbind(),
        improving_tile_mask: masks.unbind(),
        row_count: rows,
        efficiency_version: riichi_core::HAND_EFFICIENCY_VERSION,
    })
}
