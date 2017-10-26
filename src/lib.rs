#![feature(proc_macro, specialization)]

extern crate pyo3;
use pyo3::prelude::*;
use pyo3::py::modinit;

mod centroid;
use centroid::{Centroid, Numeric};


#[modinit(rust_gap)]
fn init_mod(py: Python, m: &PyModule) -> PyResult<()> {

    #[pyfn(m, "sum_as_string")]
    fn sum_as_string_py(a: i64, b: i64) -> PyResult<String> {
        let out = sum_as_string(a, b);
        Ok(out)
    }

    #[pyfn(m, "optimalK")]
    fn optimalk_py(x: Vec<f64>) -> PyResult<Vec<f64>> {

        let centroid: Centroid = Centroid::new(x);
        let out: Vec<f64> = vec![5.0, 5.6];
        Ok(out)
    }

    Ok(())
}

fn sum_as_string(a: i64, b: i64) -> String {
    format!("{}", a + b).to_string()
}