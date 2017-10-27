#![feature(proc_macro, specialization)]

extern crate pyo3;
use pyo3::prelude::*;
use pyo3::py::modinit;

mod centroid;
use centroid::{Centroid, KMeans};


#[modinit(rust_gap)]
fn init_mod(py: Python, m: &PyModule) -> PyResult<()> {

    #[pyfn(m, "sum_as_string")]
    fn sum_as_string_py(a: i64, b: i64) -> PyResult<String> {
        let out = sum_as_string(a, b);
        Ok(out)
    }

    #[pyfn(m, "optimalK")]
    fn optimalk_py(x: Vec<Vec<f64>>) -> PyResult<Vec<u32>> {

        let out: Vec<u32> = kmeans(2, x);
        Ok(out)
    }

    Ok(())
}

fn sum_as_string(a: i64, b: i64) -> String {
    format!("{}", a + b).to_string()
}

fn kmeans(n_clusters: u32, X: Vec<Vec<f64>>) -> Vec<u32> {
    /*
        Implement of the KMeans algorithm
    */
    let mut centroids: Vec<Centroid> = Vec::new();
    let mut record: Vec<f64>;

    for (i, record) in X.enumerate() {
        let centroid = Centroid::new(*record);
    }
    let labels: Vec<u32> = vec![1,2,1,2];
    labels
}