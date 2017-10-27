
use std::vec::Vec;
use std::marker::Sized;

pub struct Centroid {
    center: Vec<f64>,
    assigned_points: Vec<Vec<f64>>
}

pub struct KMeans {
    centroids: Vec<Centroid>
}


impl Centroid {

    pub fn new(x: Vec<f64>) -> Centroid {
        Centroid {center: x, assigned_points: Vec::new()}
    }

    pub fn location(&self) -> &Vec<f64> {
        &self.center
    }

    pub fn fit(&mut self, x: &Vec<Vec<f64>>) {
        /*
        Move center to new location
        */
    }

}

