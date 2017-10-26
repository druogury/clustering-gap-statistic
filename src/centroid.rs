
use std::vec::Vec;
use std::marker::Sized;

pub struct Centroid {
    center: Vec<f64>
}


impl Centroid {

    pub fn new(x: Vec<f64>) -> Centroid {
        Centroid {center: x}
    }

    pub fn location(self) -> Vec<f64> {
        self.center
    }
}

