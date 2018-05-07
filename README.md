
### Python implementation of the [Gap Statistic](http://www.web.stanford.edu/~hastie/Papers/gap.pdf)

![Build Status](https://travis-ci.org/druogury/clustering-gap-statistic.svg?branch=master)
[![Coverage Status](https://coveralls.io/repos/github/druogury/clustering-gap-statistic/badge.svg)](https://coveralls.io/github/druogury/clustering-gap-statistic)
[![Code Health](https://landscape.io/github/druogury/clustering-gap-statistic/master/landscape.svg?style=flat)](https://landscape.io/github/druogury/clustering-gap-statistic/master)

---
#### Purpose
Dynamically identify the suggested number of clusters in a data-set
using the gap statistic.

---
#### Improvements
- Correct dispersion formula (mean of log instead of log of mean)
- Compute gap statistic's standard deviation
- Add Scikit-learn KMeans and SphericalKMeans. 
- Scipy kmeans2 looks very unstable, that's why it's not the default algorithm anymore.

---

### Full example available in a notebook [HERE](Example.ipynb)

---
#### Install:  
Bleeding edge:  
```commandline
pip install git+git://github.com/druogury/clustering-gap-statistic.git
```

PyPi:  
```commandline
pip install --upgrade gap-stat
```

---
#### Uninstall:
```commandline
pip uninstall gap-stat
```
