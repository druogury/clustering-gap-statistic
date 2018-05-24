# -*- coding: utf-8 -*-

import warnings
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from typing import Union, Iterable, Tuple
from scipy.cluster.vq import kmeans, kmeans2
from sklearn.cluster import KMeans
from spherecluster import SphericalKMeans
try:
    from joblib import Parallel, delayed
except ImportError:
    warnings.warn('joblib not installed, will be unavailable as a backend for parallel processing.')



def random_sample_data(X: Union[pd.DataFrame, np.ndarray], random_sampling: str='uniform', output_image=None):
    """ create randomly sampled data using data boundaries (or statistics)

    Parameters
    ----------
    X : data (samples in rows, variables in columns)
    random_sampling : sampling method (uniform or gaussian)
    output_image : dump 2D representation of data (2 first dimensions) in a file for control

    Returns
    -------
    random_data : randomly sampled data
    """
    if not(random_sampling in ['uniform', 'gaussian']):
        raise ValueError('Unknown random_sampling argument: {}'.format(random_sampling))

    if type(X) is np.ndarray:
        X = pd.DataFrame(X)

    if random_sampling == "uniform":
        # uniform sampling in [a, b[ : (b-a) * rand([0., 1.[) + a
        max_min_vct = X.describe().loc["max"].values - X.describe().loc["min"].values
        min_vct = X.describe().loc["min"].values
        random_data = np.multiply(max_min_vct, np.random.random_sample(X.shape)) + min_vct

    elif random_sampling == "gaussian":
        # multivariate gaussian instead of uniform (assume that components are independant)
        # https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.random.multivariate_normal.html
        # !!! Given a shape of, for example, (m,n,k), m*n*k samples are generated,
        # !!! and packed in an m-by-n-by-k arrangement. Because each sample is N-dimensional,
        # !!! the output shape is (m,n,k,N)
        random_data = np.random.multivariate_normal(
            X.describe().loc["mean"].values,
            np.diag(X.describe().loc["std"].values ** 2.), # covariance
            size=(len(X),))

    assert random_data.shape == X.shape, "check sampled data shape {} vs original data shape {}".format(X.shape, random_data.shape)

    if not(output_image is None):
        fig = plt.figure()
        plt.title("Randomly sampled vs true data in 2D")
        plt.scatter(   X.values[:, 0],    X.values[:, 1], color='b', s=1, label='true data')
        plt.scatter(random_data[:, 0], random_data[:, 1], color='r', s=1, label='sampled data', alpha=0.3)
        plt.legend()
        fig.savefig(output_image)
        
    return random_data


#%%
# https://github.com/milesgranger/gap_statistic/blob/master/gap_statistic/optimalK.py
class OptimalK:
    """
    Obtain the optimal number of clusters a dataset should have using the gap statistic.
        Tibshirani, Walther, Hastie
        http://www.web.stanford.edu/~hastie/Papers/gap.pdf
    Example:
    >>> from sklearn.datasets.samples_generator import make_blobs
    >>> from gap_statistic import OptimalK
    >>> X, y = make_blobs(n_samples=int(1e5), n_features=2, centers=3, random_state=100)
    >>> optimalK = OptimalK(parallel_backend='joblib')
    >>> optimalK(X, cluster_array=[1,2,3,4,5])
    3
    """
    gap_df = None

    def __init__(self, n_jobs: int=-1, parallel_backend: str='joblib', algo: str='skl-kmeans', random_sampling: str='uniform', fname=None) -> None:
        """
        Construct OptimalK to use n_jobs (multiprocessing using joblib, multiprocessing, or single core.
        :param n_jobs - int: Number of CPU cores to use. Use all cores if n_jobs == -1
        """
        self.parallel_backend = parallel_backend if parallel_backend in ['joblib', 'multiprocessing'] else None
        self.n_jobs = n_jobs if 1 <= n_jobs <= cpu_count() else cpu_count()  # type: int
        self.n_jobs = 1 if parallel_backend is None else self.n_jobs
        self.algo = algo
        self.random_sampling = random_sampling
        self.fname = fname

    def __call__(self, X: Union[pd.DataFrame, np.ndarray], n_refs: int=3, cluster_array: Iterable[int]=()):
        """
        Calculates KMeans optimal K using Gap Statistic from Tibshirani, Walther, Hastie
        http://www.web.stanford.edu/~hastie/Papers/gap.pdf
        :param X - pandas dataframe or numpy array of data points of shape (n_samples, n_features)
        :param n_refs - int: Number of random reference data sets used as inertia reference to actual data.
        :param cluster_array - 1d iterable of integers; each representing n_clusters to try on the data.
        """

        # Convert the 1d array of n_clusters to try into an array
        # Raise error if values are less than 1 or larger than the unique sample in the set.
        cluster_array = np.array([x for x in cluster_array]).astype(int)
        if np.where(cluster_array < 1)[0].shape[0]:
            raise ValueError('cluster_array contains values less than 1: {}'
                             .format(cluster_array[np.where(cluster_array < 1)[0]])
            )
        if cluster_array.shape[0] > X.shape[0]:
            raise ValueError('The number of suggested clusters to try ({}) is larger than samples in dataset. ({})'
                             .format(cluster_array.shape[0], X.shape[0])
            )
        if not cluster_array.shape[0]:
            raise ValueError('The supplied cluster_array has no values.')

        if not(self.algo in ['kmeans', 'kmeans2', 'skl-kmeans', 'sph-kmeans']):
            raise ValueError('Unknown algorithm : {}'.format(self.algo))

        # Define the compute engine; all methods take identical args and are generators.
        if self.parallel_backend == 'joblib':
            engine = self._process_with_joblib
        elif self.parallel_backend == 'multiprocessing':
            engine = self._process_with_multiprocessing
        else:
            engine = self._process_non_parallel

        gap_df = pd.DataFrame(
            {
                'n_clusters': [],
                'gap_value': [],
                'log_dispersion': [],
                'ref_log_dispersion': [],
                'sk': []
            }
        )

        # Calculate the gaps for each cluster count.
        for (gap_value, n_clusters, log_dispersion, ref_log_dispersion, sk) in engine(X, n_refs, cluster_array):

            # Assign this loop's gap statistic to gaps
            gap_df = gap_df.append(
                {
                    'n_clusters': n_clusters,
                    'gap_value': gap_value,
                    'log_dispersion': log_dispersion,
                    'ref_log_dispersion': ref_log_dispersion,
                    'sk': sk
                },
                ignore_index=True
            )

        # sort by ascending number of clusters
        self.gap_df = gap_df.sort_values(by='n_clusters', ascending=True).reset_index(drop=True)

        # compute gap(k) - gap(k-1) - s(k)
        self.gap_df["Dgap-s"] = self.gap_df["gap_value"].diff() - self.gap_df["sk"]
        
        if not(self.fname is None):
            self.gap_df.to_csv(self.fname, compression="bz2", index=None)

        return int(self.gap_df.loc[np.argmax(self.gap_df.gap_value.values)].n_clusters)

    @staticmethod
    def _calculate_dispersion(X: Union[pd.DataFrame, np.ndarray], labels: np.ndarray, centroids: np.ndarray) -> float:
        """
        Calculate the dispersion between actual points and their assigned centroids
        """
        disp = np.sum(np.sum([np.abs(inst - centroids[label]) ** 2 for inst, label in zip(X, labels)]))  # type: float
        return disp

    def _calculate_gap(self, X: Union[pd.DataFrame, np.ndarray], n_refs: int, n_clusters: int) -> Tuple[float, int]:
        """
        Calculate the gap value of the given data, n_refs, and number of clusters.
        Return the resutling gap value and n_clusters
        """
        # Holder for reference dispersion results
        ref_dispersions = np.zeros(n_refs)  # type: np.ndarray

        # For n_references, generate random sample and perform kmeans getting resulting dispersion of each loop
        # print(0, n_refs)
        for i in range(n_refs):
            # Create new random reference set
            random_data = random_sample_data(X, random_sampling=self.random_sampling)
            
            # Fit to it, getting the centroids and labels, and add to accumulated reference dispersions array.
            if self.algo == "kmeans2":
                centroids, labels = kmeans2(data=random_data,
                                            k=n_clusters,
                                            iter=10,
                                            minit='points')  # type: Tuple[np.ndarray, np.ndarray]
                dispersion = self._calculate_dispersion(X=random_data, labels=labels, centroids=centroids)  # type: float
            elif self.algo == "kmeans":
                centroids, dispersion = kmeans(obs=random_data,
                                               k_or_guess=n_clusters,
                                               iter=10)  # type: Tuple[np.ndarray, np.ndarray]
            elif self.algo == "skl-kmeans":
                km = KMeans(n_clusters=n_clusters, random_state=0)
                km.fit(random_data)
                centroids, labels = km.cluster_centers_, km.labels_        
                dispersion = km.inertia_
            elif self.algo == "sph-kmeans":
                skm = SphericalKMeans(n_clusters=n_clusters, random_state=0)
                skm.fit(random_data)
                centroids, labels = skm.cluster_centers_, skm.labels_        
                dispersion = skm.inertia_
                
            ref_dispersions[i] = dispersion

        # Fit cluster to original data and create dispersion calc.
        if self.algo == "kmeans2":
            centroids, labels = kmeans2(data=X, k=n_clusters, iter=10, minit='points')
            dispersion = self._calculate_dispersion(X=X, labels=labels, centroids=centroids)
        elif self.algo == "kmeans":
            centroids, dispersion = kmeans(obs=X,
                                           k_or_guess=n_clusters,
                                           iter=10)  # type: Tuple[np.ndarray, np.ndarray]
        elif self.algo == "skl-kmeans":
            km = KMeans(n_clusters=n_clusters, random_state=0)
            km.fit(X)
            centroids, labels = km.cluster_centers_, km.labels_        
            dispersion = km.inertia_
        elif self.algo == "sph-kmeans":      
            skm = SphericalKMeans(n_clusters=n_clusters, random_state=0)
            skm.fit(X)
            centroids, labels = skm.cluster_centers_, skm.labels_        
            dispersion = skm.inertia_
            
        # Calculate gap statistic
        ref_log_dispersion = np.mean(np.log(ref_dispersions))
        log_dispersion = np.log(dispersion)
        gap_value = ref_log_dispersion - log_dispersion

        # compute standard deviation
        sdk = np.sqrt(np.mean((np.log(ref_dispersions) - ref_log_dispersion) ** 2.))
        sk = np.sqrt(1. + 1. / n_refs) * sdk
            
        return gap_value, int(n_clusters), log_dispersion, ref_log_dispersion, sk

    def _process_with_joblib(self, X: Union[pd.DataFrame, np.ndarray], n_refs: int, cluster_array: np.ndarray):
        """
        Process calling of .calculate_gap() method using the joblib backend
        """
        with Parallel(n_jobs=self.n_jobs) as parallel:
            for gap_value, n_clusters, log_dispersion, ref_log_dispersion, sk in parallel(delayed(self._calculate_gap)(X, n_refs, n_clusters)
                                                  for n_clusters in cluster_array):
                yield (gap_value, n_clusters, log_dispersion, ref_log_dispersion, sk)

    def _process_with_multiprocessing(self, X: Union[pd.DataFrame, np.ndarray], n_refs: int, cluster_array: np.ndarray):
        """
        Process calling of .calculate_gap() method using the multiprocessing library
        """
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:

            jobs = [executor.submit(self._calculate_gap, X, n_refs, n_clusters)
                    for n_clusters in cluster_array
            ]

            for future in as_completed(jobs):
                gap_value, k, log_dispersion, ref_log_dispersion, sk = future.result()
                yield (gap_value, k, log_dispersion, ref_log_dispersion, sk)

    def _process_non_parallel(self, X: Union[pd.DataFrame, np.ndarray], n_refs: int, cluster_array: np.ndarray):
        """
        Process calling of .calculate_gap() method using no parallel backend; simple for loop generator
        """
        for gap_value, n_clusters, log_dispersion, ref_log_dispersion, sk in [self._calculate_gap(X, n_refs, n_clusters)
                                      for n_clusters in cluster_array]:
            yield (gap_value, n_clusters, log_dispersion, ref_log_dispersion, sk)

    def __str__(self):
        return 'OptimalK(n_jobs={}, parallel_backend="{}", algorithm="{}")'.format(self.n_jobs, self.parallel_backend, self.algo)

    def __repr__(self):
        return self.__str__()

    def _repr_html_(self):
        return '<p>{}</p>'.format(self.__str__())
