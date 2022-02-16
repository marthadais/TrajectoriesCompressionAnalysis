from haversine import haversine
import numpy as np
import pandas as pd
from fastdtw import fastdtw
import pickle
import os
from joblib import Parallel, delayed
from itertools import product


def dict_reorder(x):
    """
    It reorder the dict values
    :param x: the data on dict format
    :return: dict ordered
    """
    return {k: dict_reorder(v) if isinstance(v, dict) else v for k, v in sorted(x.items())}


def MD(a, b):
    """
    Merge Distance for GPS
    :param a: trajectory A
    :param b: trajectory B
    :return: merge

    References:
    [1] Ismail, Anas, and Antoine Vigneron. "A new trajectory similarity measure for GPS data." Proceedings of the 6th ACM SIGSPATIAL International Workshop on GeoStreaming. 2015.
    [2] Li, Huanhuan, et al. "Spatio-temporal vessel trajectory clustering based on data mapping and density." IEEE Access 6 (2018): 58939-58954.
    """
    m = len(a)
    n = len(b)
    A = np.zeros([m, n])
    B = np.zeros([m, n])

    a_dist = [haversine(a[i-1], a[i]) for i in range(1, m)]
    b_dist = [haversine(b[i-1], b[i]) for i in range(1, n)]
    # ab_dist = cdist(a, b, metric=haversine)

    # initializing bounderies
    i = 0
    a_d = 0
    for j in range(n):
        k = j - 1
        if k > 0 and k < n:
            a_d = a_d + b_dist[k-1]
        A[i, j] = a_d + haversine(b[j], a[0])

    j = 0
    b_d = 0
    for i in range(m):
        k = i - 1
        if k > 0 and k < n:
            b_d = b_d + a_dist[k - 1]
        B[i, j] = b_d + haversine(a[i], b[0])

    j = 0
    for i in range(1, m):
        # A[i, j] = min(A[i - 1, j] + a_dist[i - 1], B[i - 1, j] + ab_dist[i, j])
        A[i, j] = min(A[i - 1, j] + a_dist[i - 1], B[i - 1, j] + haversine(a[i], b[j]))
    i = 0
    for j in range(1, n):
        # B[i, j] = min(A[i, j - 1] + ab_dist[i, j], B[i, j - 1] + b_dist[j - 1])
        B[i, j] = min(A[i, j - 1] + haversine(b[j], a[i]), B[i, j - 1] + b_dist[j - 1])

    # computing distances
    for i, j in product(range(1, m), range(1, n)):
        # A[i, j] = min(A[i-1, j] + a_dist[i-1], B[i-1, j] + ab_dist[i, j])
        A[i, j] = min(A[i-1, j] + a_dist[i-1], B[i-1, j] + haversine(a[i], b[j]))
        # B[i, j] = min(A[i, j-1] + ab_dist[i, j], B[i, j-1] + b_dist[j-1])
        B[i, j] = min(A[i, j-1] + haversine(b[j], a[i]), B[i, j-1] + b_dist[j-1])

    # getting the merge distance
    md_dist = min(A[-1, -1], B[-1, -1])
    if md_dist != 0:
        # md_dist = (2*md_dist) / (A[-1,-1] + B[-1,-1])
        md_dist = ((2*md_dist) / (np.sum(a_dist)+np.sum(b_dist)))-1
    return md_dist


### functions to parallelize ###
def _dist_func(dataset, metric, mmsis, dim_set, id_b, id_a, s_a, dist_matrix):
    # trajectory b
    s_b = [dataset[mmsis[id_b]][dim] for dim in dim_set]
    # compute distance
    if metric == 'dtw':
        dist_matrix[mmsis[id_a]][mmsis[id_b]] = fastdtw(np.array(s_a).T, np.array(s_b).T, dist=haversine)[0]
    else:
        dist_matrix[mmsis[id_a]][mmsis[id_b]] = MD(np.array(s_a).T, np.array(s_b).T)
    dist_matrix[mmsis[id_b]][mmsis[id_a]] = dist_matrix[mmsis[id_a]][mmsis[id_b]]


def compute_distance_matrix(dataset, path, verbose=True, njobs=3, metric='dtw'):
    if not os.path.exists(path):
        _dim_set = ['lat', 'lon']
        _mmsis = list(dataset.keys())

        dist_matrix = {}
        for id_a in range(len(_mmsis)):
            dist_matrix[_mmsis[id_a]] = {}

        for id_a in range(len(_mmsis)):
            if verbose:
                print(f"{metric}: {id_a} of {len(_mmsis)}")
            dist_matrix[_mmsis[id_a]][_mmsis[id_a]] = 0
            # trajectory a
            s_a = [dataset[_mmsis[id_a]][dim] for dim in _dim_set]
            Parallel(n_jobs=njobs, require='sharedmem')(delayed(_dist_func)(dataset, metric, _mmsis, _dim_set, id_b, id_a,
                                                                            s_a, dist_matrix)
                                                        for id_b in list(range(id_a + 1, len(_mmsis))))

        dist_matrix = dict_reorder(dist_matrix)
        dm = np.array([list(item.values()) for item in dist_matrix.values()])

        # saving features
        os.makedirs(path)
        pickle.dump(dm, open(f'{path}/features_distance.p', 'wb'))
    else:
        print('Distances already computed.')
    dm_path = f'{path}/features_distance.p'
    return dm_path
