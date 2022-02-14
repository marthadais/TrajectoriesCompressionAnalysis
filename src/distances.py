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


class DistanceMatrix:
    def __init__(self, dataset, verbose=True, **args):
        self.verbose = verbose
        self.dataset = dataset
        self.dm = None
        # self.num_cores = 2*(multiprocessing.cpu_count()//3)
        self.num_cores = 3
        if 'njobs' in args.keys():
            self.num_cores = args['njobs']

        self.features_opt = 'dtw'
        if 'features_opt' in args.keys():
            self.features_opt = args['features_opt']

        self._dim_set = ['lat', 'lon']
        self._mmsis = list(self.dataset.keys())
        self._calc_dists()

        # saving features
        if 'folder' in args.keys():
            self.path = args['folder']

            if not os.path.exists(self.path):
                os.makedirs(self.path)

            pickle.dump(self.dm, open(f'{self.path}/features_distance.p', 'wb'))
            df_features = pd.DataFrame(self.dm)
            df_features.to_csv(f'{self.path}/features_distance.csv')
            self.dm_path = f'{self.path}/features_distance.p'

    def _calc_dists(self):
        dist_matrix = {}
        for id_a in range(len(self._mmsis)):
            dist_matrix[self._mmsis[id_a]] = {}

        # save state
        if os.path.exists(f'save_state/dtw_dist_matrix_matrix.p'):
            dist_matrix = pickle.load(open(f'save_state/dtw_dist_matrix_matrix.p', 'rb'))
            id_a = pickle.load(open(f'save_state/dtw_id_a_matrix.p', 'rb'))
            print(dist_matrix)
        else:
            id_a = 0
            pickle.dump(dist_matrix, open(f'save_state/dtw_dist_matrix_matrix.p', 'wb'))
            pickle.dump(id_a, open(f'save_state/dtw_id_a_matrix.p', 'wb'))

        # for id_a in range(len(self._ids)):
        while id_a < len(self._mmsis):
            if self.verbose:
                print(f"{self.features_opt}: {id_a} of {len(self._mmsis)}")
            dist_matrix[self._mmsis[id_a]][self._mmsis[id_a]] = 0
            # trajectory a
            s_a = [self.dataset[self._mmsis[id_a]][dim] for dim in self._dim_set]
            Parallel(n_jobs=self.num_cores, require='sharedmem')(delayed(self._dist_func)(id_b, id_a, s_a, dist_matrix) for id_b in list(range(id_a + 1, len(self._mmsis))))

            id_a = id_a + 1
            pickle.dump(dist_matrix, open(f'save_state/dtw_dist_matrix_matrix.p', 'wb'))
            pickle.dump(id_a, open(f'save_state/dtw_id_a_matrix.p', 'wb'))

        # delete save state
        os.remove(f'save_state/dtw_dist_matrix_matrix.p')
        os.remove(f'save_state/dtw_id_a_matrix.p')

        dist_matrix = dict_reorder(dist_matrix)
        self.dm = np.array([list(item.values()) for item in dist_matrix.values()])

    ### functions to parallelize ###
    def _dist_func(self, id_b, id_a, s_a, dist_matrix):
        # trajectory b
        s_b = [self.dataset[self._mmsis[id_b]][dim] for dim in self._dim_set]
        # compute distance
        if self.features_opt == 'dtw':
            dist_matrix[self._mmsis[id_a]][self._mmsis[id_b]] = fastdtw(np.array(s_a).T, np.array(s_b).T, dist=haversine)[0]
        else:
            dist_matrix[self._mmsis[id_a]][self._mmsis[id_b]] = MD(np.array(s_a).T, np.array(s_b).T)
        dist_matrix[self._mmsis[id_b]][self._mmsis[id_a]] = dist_matrix[self._mmsis[id_a]][self._mmsis[id_b]]
