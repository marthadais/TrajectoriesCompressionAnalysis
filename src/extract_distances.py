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


def avg_std_dict_data(x, dim_set):
    """
    It computes the average and the standard deviation of a dict dataset considering a set of atributes
    :param x: the dataset in dict format
    :param dim_set: a list of the attributes to be computed
    :return: average and standard deviation
    """
    avg = {}
    std = {}
    maxv = {}
    for dim in dim_set:
        aux = np.concatenate([x[k].get(dim) for k in x])
        avg[dim] = aux.mean()
        std[dim] = aux.std()
        maxv[dim] = aux.max()
    return avg, std, maxv


def normalize(x, dim_set, verbose=True, znorm=True, centralize=False, norm_geo=True):
    """
    Computes Z-normalization or centralization of a dict dataset for a set of attributes
    :param x: dict dataset
    :param dim_set: set of attributes
    :param verbose: if True, print comments
    :param znorm: if True, it computes the z-normalization
    :param centralize: it True, it computes the centralization
    :return: normalized dict dataset
    """
    if verbose:
        print(f"Normalizing dataset")
    avg, std, maxv = avg_std_dict_data(x, dim_set)

    ids = list(x.keys())
    for id_a in range(len(ids)):
        # normalize features
        if znorm:
            for dim in dim_set:
                x[ids[id_a]][dim] = (x[ids[id_a]][dim]-avg[dim]) / std[dim]
        elif centralize:
            for dim in dim_set:
                x[ids[id_a]][dim] = x[ids[id_a]][dim]-avg[dim]
        elif norm_geo:
            for dim in dim_set:
                if (dim == 'lat') or (dim == 'LAT'):
                    x[ids[id_a]][dim] = x[ids[id_a]][dim]/90
                elif (dim == 'lon') or (dim == 'LON'):
                    x[ids[id_a]][dim] = x[ids[id_a]][dim]/180
                else:
                    # x[ids[id_a]][dim] = x[ids[id_a]][dim]/maxv[dim]
                    x[ids[id_a]][dim] = (x[ids[id_a]][dim]-avg[dim]) / std[dim]

    return x


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
        self.dm = None
        self.coeffs = None
        self.measures = None
        # self.num_cores = 2*(multiprocessing.cpu_count()//3)
        self.num_cores = 3
        if 'njobs' in args.keys():
            self.num_cores = args['njobs']

        self.features_opt = 'dtw'
        if 'features_opt' in args.keys():
            self.features_opt = args['features_opt']

        self._dim_set = ['lat', 'lon']
        if 'dim_set' in args.keys():
            self._dim_set = args['dim_set']

        self._znorm = False
        if 'znorm' in args.keys():
            self._znorm = args['znorm']

        self._centralize = False
        if 'centralize' in args.keys():
            self._centralize = args['centralize']

        self._normalizationGeo = True
        if 'norm_geo' in args.keys():
            self._normalizationGeo = args['norm_geo']

        self.dataset_norm = dataset
        if self._znorm or self._centralize or self._normalizationGeo:
            self.dataset_norm = normalize(self.dataset_norm, self._dim_set, verbose=verbose, znorm=self._znorm,
                                          centralize=self._centralize, norm_geo=self._normalizationGeo)
        self._ids = list(self.dataset_norm.keys())

        # methods parameters
        self.metric = 'haversine'
        if 'metric' in args.keys():
            self.metric = args['metric']

        _metrics_dict = self.create_data_dict()
        _metrics_dict[self.features_opt]()

        # saving features
        if 'folder' in args.keys():
            self.path = args['folder']

            if not os.path.exists(self.path):
                os.makedirs(self.path)

            pickle.dump(self.dm, open(f'{self.path}/features_distance.p', 'wb'))
            df_features = pd.DataFrame(self.dm)
            df_features.to_csv(f'{self.path}/features_distance.csv')
            self.dm_path = f'{self.path}/features_distance.p'


    def mdtw(self):
        dist_matrix = {}
        for id_a in range(len(self._ids)):
            dist_matrix[self._ids[id_a]] = {}

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
        while id_a < len(self._ids):
            if self.verbose:
                print(f"DTW: {id_a} of {len(self._ids)}")
            dist_matrix[self._ids[id_a]][self._ids[id_a]] = 0
            # trajectory a
            s_a = [self.dataset_norm[self._ids[id_a]][dim] for dim in self._dim_set]
            Parallel(n_jobs=self.num_cores, require='sharedmem')(delayed(self._mdtw_func)(id_b, id_a, s_a, dist_matrix) for id_b in list(range(id_a + 1, len(self._ids))))

            id_a = id_a + 1
            pickle.dump(dist_matrix, open(f'save_state/dtw_dist_matrix_matrix.p', 'wb'))
            pickle.dump(id_a, open(f'save_state/dtw_id_a_matrix.p', 'wb'))

        # delete save state
        os.remove(f'save_state/dtw_dist_matrix_matrix.p')
        os.remove(f'save_state/dtw_id_a_matrix.p')

        dist_matrix = dict_reorder(dist_matrix)
        dm = np.array([list(item.values()) for item in dist_matrix.values()])
        self.dm = dm/dm.max()

    def dmd(self):
        dist_matrix = {}
        for id_a in range(len(self._ids)):
            dist_matrix[self._ids[id_a]] = {}

        # save state
        if os.path.exists(f'save_state/md_dist_matrix_matrix.p'):
            dist_matrix = pickle.load(open(f'save_state/md_dist_matrix_matrix.p', 'rb'))
            id_a = pickle.load(open(f'save_state/md_id_a_matrix.p', 'rb'))
            print(dist_matrix)
        else:
            id_a = 0
            pickle.dump(dist_matrix, open(f'save_state/md_dist_matrix_matrix.p', 'wb'))
            pickle.dump(id_a, open(f'save_state/md_id_a_matrix.p', 'wb'))

        # for id_a in range(len(self._ids)):
        while id_a < len(self._ids):
            if self.verbose:
                print(f"MD: {id_a} of {len(self._ids)}")
            dist_matrix[self._ids[id_a]][self._ids[id_a]] = 0
            # trajectory a
            s_a = [self.dataset_norm[self._ids[id_a]][dim] for dim in self._dim_set]
            Parallel(n_jobs=self.num_cores, require='sharedmem')(delayed(self._md_func)(id_b, id_a, s_a, dist_matrix) for id_b in list(range(id_a + 1, len(self._ids))))

            id_a = id_a + 1
            pickle.dump(dist_matrix, open(f'save_state/md_dist_matrix_matrix.p', 'wb'))
            pickle.dump(id_a, open(f'save_state/md_id_a_matrix.p', 'wb'))

        # delete save state
        os.remove(f'save_state/md_dist_matrix_matrix.p')
        os.remove(f'save_state/md_id_a_matrix.p')

        dist_matrix = dict_reorder(dist_matrix)
        dm = np.array([list(item.values()) for item in dist_matrix.values()])
        self.dm = dm/dm.max()

    ### functions to parallelize ###
    def _mdtw_func(self, id_b, id_a, s_a, dist_matrix):
        # trajectory b
        s_b = [self.dataset_norm[self._ids[id_b]][dim] for dim in self._dim_set]
        # compute distance
        dist_matrix[self._ids[id_a]][self._ids[id_b]] = fastdtw(np.array(s_a).T, np.array(s_b).T, dist=haversine)[0]
        dist_matrix[self._ids[id_b]][self._ids[id_a]] = dist_matrix[self._ids[id_a]][self._ids[id_b]]

    def _md_func(self, id_b, id_a, s_a, dist_matrix):
        if self.verbose:
            print(f"\tComputing {id_a} with {id_b} of {len(self._ids)}")
        s_b = [self.dataset_norm[self._ids[id_b]][dim] for dim in self._dim_set]
        # compute distance
        dist_matrix[self._ids[id_a]][self._ids[id_b]] = MD(np.array(s_a).T, np.array(s_b).T)
        dist_matrix[self._ids[id_b]][self._ids[id_a]] = dist_matrix[self._ids[id_a]][self._ids[id_b]]


    def create_data_dict(self):
        return {'dtw': self.mdtw,
                'md': self.dmd}


