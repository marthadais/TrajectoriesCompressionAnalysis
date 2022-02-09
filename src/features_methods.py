import numpy as np
import math
import scipy
from itertools import product
from haversine import haversine


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


def divide_max_value(x):
    return x/x.max()


def avg_std_dict_data(x, dim_set):
    """
    Computes the average and the standard deviation of a dict dataset considering a set of atributes
    :param x:
    :param dim_set:
    :return: average and standard deviation
    """
    avg = {}
    std = {}
    for dim in dim_set:
        aux = np.concatenate([x[k].get(dim) for k in x])
        avg[dim] = aux.mean()
        std[dim] = aux.std()
    return avg, std


def normalize(x, dim_set, verbose=True, znorm=True, centralize=False):
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
    avg, std = avg_std_dict_data(x, dim_set)

    ids = list(x.keys())
    for id_a in range(len(ids)):
        # normalize features
        if znorm:
            for dim in dim_set:
                x[ids[id_a]][dim] = (x[ids[id_a]][dim]-avg[dim]) / std[dim]
        elif centralize:
            for dim in dim_set:
                x[ids[id_a]][dim] = x[ids[id_a]][dim]-avg[dim]

    return x


############ OU PROCESS ###########
def ou_process(t, x, start=None):
    """ OU (Ornstein-Uhlenbeck) process
        dX = -A(X-alpha)dt + v dB
        Maximum-likelihood estimator
        Piece of code from:
        https://github.com/jwergieluk/ou_noise/tree/c5eee685c8a80a079dd32c759df3b97e05ef51ef
    """

    if start is None:
        v = est_v_quadratic_variation(t, x)
        start = (0.5, np.mean(x), v)

    def error_fuc(theta):
        return -loglik(t, x, theta[0], theta[1], theta[2])

    start = np.array(start)
    result = scipy.optimize.minimize(error_fuc, start, method='L-BFGS-B',
                                     bounds=[(1e-6, None), (None, None), (1e-8, None)],
                                     options={'maxiter': 500, 'disp': False})
    return result.x


def est_v_quadratic_variation(t, x, weights=None):
    """ Estimate v using quadratic variation"""
    assert len(t) == x.shape[1]
    q = quadratic_variation(x, weights)
    return math.sqrt(q/(t[-1] - t[0]))


def quadratic_variation(x, weights=None):
    """ Realized quadratic variation of a path. The weights must sum up to one. """
    assert x.shape[1] > 1
    dx = np.diff(x)
    if weights is None:
        return np.sum(dx*dx)
    return x.shape[1]*np.sum(dx * dx * weights)


def loglik(t, x, mean_rev_speed, mean_rev_level, vola):
    """Calculates log likelihood of a path"""
    dt = np.diff(t)
    mu = mean(x[:, :-1], dt, mean_rev_speed, mean_rev_level)
    sigma = std(dt, mean_rev_speed, vola)
    return np.sum(scipy.stats.norm.logpdf(x[:, 1:], loc=mu, scale=sigma))


def mean(x0, t, mean_rev_speed, mean_rev_level):
    assert mean_rev_speed >= 0
    return x0 * np.exp(-mean_rev_speed * t) + (1.0 - np.exp(- mean_rev_speed * t)) * mean_rev_level


def std(t, mean_rev_speed, vola):
    return np.sqrt(variance(t, mean_rev_speed, vola))


def variance(t, mean_rev_speed, vola):
    assert mean_rev_speed >= 0
    assert vola >= 0
    return vola * vola * (1.0 - np.exp(- 2.0 * mean_rev_speed * t)) / (2 * mean_rev_speed)

