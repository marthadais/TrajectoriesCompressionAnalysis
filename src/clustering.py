from sklearn.cluster import DBSCAN, SpectralClustering, AgglomerativeClustering
import os, pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
import src.statistics as st
from scipy.cluster.hierarchy import dendrogram
import time


def my_DBSCAN(data, **args):
    """
    It generates the DBSCAN clustering
    :param data: the distance matrix
    :return: the clustering results
    """
    min_samples = 2
    if 'min_samples' in args.keys():
        min_samples = args['min_samples']

    eps = 3
    if 'eps' in args.keys():
        eps = args['eps']

    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    clustering.fit(data)

    return clustering


def my_hierarchical(data, **args):
    """
    It generates the hierarchical clustering
    :param data: the distance matrix
    :return: the clustering results
    """
    linkage = 'average'
    if 'linkage' in args.keys():
        linkage = args['linkage']

    k = 5
    if 'k' in args.keys():
        k = args['k']

    thr = None
    if 'thr' in args.keys():
        thr = args['thr']
        k = None

    data[np.isnan(data)] = 0
    clustering = AgglomerativeClustering(n_clusters=k, linkage=linkage, affinity='precomputed', distance_threshold=thr)
    clustering.fit(data)

    return clustering


def my_spectral(data, **args):
    """
    It generates the spectral clustering
    :param data: the distance matrix
    :return: the clustering results
    """
    k = 5
    if 'k' in args.keys():
        k = args['k']

    delta = 1
    if 'delta' in args.keys():
        delta = args['delta']

    n_eigenvectors = 3
    if 'n_eigenvectors' in args.keys():
        n_eigenvectors = args['n_eigenvectors']

    # convert distance to similarity
    data = 1 - data
    # data = np.exp(- data ** 2 / (2. * delta ** 2))
    clustering = SpectralClustering(n_clusters=k, n_components=n_eigenvectors, assign_labels="kmeans", affinity='precomputed', random_state=42, n_init=1)
    clustering.fit(data)

    return clustering


def plot_dendrogram(dm, folder):
    """
    It generates the dendrogram for hierarchical clustering.
    :param dm: the distance matrix
    :param folder: the folder path to save the image
    """
    model = my_hierarchical(dm, thr=0.2)
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    plt.figure()
    dendrogram(linkage_matrix, truncate_mode='lastp', p=25, labels=np.repeat('(1)', linkage_matrix.shape[0]+1), leaf_rotation=90, leaf_font_size=14)

    # change the fontsize of the xtick and ytick labels and axes
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)
    plt.rc('axes', labelsize=15)

    plt.ylabel('Distance')
    plt.xlabel('Number of instances in each cluster.')
    plt.savefig(f'{folder}/dendrogram.png', bbox_inches='tight')
    plt.close()


def create_algorithms_dict():
    """
    Dictionary of clustering options.
    """
    return {'dbscan': my_DBSCAN,
            'hierarchical': my_hierarchical,
            'spectral': my_spectral}


class Clustering:
    def __init__(self, ais_data_path, distance_matrix_path, verbose=True, **args):
        """
        It receives the preprocessed DCAIS dataset in dict format.
        It applies the selected model on the trajectories and compute the euclidean distance.
        :param ais_data_path: the path were is the dataset
        :param distance_matrix_path: the path were is the distance matrix of the dataset
        :param verbose: if True, it shows the messages (Default: True).
        """
        self._alg_dict = create_algorithms_dict()
        self.ais_data_path = ais_data_path
        self._verbose = verbose
        self.dm = abs(pickle.load(open(distance_matrix_path, 'rb')))
        self._model = None
        self.labels = None
        self.SC = None
        self.SC_cluster_mean = None
        self.SC_sample = None

        self.cluster_algorithm = 'dbscan'
        if 'cluster_algorithm' in args.keys():
            self.cluster_algorithm = args['cluster_algorithm']

        self.eps = None
        if 'eps' in args.keys():
            self.eps = args['eps']

        self._k = None
        if 'k' in args.keys():
            self._k = args['k']

        self._min_samples = 3
        if 'min_samples' in args.keys():
            self._min_samples = args['min_samples']

        self._linkage = 'average'
        if 'linkage' in args.keys():
            self._linkage = args['linkage']

        if 'norm_dist' in args.keys():
            if args['norm_dist']:
                self.dm = self.dm/self.dm.max()

        # saving features
        self.path = None
        self.labels_file_path = None
        self.results_file_path = None
        if 'folder' in args.keys():
            aux = args['folder']
            self.path = f'{aux}/{self.cluster_algorithm}'
            if self.cluster_algorithm == 'hierarchical':
                self.path = f'{aux}/{self.cluster_algorithm}-{self._linkage}'

            if not os.path.exists(self.path):
                os.makedirs(self.path)

            if self.cluster_algorithm == 'dbscan':
                self.results_file_path = f'{self.path}/{self.cluster_algorithm}_{self.eps}.csv'
                self.labels_file_path = f'{self.path}/labels_{self.cluster_algorithm}_{self.eps}.csv'
                self.time_path = f'{self.path}/time_{self.cluster_algorithm}_{self.eps}.csv'
            elif self.cluster_algorithm == 'hierarchical':
                self.results_file_path = f'{self.path}/{self.cluster_algorithm}_{self._k}_{self._linkage}.csv'
                self.labels_file_path = f'{self.path}/labels_{self.cluster_algorithm}_{self._k}_{self._linkage}.csv'
                self.time_path = f'{self.path}/time_{self.cluster_algorithm}_{self._k}_{self._linkage}.csv'
            else:
                self.results_file_path = f'{self.path}/{self.cluster_algorithm}_{self._k}.csv'
                self.labels_file_path = f'{self.path}/labels_{self.cluster_algorithm}_{self._k}.csv'
                self.time_path = f'{self.path}/time_{self.cluster_algorithm}_{self._k}.csv'

        if not os.path.exists(self.results_file_path):
            t0 = time.time_ns()
            self.computer_clustering()
            t1 = time.time_ns() - t0
            self.time_elapsed = t1
            pickle.dump(self.time_elapsed, open(self.time_path, 'wb'))
            if self.cluster_algorithm == 'hierarchical':
                plot_dendrogram(self.dm, self.path)
        else:
            self.time_elapsed = pickle.load(open(self.time_path, 'rb'))
            self.labels = pd.read_csv(self.labels_file_path).Clusters

    def computer_clustering(self):
        """
        It computes the clustering algorithm selected.
        """
        if self._verbose:
            print(f'Clustering data using {self.cluster_algorithm}')
        if self.eps is None:
            if self.cluster_algorithm == 'dbscan':
                self._estimating_epsilon()

        if self._k is None:
            if self.cluster_algorithm != 'dbscan':
                self._estimate_number_clusters()

        self._model = self._alg_dict[self.cluster_algorithm](self.dm, metric='precomputed', eps=self.eps,
                                                             k=self._k,
                                                             linkage=self._linkage, min_samples=self._min_samples)
        self.labels = self._model.labels_
        self.silhouette()
        if self.path is not None:
            if self._verbose:
                print('Saving in csv file')
            self._agg_cluster_labels()

        print('Computing statistics...')
        st.file_statistics(self.results_file_path, self.path)

    def silhouette(self):
        """
        It computes the silhouette measure.
        """
        if len(np.unique(self.labels)) > 1:
            self.SC_sample = metrics.silhouette_samples(self.dm, labels=self.labels, metric='precomputed')
            self.SC = metrics.silhouette_score(self.dm, labels=self.labels, metric='precomputed')
        else:
            self.SC_sample = np.zeros(self.dm.shape[0])
            self.SC = 0

    def get_clustering_options(self):
        """
        It list all clustering options.
        :return: list with clustering options
        """
        return list(self._alg_dict.keys())

    def _estimating_epsilon(self):
        """
        It estimates the epsilon for DBSCAN.
        """
        if self._verbose:
            print(f'\testimating eps...')
        distances = np.sort(self.dm, axis=1)
        distances = np.sort(distances)
        distances = distances[:, 1:(self._min_samples+1)]
        distances = np.sort(distances.ravel())
        pickle.dump(distances, open(f'{self.path}/sorted_distances.p', 'wb'))

        # change the fontsize of the xtick and ytick labels and axes
        plt.rc('xtick', labelsize=15)
        plt.rc('ytick', labelsize=15)
        plt.rc('axes', labelsize=15)

        plt.plot(range(0, len(distances)), distances, marker='o')
        plt.xlabel('Number of instances')
        plt.ylabel('Distance')
        plt.savefig(f'{self.path}/sorted_distances.png',
                    bbox_inches='tight')
        plt.close()

        ids = (np.where(distances >= distances.mean())[0][0])
        self.eps = distances[ids]

    def _estimate_number_clusters(self):
        """
        It estimates the number of clusters for Hierarchical and Spectral clustering.
        """
        if self._verbose:
            print('\tAnalysing the best number of clusters')
        K_min = 2
        K = 20
        if self.dm.shape[0] < K:
            K = self.dm.shape[0]
        sc = np.repeat(-9.9, K+1)
        for k in range(K_min, K+1):
            if self._verbose:
                print(f'\t\trunning with {k} number of clusters')
            self._k = k
            self._model = self._alg_dict[self.cluster_algorithm](self.dm, metric='precomputed', eps=self.eps,
                                                                 k=self._k,
                                                                 linkage=self._linkage, min_samples=self._min_samples)
            self.labels = self._model.labels_
            self.silhouette()
            sc[k] = self.SC

        self._k = int(np.where(sc == sc.max())[0][0])

        if self.path is not None:
            # change the fontsize of the xtick and ytick labels and axes
            plt.rc('xtick', labelsize=15)
            plt.rc('ytick', labelsize=15)
            plt.rc('axes', labelsize=15)

            plt.plot(range(K_min, K+1), sc[K_min:], marker='o')
            plt.xlabel('Number of clusters')
            plt.ylabel('Silhouette score')
            # plt.title(f'Silhouette Scores for {self.cluster_algorithm}')
            plt.savefig(f'{self.path}/{self.cluster_algorithm}_{self._linkage}_silhoutte_line_graph.png', bbox_inches='tight')
            plt.close()

    def _agg_cluster_labels(self):
        """
        It includes the label information provided by the Clustering algorithm into the dataset.
        """
        data = pd.read_csv(self.ais_data_path)

        labels = pd.DataFrame([self.labels], columns=data['trajectory'].unique()).to_dict('records')[0]
        aux = data['trajectory']
        aux = aux.map(labels)
        aux.name = 'Clusters'
        cluster_dataset = pd.concat([data, aux], axis=1)

        sc = pd.DataFrame([self.SC_sample], columns=data['trajectory'].unique()).to_dict('records')[0]
        aux = data['trajectory']
        aux = aux.map(sc)
        aux.name = 'silhouette'
        cluster_dataset = pd.concat([cluster_dataset, aux], axis=1)
        labels_mmsi = cluster_dataset[['mmsi', 'trajectory', 'Clusters']].drop_duplicates()

        aux_data = cluster_dataset[['trajectory', 'silhouette']].copy()
        aux_data.drop_duplicates(['trajectory'], inplace=True)

        limit_std = aux_data['silhouette'].mean() - 3 * aux_data['silhouette'].std()
        aux[aux > limit_std] = 1
        aux[aux <= limit_std] = -1
        aux.name = 'scores-3std'
        cluster_dataset['threshold_std'] = limit_std
        cluster_dataset = pd.concat([cluster_dataset, aux], axis=1)

        cluster_dataset = cluster_dataset.assign(Cl_Silhouette=self.SC)

        cluster_dataset.to_csv(self.results_file_path)
        labels_mmsi.to_csv(self.labels_file_path)
    """
    It reads the distance matrix and execute the clustering algortihm.
    """
