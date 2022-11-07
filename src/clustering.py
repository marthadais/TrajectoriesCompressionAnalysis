from sklearn.cluster import DBSCAN
import os, pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import hdbscan

class Clustering:
    def __init__(self, ais_data_path, distance_matrix_path, folder, verbose=True, **args):
        """
        It receives the preprocessed DCAIS dataset in dict format.
        It applies the selected model on the trajectories and compute the euclidean distance.
        :param ais_data_path: the path were is the dataset
        :param distance_matrix_path: the path were is the distance matrix of the dataset
        :param verbose: if True, it shows the messages (Default: True).
        """
        self.ais_data_path = ais_data_path
        self._verbose = verbose
        self.dm = abs(pickle.load(open(distance_matrix_path, 'rb')))
        self._model = None
        self.labels = None

        self.dm[np.isinf(self.dm)] = self.dm[~np.isinf(self.dm)].max() + 1
        if 'norm_dist' in args.keys():
            if args['norm_dist']:
                self.dm[self.dm < 0] = 0
                self.dm = self.dm/abs(self.dm).max().max()

        # saving features
        self.path = f'{folder}/hdbscan'
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        self.results_file_path = f'{self.path}/hdbscan.csv'
        self.labels_file_path = f'{self.path}/labels_hdbscan.csv'
        self.time_path = f'{self.path}/time_hdbscan.csv'

        # if not os.path.exists(self.results_file_path):
        t0 = time.time()
        self.computer_clustering()
        t1 = time.time() - t0
        self.time_elapsed = t1
        pickle.dump(self.time_elapsed, open(self.time_path, 'wb'))
        # else:
        #     self.time_elapsed = pickle.load(open(self.time_path, 'rb'))
        #     self.labels = pd.read_csv(self.labels_file_path).Clusters

    def computer_clustering(self):
        """
        It computes the clustering algorithm selected.
        """
        if self._verbose:
            print(f'Clustering data using HDBSCAN')

        # self._model = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1, cluster_selection_epsilon=0.003, metric='precomputed')
        self._model = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=1, cluster_selection_epsilon=0.003, allow_single_cluster=True, metric='precomputed')
        model = self._model.fit(self.dm)
        self.labels = model.labels_

        self._agg_cluster_labels()

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

    def _agg_cluster_labels(self):
        """
        It includes the label information provided by the Clustering algorithm into the dataset.
        """
        data = pd.read_csv(self.ais_data_path)
        if not 'trips' in data.columns:
            data = data.rename(columns={'trajectory': 'trips'})
        labels = pd.DataFrame([self.labels], columns=data['trips'].unique()).to_dict('records')[0]
        aux = data['trips']
        aux = aux.map(labels)
        aux.name = 'Clusters'
        cluster_dataset = pd.concat([data, aux], axis=1)
        labels_mmsi = cluster_dataset[['mmsi', 'trips', 'Clusters']].drop_duplicates()
        cluster_dataset.to_csv(self.results_file_path)
        labels_mmsi.to_csv(self.labels_file_path)

