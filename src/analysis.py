import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from src.distances import compute_distance_matrix
from src.clustering import Clustering
from sklearn import metrics
import mantel
from matplotlib import rc
from preprocessing.compress_trajectories import compress_trips, get_raw_dataset
from sklearn.manifold import MDS


def get_time(path):
    """
    It reads and computes the total processing time of the distances calculations
     by getting the upper triangle of the matrix.

    :param path: path that contains the matrix with the processing time.
    :return: the total processing time
    """
    up = pickle.load(open(path, 'rb'))
    up = pd.DataFrame.from_dict(up)
    up[up.isna()] = 0
    up = up.to_numpy()
    up = up[np.triu_indices_from(up)]
    return up


def purity_score(y_true, y_pred):
    """
    It computes contingency matrix (also called confusion matrix) and the purity value.

    :param y_true: true labels.
    :param y_pred: predicted labels.
    :return: purity measure
    """
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


def lines_ca_score(folder, score, options, comp_lbl, col, lines_style, mark_size, line_size):
    """
    It plots the lines for the clustering score.

    :param folder: folder that contains the results.
    :param score: measure to evaluate the cluster.
    :param options: compression methods evaluated.
    :param comp_lbl: labels of the compression techniques.
    :param col: color vector.
    :param lines_style: line style.
    :param mark_size: size of the marks.
    :param line_size: size of the lines.
    """
    fig = plt.figure(figsize=(10, 8))
    i = 0
    for compress_opt in options:
        x = pd.read_csv(f'{folder}/clustering_{compress_opt}_{score}.csv', index_col=0)
        x.index = x.index.astype(str)
        plt.plot(x.iloc[x.shape[0]:None:-1], color=col[i], marker="p", linestyle=lines_style[i],
                 linewidth=line_size[i], markersize=mark_size[i], label=comp_lbl[compress_opt])
        i = i + 1
    plt.ylabel(f'{score.upper()}', fontsize=25)
    plt.xlabel('Factors', fontsize=25)
    plt.legend(fontsize=18)
    plt.xticks(range(len(x)), [r'$\frac{1}{128}$', r'$\frac{1}{64}$', r'$\frac{1}{32}$', r'$\frac{1}{16}$',
                   r'$\frac{1}{8}$', r'$\frac{1}{4}$', r'$\frac{1}{2}$', r'$1$', r'$1.5$', r'$2$'], fontsize=25)
    plt.yticks(fontsize=20)
    plt.ylim((0,1))
    plt.tight_layout()
    plt.savefig(f'{folder}/lines-clustering-{score}.png', bbox_inches='tight')
    plt.close()


def time_mean(folder, item, options, comp_lbl, col, lines_style, mark_size, line_size):
    """
    It plots the lines for the processing time.

    :param folder: folder that contains the results.
    :param item: rates or processing time.
    :param options: compression methods evaluated.
    :param comp_lbl: labels of the compression techniques.
    :param col: color vector.
    :param lines_style: line style.
    :param mark_size: size of the marks.
    :param line_size: size of the lines.
    """
    fig = plt.figure(figsize=(10, 8))
    i=0
    for compress_opt in options:
        x = pd.read_csv(f'{folder}/{compress_opt}-compression_{item}.csv')
        plt.plot(range(len(x.mean(axis=0))), x.mean(axis=0).iloc[x.shape[0]:None:-1], color=col[i], marker="p",
                 linestyle=lines_style[i], linewidth=line_size[i],
                 markersize=mark_size[i], label=comp_lbl[compress_opt])
        i = i+1
    plt.ylabel(f'Average of Compression {item}',fontsize=25)
    plt.xlabel('Factors',fontsize=25)
    plt.legend(fontsize=18)
    plt.xticks(range(len(x.mean(axis=0))), [r'$\frac{1}{128}$', r'$\frac{1}{64}$', r'$\frac{1}{32}$', r'$\frac{1}{16}$',
                   r'$\frac{1}{8}$', r'$\frac{1}{4}$', r'$\frac{1}{2}$', r'$1$', r'$1.5$', r'$2$'], fontsize=25)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.savefig(f'{folder}/lines-compression-{item}.png', bbox_inches='tight')
    plt.close()


def lines_compression(folder, metric='dtw'):
    """
    It plots the lines for the compression rates, scores, and processing time.

    :param folder: folder that contains the results.
    :param metric: distance measure selected.
    """
    # parameters for the plot
    rc('text', usetex=True)
    rc('text.latex', preamble=r'\usepackage{cmbright}')
    rc('font', size=25)
    rc('legend', fontsize=25)
    options = ['DP', 'TR', 'SP', 'TR_SP', 'SP_TR', 'DP_SP', 'SP_DP', 'TR_DP', 'DP_TR']
    comp_lbl = {'DP': 'DP', 'TR': 'TR', 'SP': 'SB', 'TR_SP': 'TR+SB', 'SP_TR': 'SB+TR', 'DP_SP': 'DP+SB',
                'SP_DP': 'SB+DP', 'TR_DP': 'TR+DP', 'DP_TR': 'DP+TR'}
    # col = ['crimson', 'blue', 'green', 'darkorange', 'black', 'violet', 'chocolate', 'blueviolet', 'olive']
    col = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'black', 'tab:purple', 'tab:brown',
           'tab:pink', 'tab:olive']
    lines_style = [(0, (3,1,1,1)), (0, (5, 1)), (0, (3, 5, 1, 5)), 'dotted',
                   (0, (1, 3)), 'dashdot', (0, (3, 3, 1, 3)), (0, (3, 1, 1, 1, 1, 1)), 'solid']
    mark_size = ['11', '11', '11', '9', '9', '6', '6', '3', '3']
    line_size = ['3', '3', '3', '2', '2', '1.5', '1.5', '1', '1']
    factors_str = [r'$\frac{1}{128}$', r'$\frac{1}{64}$', r'$\frac{1}{32}$', r'$\frac{1}{16}$',
                   r'$\frac{1}{8}$', r'$\frac{1}{4}$', r'$\frac{1}{2}$', r'$1$', r'$1.5$', r'$2$']

    # plot the compression rates
    time_mean(folder, 'rates', options, comp_lbl, col, lines_style, mark_size, line_size)

    # figure of the total processing time
    fig = plt.figure(figsize=(10, 8))
    i = 0
    for compress_opt in options:
        times_cl = pd.read_csv(f'{folder}/clustering_{compress_opt}_times.csv', index_col=0)
        times = pd.read_csv(f'{folder}/{metric}_{compress_opt}_times.csv')
        times = (times.sum(axis=0) + times_cl.T).T
        times_compression = pd.read_csv(f'{folder}/{compress_opt}-compression_times.csv')
        times_compression = times_compression
        times[1:] = (times[1:].T + times_compression.sum()).T.iloc[10:None:-1]
        plt.plot(times, color=col[i], marker="p", linestyle=lines_style[i],
                 linewidth=line_size[i], markersize=mark_size[i], label=comp_lbl[compress_opt])
        i = i + 1
    plt.ylabel('Processing Time (s)', fontsize=25)
    plt.xlabel('Factors', fontsize=25)
    plt.legend(fontsize=18)
    plt.xticks(range(len(times)), [r'Control'] + factors_str, fontsize=25)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.savefig(f'{folder}/lines-total-times.png', bbox_inches='tight')
    plt.close()

    # figure of the clustering purity
    lines_ca_score(folder, 'mh', options, comp_lbl, col, lines_style=lines_style,
                   mark_size=mark_size, line_size=line_size)
    lines_ca_score(folder, 'nmi', options, comp_lbl, col, lines_style=lines_style,
                   mark_size=mark_size, line_size=line_size)

    # figure of the pearson correlation
    fig = plt.figure(figsize=(10, 8))
    i = 0
    for compress_opt in options:
        measure = pd.read_csv(f'{folder}/measures_{metric}_{compress_opt}_times.csv', index_col=0)
        measure = measure.loc['mantel-corr']
        plt.plot(measure.iloc[measure.shape[0]:None:-1], color=col[i], marker="p", linestyle=lines_style[i],
                 linewidth=line_size[i], markersize=mark_size[i], label=comp_lbl[compress_opt])
        i = i + 1
    plt.ylabel('Mantel Correlation - Pearson', fontsize=25)
    plt.xlabel('Factors', fontsize=25)
    plt.legend(fontsize=18)
    plt.xticks(range(len(measure)), factors_str, fontsize=25)
    plt.yticks(fontsize=20)
    # plt.tight_layout()
    plt.savefig(f'{folder}/lines-measure-mantel.png', bbox_inches='tight')
    plt.close()


def factor_analysis(dataset_path, compress_opt, folder):
    """
    It evaluates each compression technique in the dataset for different factors.

    :param dataset_path: path of the dataset.
    :param compress_opt: compression technique under analysis.
    :param folder: folder to save the results.
    :return: compression rate and processing time
    """
    factors = [2, 1.5, 1, 1/2, 1/4, 1/8, 1/16, 1/32, 1/64, 1/128]
    rates = pd.DataFrame()
    times = pd.DataFrame()
    for i in factors:
        comp_dataset, comp_rate, comp_times = compress_trips(dataset_path, compress=compress_opt, alpha=i)
        comp_times = comp_times
        rates = pd.concat([rates, pd.DataFrame(comp_rate)], axis=1)
        times = pd.concat([times, pd.DataFrame(comp_times)], axis=1)
    rates.columns = [str(i) for i in factors]
    times.columns = [str(i) for i in factors]
    rates.to_csv(f'{folder}/{compress_opt}-compression_rates.csv', index=False)
    times.to_csv(f'{folder}/{compress_opt}-compression_times.csv', index=False)

    return rates, times


def factor_dist_analysis(dataset_path, compress_opt, folder, ncores=15, metric='dtw'):
    """
    It evaluates the distance matrix of each compression technique in the dataset for different factors.

    :param dataset_path: path of the dataset.
    :param compress_opt: compression technique under analysis.
    :param folder: folder to save the results.
    :param ncores: number of cores for parallel process (Default: 15).
    :param metric: distance measure selected (Default: 'dtw').
    :return: compression rate and processing time
    """
    factors = [2, 1.5, 1, 1/2, 1/4, 1/8, 1/16, 1/32, 1/64, 1/128]
    times = pd.DataFrame()
    # comparing distances
    measures = {}
    features_folder = f'{folder}/NO/'
    if not os.path.exists(features_folder):
        os.makedirs(features_folder)
    features_path, main_time = compute_distance_matrix(get_raw_dataset(dataset_path), features_folder, verbose=True,
                                                       njobs=ncores, metric=metric)

    dist_raw = pickle.load(open(features_path, 'rb'))
    print(dist_raw)
    dist_raw = dist_raw/dist_raw.max().max()
    dist_raw[np.isinf(dist_raw)] = dist_raw[~np.isinf(dist_raw)].max() + 1
    dist_raw[dist_raw < 0] = 0
    dist_raw = dist_raw/dist_raw.max().max()
    # mapData(dist_raw, 'NO')

    dist_raw_time = get_time(main_time)
    times = pd.concat([times, pd.DataFrame(dist_raw_time)], axis=1)
    for i in factors:
        comp_dataset, comp_rate, comp_times = compress_trips(dataset_path, compress=compress_opt, alpha=i)
        features_folder = f'{folder}/{compress_opt}-{i}/'
        if not os.path.exists(features_folder):
            os.makedirs(features_folder)
        print({features_folder})
        # DTW distances
        features_path, feature_time = compute_distance_matrix(comp_dataset, features_folder, verbose=True,
                                                                   njobs=ncores, metric=metric)
        dtw_factor = pickle.load(open(features_path, 'rb'))
        measures[i] = {}
        dtw_factor[np.isinf(dtw_factor)] = dtw_factor[~np.isinf(dtw_factor)].max() + 1
        dtw_factor[dtw_factor < 0] = 0
        dtw_factor = dtw_factor/dtw_factor.max().max()
        print(dtw_factor)
        # mapData(dtw_factor, f'{i}')
        measures[i]['mantel-corr'], measures[i]['mantel-pvalue'], _ = mantel.test(dist_raw, dtw_factor,
                                                                                  method='pearson', tail='upper')
        print(f"mantel - factor {i}: {measures[i]['mantel-corr']} - {measures[i]['mantel-pvalue']}")

        dtw_factor_time = get_time(feature_time)
        times = pd.concat([times, pd.DataFrame(dtw_factor_time)], axis=1)

    measures = pd.DataFrame(measures)
    measures.columns = [str(i) for i in factors]
    measures.to_csv(f'{folder}/measures_{metric}_{compress_opt}_times.csv')

    times.columns = ['no'] + [str(i) for i in factors]
    times.to_csv(f'{folder}/{metric}_{compress_opt}_times.csv', index=False)

    return measures


def factor_cluster_analysis(dataset_path, compress_opt, folder, ncores=15, metric='dtw', mcs=2):
    """
    It evaluates the clustering results of distance matrix computed accordingly with each compression technique in the
    dataset using different factors.

    :param dataset_path: path of the dataset.
    :param compress_opt: compression technique under analysis.
    :param folder: folder to save the results.
    :param ncores: number of cores for parallel process (Default: 15).
    :param metric: distance measure selected (Default: 'dtw').
    :return: compression rate and processing time
    """
    factors = [2, 1.5, 1, 1/2, 1/4, 1/8, 1/16, 1/32, 1/64, 1/128]
    measures_mh = {}
    measures_nmi = {}
    times_cl = {}

    # comparing distances
    features_folder = f'{folder}/NO/'
    if not os.path.exists(features_folder):
        os.makedirs(features_folder)
    features_path, _ = compute_distance_matrix(get_raw_dataset(dataset_path), features_folder, verbose=True,
                                               njobs=ncores, metric=metric)

    #clustering
    model = Clustering(ais_data_path=dataset_path, distance_matrix_path=features_path, folder=features_folder,
                       minClusterSize=mcs, norm_dist=True)
    times_cl['no'] = model.time_elapsed
    labels_raw = model.labels
    for i in factors:
        comp_dataset, comp_rate, comp_times = compress_trips(dataset_path, compress=compress_opt, alpha=i)
        features_folder = f'{folder}/{compress_opt}-{i}/'
        if not os.path.exists(features_folder):
            os.makedirs(features_folder)
        print({features_folder})
        # DTW distances
        features_path, feature_time = compute_distance_matrix(comp_dataset, features_folder, verbose=True,
                                                                   njobs=ncores, metric=metric)
        # clustering
        model = Clustering(ais_data_path=dataset_path, distance_matrix_path=features_path, folder=features_folder,
                           minClusterSize=mcs, norm_dist=True)
        times_cl[str(i)] = model.time_elapsed
        labels_factor = model.labels

        # measures
        measures_purity = purity_score(labels_raw, labels_factor)
        measures_coverage = purity_score(labels_factor, labels_raw)
        measures_mh[str(i)] = 2/(1/measures_purity + 1/measures_coverage)
        measures_nmi[str(i)] = metrics.normalized_mutual_info_score(labels_raw, labels_factor)

        print(f'raw = {labels_raw.max()}, {compress_opt}-{i} = {labels_factor.max()} - NMI = {measures_nmi[str(i)]}')
        print(labels_raw)
        print(labels_factor)

    measures_mh = pd.Series(measures_mh)
    measures_mh.to_csv(f'{folder}/clustering_{compress_opt}_mh.csv')
    measures_nmi = pd.Series(measures_nmi)
    measures_nmi.to_csv(f'{folder}/clustering_{compress_opt}_nmi.csv')

    times_cl = pd.Series(times_cl)
    times_cl.columns = ['no'] + [str(i) for i in factors]
    times_cl.to_csv(f'{folder}/clustering_{compress_opt}_times.csv')

    return measures_nmi

