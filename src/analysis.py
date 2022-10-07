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


def lines_ca_score(folder, score, options, col, lines_style, mark_size, line_size, eps=0.02):
    # comp_lbl = {'DP': 'DP', 'TR': 'TR', 'SP': 'SB', 'TR_SP': 'TR+SB', 'SP_TR': 'SB+TR'}
    comp_lbl = {'DP': 'DP', 'TR': 'TR', 'SP': 'SB', 'TR_SP': 'TR+SB', 'SP_TR': 'SB+TR', 'DP_SP': 'DP+SB',
                'SP_DP': 'SB+DP', 'TR_DP': 'TR+DP', 'DP_TR': 'DP+TR'}
    # figure of the clustering purity
    ca = 'dbscan'
    fig = plt.figure(figsize=(10, 8))
    i = 0
    for compress_opt in options:
        x = pd.read_csv(f'{folder}/clustering_{ca}_{eps}_{compress_opt}_{score}.csv', index_col=0)
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
    plt.tight_layout()
    plt.savefig(f'{folder}/lines-clustering-{score}.png', bbox_inches='tight')
    plt.close()

def time_mean(folder, item, options, col, lines_style, mark_size, line_size):
    # comp_lbl = {'DP': 'DP', 'TR': 'TR', 'SP': 'SB', 'TR_SP': 'TR+SB', 'SP_TR': 'SB+TR'}
    comp_lbl = {'DP': 'DP', 'TR': 'TR', 'SP': 'SB', 'TR_SP': 'TR+SB', 'SP_TR': 'SB+TR', 'DP_SP': 'DP+SB',
                'SP_DP': 'SB+DP', 'TR_DP': 'TR+DP', 'DP_TR': 'DP+TR'}
    fig = plt.figure(figsize=(10, 8))
    i=0
    for compress_opt in options:
        x = pd.read_csv(f'{folder}/{compress_opt}-compression_{item}.csv')
        plt.plot(range(len(x.mean(axis=0))), x.mean(axis=0).iloc[x.shape[0]:None:-1], color=col[i], marker="p", linestyle=lines_style[i], linewidth=line_size[i],
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


def lines_compression(folder, metric='dtw', eps=0.02):
    rc('text', usetex=True)
    rc('text.latex', preamble=r'\usepackage{cmbright}')
    rc('font', size=25)
    rc('legend', fontsize=25)
    options = ['DP', 'TR', 'SP']
    # options = ['DP', 'TR', 'SP', 'TR_SP', 'SP_TR', 'DP_SP', 'SP_DP', 'TR_DP', 'DP_TR']
    # comp_lbl = {'DP': 'DP', 'TR': 'TR', 'SP': 'SB', 'TR_SP': 'TR+SB', 'SP_TR': 'SB+TR'}
    comp_lbl = {'DP': 'DP', 'TR': 'TR', 'SP': 'SB', 'TR_SP': 'TR+SB', 'SP_TR': 'SB+TR', 'DP_SP': 'DP+SB',
                'SP_DP': 'SB+DP', 'TR_DP': 'TR+DP', 'DP_TR': 'DP+TR'}
    # col = ['crimson', 'blue', 'green', 'darkorange', 'black']
    col = ['crimson', 'blue', 'green', 'darkorange', 'black', 'violet', 'chocolate', 'blueviolet', 'olive']
    col = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'black', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:olive']
    # lines_style = [(0, (3,1,1,1)), 'dotted', 'dashed', 'dashdot', 'solid']
    lines_style = [(0, (3,1,1,1)), (0, (5, 1)), (0, (3, 5, 1, 5)), 'dotted',
                   (0, (1, 3)), 'dashdot', (0, (3, 3, 1, 3)), (0, (3, 1, 1, 1, 1, 1)), 'solid']
    # mark_size = ['11', '9', '7', '5', '3']
    mark_size = ['11', '11', '11', '9', '9', '6', '6', '3', '3']
    # line_size = ['3', '2.5', '2', '1.5', '1']
    line_size = ['3', '3', '3', '2', '2', '1.5', '1.5', '1', '1']
    factors = [2, 1.5, 1, 1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32, 1 / 64, 1 / 128]
    # factors_str = [r'$2$', r'$1.5$', r'$1$', r'$\frac{1}{2}$', r'$\frac{1}{4}$', r'$\frac{1}{8}$',
    #                            r'$\frac{1}{16}$', r'$\frac{1}{32}$', r'$\frac{1}{64}$', r'$\frac{1}{128}$']
    factors_str = [r'$\frac{1}{128}$', r'$\frac{1}{64}$', r'$\frac{1}{32}$', r'$\frac{1}{16}$',
                   r'$\frac{1}{8}$', r'$\frac{1}{4}$', r'$\frac{1}{2}$', r'$1$', r'$1.5$', r'$2$']

    time_mean(folder, 'rates', options, col, lines_style, mark_size, line_size)
    # time_mean(folder, 'times', factors, options, col, lines_style, mark_size, line_size)

    # figure of the total time
    fig = plt.figure(figsize=(10,8))
    i = 0
    for compress_opt in options:
        times_cl = pd.read_csv(f'{folder}/clustering_{compress_opt}_times.csv', index_col=0)
        times = pd.read_csv(f'{folder}/{metric}_{compress_opt}_times.csv')
        if times.max().max() > 2e5:
            times.iloc[:,1:] = times.iloc[:,1:]
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
    lines_ca_score(folder, 'mh', options, col, eps=eps, lines_style=lines_style, mark_size=mark_size, line_size=line_size)
    lines_ca_score(folder, 'mi', options, col, eps=eps, lines_style=lines_style, mark_size=mark_size, line_size=line_size)
    lines_ca_score(folder, 'nmi', options, col, eps=eps, lines_style=lines_style, mark_size=mark_size, line_size=line_size)
    lines_ca_score(folder, 'ami', options, col, eps=eps, lines_style=lines_style, mark_size=mark_size, line_size=line_size)
    lines_ca_score(folder, 'f1', options, col, eps=eps, lines_style=lines_style, mark_size=mark_size, line_size=line_size)

    # figure of the pearson
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

    # # figure of the spearman
    fig = plt.figure(figsize=(10, 8))
    i = 0
    for compress_opt in options:
        measure = pd.read_csv(f'{folder}/measures_{metric}_{compress_opt}_times.csv', index_col=0)
        measure = measure.loc['mantel-pvalue']
        plt.plot(measure.iloc[measure.shape[0]:None:-1], color=col[i], marker="p", linestyle=lines_style[i],
                 linewidth=line_size[i], markersize=mark_size[i], label=comp_lbl[compress_opt])
        i = i + 1
    plt.ylabel('Mantel Test p-value', fontsize=25)
    plt.xlabel('Factors', fontsize=25)
    plt.legend(fontsize=18)
    plt.xticks(range(len(measure)), factors_str, fontsize=25)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.savefig(f'{folder}/lines-measure-mantel-pvalue.png', bbox_inches='tight')
    plt.close()


def factor_analysis(dataset_path, compress_opt, folder):
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


def get_time_dtw(path):
    up = pickle.load(open(path, 'rb'))
    up = pd.DataFrame.from_dict(up)
    up[up.isna()] = 0
    up = up.to_numpy()
    up = up[np.triu_indices_from(up)]
    # up = up * 1e-9
    return up


def factor_dist_analysis(dataset_path, compress_opt, folder, ncores=15, metric='dtw'):
    factors = [2, 1.5, 1, 1/2, 1/4, 1/8, 1/16, 1/32, 1/64, 1/128]
    times = pd.DataFrame()
    # comparing distances
    measures = {}
    features_folder = f'{folder}/NO/'
    if not os.path.exists(features_folder):
        os.makedirs(features_folder)
    features_path, main_time = compute_distance_matrix(get_raw_dataset(dataset_path), features_folder, verbose=True, njobs=ncores, metric=metric)

    dtw_raw = pickle.load(open(features_path, 'rb'))
    dtw_raw[np.isinf(dtw_raw)] = dtw_raw[~np.isinf(dtw_raw)].max() + 1
    dtw_raw_time = get_time_dtw(main_time)
    times = pd.concat([times, pd.DataFrame(dtw_raw_time)], axis=1)
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
        measures[i]['mantel-corr'], measures[i]['mantel-pvalue'], _ = mantel.test(dtw_raw, dtw_factor, method='spearman', tail='upper')
        print(f"mantel - factor {i}: {measures[i]['mantel-corr']} - {measures[i]['mantel-pvalue']}")

        dtw_factor_time = get_time_dtw(feature_time)
        times = pd.concat([times, pd.DataFrame(dtw_factor_time)], axis=1)

    measures = pd.DataFrame(measures)
    measures.columns = [str(i) for i in factors]
    measures.to_csv(f'{folder}/measures_{metric}_{compress_opt}_times.csv')

    times.columns = ['no'] + [str(i) for i in factors]
    times.to_csv(f'{folder}/{metric}_{compress_opt}_times.csv', index=False)

    return measures


def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


def factor_cluster_analysis(dataset_path, compress_opt, folder, ncores=15, ca='dbscan', eps=0.02, metric='dtw'):
    factors = [2, 1.5, 1, 1/2, 1/4, 1/8, 1/16, 1/32, 1/64, 1/128]
    measures_purity = {}
    measures_coverage = {}
    measures_mh = {}
    measures_mi = {}
    measures_nmi = {}
    measures_ami = {}
    measures_f1 = {}
    times_cl = {}
    # comparing distances
    features_folder = f'{folder}/NO/'
    if not os.path.exists(features_folder):
        os.makedirs(features_folder)
    features_path,_ = compute_distance_matrix(get_raw_dataset(dataset_path), features_folder, verbose=True, njobs=ncores, metric=metric)

    #clustering
    model = Clustering(ais_data_path=dataset_path, distance_matrix_path=features_path,
                                       cluster_algorithm=ca, folder=features_folder, norm_dist=True, eps=eps)
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
        model = Clustering(ais_data_path=dataset_path, distance_matrix_path=features_path,
                                      cluster_algorithm=ca, folder=features_folder, norm_dist=True, eps=eps)
        times_cl[str(i)] = model.time_elapsed
        labels_factor = model.labels
        measures_f1[str(i)] = metrics.f1_score(labels_raw, labels_factor, average='macro')
        measures_purity[str(i)] = purity_score(labels_raw, labels_factor)
        measures_coverage[str(i)] = purity_score(labels_factor, labels_raw)
        measures_mh[str(i)] = 2/(1/measures_purity[str(i)] + 1/measures_coverage[str(i)])
        measures_mi[str(i)] = metrics.mutual_info_score(labels_raw, labels_factor)
        measures_nmi[str(i)] = metrics.normalized_mutual_info_score(labels_raw, labels_factor)
        measures_ami[str(i)] = metrics.adjusted_mutual_info_score(labels_raw, labels_factor)

    measures_f1 = pd.Series(measures_f1)
    measures_f1.to_csv(f'{folder}/clustering_{ca}_{eps}_{compress_opt}_f1.csv')
    measures_purity = pd.Series(measures_purity)
    measures_purity.to_csv(f'{folder}/clustering_{ca}_{eps}_{compress_opt}_purity.csv')
    measures_coverage = pd.Series(measures_coverage)
    measures_coverage.to_csv(f'{folder}/clustering_{ca}_{eps}_{compress_opt}_coverage.csv')
    measures_mh = pd.Series(measures_mh)
    measures_mh.to_csv(f'{folder}/clustering_{ca}_{eps}_{compress_opt}_mh.csv')
    measures_mi = pd.Series(measures_mi)
    measures_mi.to_csv(f'{folder}/clustering_{ca}_{eps}_{compress_opt}_mi.csv')
    measures_nmi = pd.Series(measures_nmi)
    measures_nmi.to_csv(f'{folder}/clustering_{ca}_{eps}_{compress_opt}_nmi.csv')
    measures_ami = pd.Series(measures_ami)
    measures_ami.to_csv(f'{folder}/clustering_{ca}_{eps}_{compress_opt}_ami.csv')

    times_cl = pd.Series(times_cl)
    times_cl.columns = ['no'] + [str(i) for i in factors]
    times_cl.to_csv(f'{folder}/clustering_{compress_opt}_times.csv')

    return measures_purity

