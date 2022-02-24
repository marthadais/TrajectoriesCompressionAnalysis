import os
import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from src.distances import compute_distance_matrix
from src.clustering import Clustering

def lines_compression(folder):
    options = ['TR', 'DP', 'SP', 'TR_SP', 'SP_TR']
    # options = ['DP']
    factors = [2, 1.5, 1, 1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32, 1 / 64, 1 / 128]
    avgs = pd.DataFrame()
    fig = plt.figure(figsize=(10, 7))
    col=['red', 'blue', 'green', 'orange', 'yellow']
    i=0
    for compress_opt in options:
        rates = pd.read_csv(f'{folder}/{compress_opt}-compression_rates.csv')
        avgs = pd.concat([avgs, rates.mean(axis=0)], axis=1)
        plt.plot(range(len(rates.mean(axis=0))), rates.mean(axis=0), color=col[i], marker="o", linestyle="-", linewidth=3,
        markersize=10, label=compress_opt)
        i = i+1
    plt.ylabel('Average of Compression rate',fontsize=15)
    plt.xlabel('Factors',fontsize=15)
    plt.legend(fontsize=15)
    plt.xticks(range(len(rates.mean(axis=0))), [str(i) for i in factors], fontsize=15, rotation=45)
    plt.yticks(fontsize=15)
    plt.tight_layout()
    plt.savefig(f'{folder}/lines-compression.png', bbox_inches='tight')
    plt.close()

    # figure of the time compression
    fig = plt.figure(figsize=(10, 7))
    col = ['red', 'blue', 'green', 'orange', 'yellow']
    i = 0
    for compress_opt in options:
        times = pd.read_csv(f'{folder}/{compress_opt}-compression_times.csv')
        avgs = pd.concat([avgs, times.mean(axis=0)], axis=1)
        plt.plot(range(len(times.mean(axis=0))), times.mean(axis=0), color=col[i], marker="o", linestyle="-",
                 linewidth=3, markersize=10, label=compress_opt)
        i = i + 1
    plt.ylabel('Average of Processing Time', fontsize=15)
    plt.xlabel('Factors', fontsize=15)
    plt.legend(fontsize=15)
    plt.xticks(range(len(times.mean(axis=0))), [str(i) for i in factors], fontsize=15, rotation=45)
    plt.yticks(fontsize=15)
    plt.tight_layout()
    plt.savefig(f'{folder}/lines-compression-times.png', bbox_inches='tight')
    plt.close()

    # figure of the dtw time compression
    fig = plt.figure(figsize=(10, 7))
    col = ['red', 'blue', 'green', 'orange', 'yellow']
    i = 0
    for compress_opt in options:
        times = pd.read_csv(f'{folder}/dtw_{compress_opt}_times.csv')
        avgs = pd.concat([avgs, times.mean(axis=0)], axis=1)
        plt.plot(range(len(times.mean(axis=0))), times.mean(axis=0), color=col[i], marker="o", linestyle="-",
                 linewidth=3, markersize=10, label=compress_opt)
        i = i + 1
    plt.ylabel('Average of Processing Time (s)', fontsize=15)
    plt.xlabel('Factors', fontsize=15)
    plt.legend(fontsize=15)
    plt.xticks(range(len(times.mean(axis=0))), times.columns, fontsize=15, rotation=45)
    plt.yticks(fontsize=15)
    plt.tight_layout()
    plt.savefig(f'{folder}/lines-dtw-times.png', bbox_inches='tight')
    plt.close()

    # figure of the clustering time compression
    fig = plt.figure(figsize=(10, 7))
    col = ['red', 'blue', 'green', 'orange', 'yellow']
    i = 0
    for compress_opt in options:
        times = pd.read_csv(f'{folder}/clustering_{compress_opt}_times.csv', index_col=0)
        avgs = pd.concat([avgs, times], axis=1)
        plt.plot(times, color=col[i], marker="o", linestyle="-",
                 linewidth=3, markersize=10, label=compress_opt)
        i = i + 1
    plt.ylabel('Processing Time (s)', fontsize=15)
    plt.xlabel('Factors', fontsize=15)
    plt.legend(fontsize=15)
    plt.xticks(range(len(times)), times.index, fontsize=15, rotation=45)
    plt.yticks(fontsize=15)
    plt.tight_layout()
    plt.savefig(f'{folder}/lines-clustering-times.png', bbox_inches='tight')
    plt.close()


def factor_analysis(dataset, compress_opt, folder):
    factors = [2, 1.5, 1, 1/2, 1/4, 1/8, 1/16, 1/32, 1/64, 1/128]
    rates = pd.DataFrame()
    times = pd.DataFrame()
    for i in factors:
        comp_dataset, comp_rate, comp_times = dataset.compress_trips(compress=compress_opt, alpha=i)
        comp_times = comp_times * 1e-9
        rates = pd.concat([rates, pd.DataFrame(comp_rate)], axis=1)
        times = pd.concat([times, pd.DataFrame(comp_times)], axis=1)
    rates.columns = [str(i) for i in factors]
    times.columns = [str(i) for i in factors]
    rates.to_csv(f'{folder}/{compress_opt}-compression_rates.csv', index=False)
    times.to_csv(f'{folder}/{compress_opt}-compression_times.csv', index=False)

    fig = plt.figure(figsize=(10, 5))
    rates.boxplot()
    plt.ylabel('Compression Rate', fontsize=15)
    plt.xlabel('Factor', fontsize=15)
    plt.xticks(range(rates.shape[1]), rates.columns, fontsize=15, rotation=45)
    plt.yticks(fontsize=15)
    plt.tight_layout()
    plt.savefig(f'{folder}/boxplot-{compress_opt}.png', bbox_inches='tight')
    plt.close()

    fig = plt.figure(figsize=(10, 5))
    times.boxplot()
    plt.ylabel('Processing Time (s)', fontsize=15)
    plt.xlabel('Factor', fontsize=15)
    plt.xticks(range(times.shape[1]), times.columns, fontsize=15, rotation=45)
    plt.yticks(fontsize=15)
    plt.tight_layout()
    plt.savefig(f'{folder}/boxplot-{compress_opt}-times.png', bbox_inches='tight')
    plt.close()

    return rates, times


def permutation(m1, m2, measure='spearman', seed=1234, n_iter=5000, plot=False):
    """Nonparametric permutation testing Monte Carlo"""
    # https://towardsdatascience.com/how-to-measure-similarity-between-two-correlation-matrices-ce2ea13d8231
    np.random.seed(seed)
    rhos = []
    ps = []
    m1_v = m1[np.triu_indices_from(m1)]
    m2_v = m2[np.triu_indices_from(m2)]
    if measure == 'spearman':
        true_rho, perm_p = stats.spearmanr(m1_v, m2_v)
    elif measure == 'kendall':
        true_rho, perm_p = stats.kendalltau(m1_v, m2_v)
    else:
        true_rho, perm_p = stats.ttest_ind(m1_v, m2_v)
    # matrix permutation, shuffle the groups
    if n_iter > 0:
        m_ids1 = list(range(m1.shape[1]))
        m_ids2 = list(range(m2.shape[1]))
        for iter in range(n_iter):
            np.random.shuffle(m_ids1)  # shuffle list
            np.random.shuffle(m_ids2)
            m1_v = m1[m_ids1][:, m_ids1][np.triu_indices_from(m1[m_ids1][:, m_ids1])]
            m2_v = m2[m_ids2][:, m_ids2][np.triu_indices_from(m2[m_ids2][:, m_ids2])]
            # shuffle list
            if measure == 'spearman':
                r, p = stats.spearmanr(m1_v, m2_v)
            elif measure == 'kendall':
                r, p = stats.kendalltau(m1_v, m2_v)
            else:
                r, p = stats.ttest_ind(m1_v, m2_v)
            rhos.append(r)
            ps.append(p)
        perm_p = ((np.sum(np.abs(rhos) >= np.abs(true_rho)))+1) / (n_iter+1)  # two-tailed test
        # true_rho = np.mean(rhos)

        if plot:
            f, ax = plt.subplots()
            plt.hist(rhos, bins=20)
            ax.axvline(true_rho, color='r', linestyle='--')
            ax.set(title=f"{measure} Permuted p: {perm_p:.3f}", ylabel="counts", xlabel="rho")
            plt.show()

    return perm_p, true_rho


def get_time_dtw(path):
    up = pickle.load(open(path, 'rb'))
    up = pd.DataFrame.from_dict(up)
    up[up.isna()] = 0
    up = up.to_numpy()
    up = up[np.triu_indices_from(up)]
    up = up * 1e-9
    return up


def factor_dist_analysis(dataset, compress_opt, folder, ncores=4):
    factors = [2, 1.5, 1, 1/2, 1/4, 1/8, 1/16, 1/32, 1/64, 1/128]
    times = pd.DataFrame()
    times_cl = {}
    # comparing distances
    measures = {}
    features_folder = f'{folder}/NO/'
    if not os.path.exists(features_folder):
        os.makedirs(features_folder)
    features_path, main_time = compute_distance_matrix(dataset.get_dataset(), features_folder, verbose=True, njobs=ncores, metric='dtw')
    #clustering
    times_cl['no'] = Clustering(ais_data_path=dataset.preprocessed_path, distance_matrix_path=features_path,
                                       cluster_algorithm='hierarchical', folder=features_folder, norm_dist=True, k=5).time_elapsed

    dtw_raw = pickle.load(open(features_path, 'rb'))
    dtw_raw_time = get_time_dtw(main_time)
    times = pd.concat([times, pd.DataFrame(dtw_raw_time)], axis=1)
    for i in factors:
        comp_dataset, comp_rate, comp_times = dataset.compress_trips(compress=compress_opt, alpha=i)
        features_folder = f'{folder}/{compress_opt}-{i}/'
        if not os.path.exists(features_folder):
            os.makedirs(features_folder)
        print({features_folder})
        # DTW distances
        features_path, feature_time = compute_distance_matrix(comp_dataset, features_folder, verbose=True,
                                                                   njobs=ncores, metric='dtw')
        # clustering
        times_cl[str(i)] = Clustering(ais_data_path=dataset.preprocessed_path, distance_matrix_path=features_path,
                                      cluster_algorithm='hierarchical', folder=features_folder, norm_dist=True, k=5).time_elapsed
        dtw_factor = pickle.load(open(features_path, 'rb'))
        measures[i] = {}
        measures[i]['spearman-pvalue'], measures[i]['spearman-corr'] = permutation(dtw_raw, dtw_factor)
        measures[i]['Ttest_ind-pvalue'], measures[i]['Ttest_ind-sigma'] = permutation(dtw_raw, dtw_factor, measure='Ttest_ind')
        print(f"spearman - factor {i}: {measures[i]['spearman-corr']} - {measures[i]['spearman-pvalue']}")
        print(f"Ttest_ind - factor {i}: {measures[i]['Ttest_ind-sigma']} - {measures[i]['Ttest_ind-pvalue']}")

        dtw_factor_time = get_time_dtw(feature_time)
        times = pd.concat([times, pd.DataFrame(dtw_factor_time)], axis=1)

    measures = pd.DataFrame(measures)
    measures.columns = [str(i) for i in factors]

    times.columns = ['no'] + [str(i) for i in factors]
    times.to_csv(f'{folder}/dtw_{compress_opt}_times.csv', index=False)

    times_cl = pd.Series(times_cl) * 1e-9
    times_cl = times.sum(axis=0) + times_cl
    times_cl.to_csv(f'{folder}/clustering_{compress_opt}_times.csv')

    fig = plt.figure(figsize=(10, 5))
    times.boxplot()
    plt.ylabel('Processing Time (s)', fontsize=15)
    plt.xlabel('Factor', fontsize=15)
    plt.xticks(range(times.shape[1]), times.columns, fontsize=15, rotation=45)
    plt.yticks(fontsize=15)
    plt.tight_layout()
    plt.savefig(f'{folder}/boxplot-dtw-{compress_opt}-times.png', bbox_inches='tight')
    plt.close()

    return measures

