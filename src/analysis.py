import os
import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from src.distances import compute_distance_matrix
from src.clustering import Clustering
from sklearn import metrics
import mantel

def lines_ca_score(folder, score, options, col, eps=0.02):
    # figure of the clustering purity
    ca = 'dbscan'
    fig = plt.figure(figsize=(10, 7))
    i = 0
    for compress_opt in options:
        x = pd.read_csv(f'{folder}/clustering_{ca}_{eps}_{compress_opt}_{score}.csv', index_col=0)
        x.index = x.index.astype(str)
        plt.plot(x, color=col[i], marker="o", linestyle="-",
                 linewidth=3, markersize=10, label=compress_opt)
        i = i + 1
    plt.ylabel(f'{score} score', fontsize=15)
    plt.xlabel('Factors', fontsize=15)
    plt.legend(fontsize=15)
    plt.xticks(range(len(x)), x.index, fontsize=15, rotation=45)
    plt.yticks(fontsize=15)
    plt.tight_layout()
    plt.savefig(f'{folder}/lines-clustering-{score}.png', bbox_inches='tight')
    plt.close()

def time_mean(folder, item, factors, options, col):
    fig = plt.figure(figsize=(10, 7))
    i=0
    for compress_opt in options:
        x = pd.read_csv(f'{folder}/{compress_opt}-compression_{item}.csv')
        plt.plot(range(len(x.mean(axis=0))), x.mean(axis=0), color=col[i], marker="o", linestyle="-", linewidth=3,
        markersize=10, label=compress_opt)
        i = i+1
    plt.ylabel(f'Average of Compression {item}',fontsize=15)
    plt.xlabel('Factors',fontsize=15)
    plt.legend(fontsize=15)
    plt.xticks(range(len(x.mean(axis=0))), [str(i) for i in factors], fontsize=15, rotation=45)
    plt.yticks(fontsize=15)
    plt.tight_layout()
    plt.savefig(f'{folder}/lines-compression-{item}.png', bbox_inches='tight')
    plt.close()


def lines_compression(folder, metric='dtw', eps=0.02):
    options = ['DP', 'TR', 'SP', 'TR_SP', 'SP_TR']
    col = ['black', 'blue', 'green', 'orange', 'red']
    # options = ['DP']
    factors = [2, 1.5, 1, 1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32, 1 / 64, 1 / 128]

    time_mean(folder, 'rates', factors, options, col)
    time_mean(folder, 'times', factors, options, col)

    # # figure of the dtw avg time
    # fig = plt.figure(figsize=(10, 7))
    # i = 0
    # for compress_opt in options:
    #     times = pd.read_csv(f'{folder}/{metric}_{compress_opt}_times.csv')
    #     plt.plot(range(len(times.mean(axis=0))), times.mean(axis=0), color=col[i], marker="o", linestyle="-",
    #              linewidth=3, markersize=10, label=compress_opt)
    #     i = i + 1
    # plt.ylabel('Average of Processing Time (s)', fontsize=15)
    # plt.xlabel('Factors', fontsize=15)
    # plt.legend(fontsize=15)
    # plt.xticks(range(len(times.mean(axis=0))), times.columns, fontsize=15, rotation=45)
    # plt.yticks(fontsize=15)
    # plt.tight_layout()
    # plt.savefig(f'{folder}/lines-{metric}-avg-times.png', bbox_inches='tight')
    # plt.close()

    # # figure of the dtw total time
    # fig = plt.figure(figsize=(10, 7))
    # i = 0
    # for compress_opt in options:
    #     times = pd.read_csv(f'{folder}/{metric}_{compress_opt}_times.csv')
    #     plt.plot(times.sum(axis=0), color=col[i], marker="o", linestyle="-",
    #              linewidth=3, markersize=10, label=compress_opt)
    #     i = i + 1
    # plt.ylabel('Processing Time (s)', fontsize=15)
    # plt.xlabel('Factors', fontsize=15)
    # plt.legend(fontsize=15)
    # plt.xticks(range(len(times.sum(axis=0))), times.sum(axis=0).index, fontsize=15, rotation=45)
    # plt.yticks(fontsize=15)
    # plt.tight_layout()
    # plt.savefig(f'{folder}/lines-{metric}-times.png', bbox_inches='tight')
    # plt.close()

    # # figure of the {metric}+clustering time
    # fig = plt.figure(figsize=(10, 7))
    # i = 0
    # for compress_opt in options:
    #     times_cl = pd.read_csv(f'{folder}/clustering_{compress_opt}_times.csv', index_col=0)
    #     times = pd.read_csv(f'{folder}/{metric}_{compress_opt}_times.csv')
    #     times = (times.sum(axis=0) + times_cl.T).T
    #     plt.plot(times, color=col[i], marker="o", linestyle="-",
    #              linewidth=3, markersize=10, label=compress_opt)
    #     i = i + 1
    # plt.ylabel('Processing Time (s)', fontsize=15)
    # plt.xlabel('Factors', fontsize=15)
    # plt.legend(fontsize=15)
    # plt.xticks(range(len(times)), times.index, fontsize=15, rotation=45)
    # plt.yticks(fontsize=15)
    # plt.tight_layout()
    # plt.savefig(f'{folder}/lines-{metric}+clustering-times.png', bbox_inches='tight')
    # plt.close()

    # figure of the clustering time
    # fig = plt.figure(figsize=(10, 7))
    # i = 0
    # for compress_opt in options:
    #     times = pd.read_csv(f'{folder}/clustering_{compress_opt}_times.csv', index_col=0)
    #     plt.plot(times, color=col[i], marker="o", linestyle="-",
    #              linewidth=3, markersize=10, label=compress_opt)
    #     i = i + 1
    # plt.ylabel('Processing Time (s)', fontsize=15)
    # plt.xlabel('Factors', fontsize=15)
    # plt.legend(fontsize=15)
    # plt.xticks(range(len(times)), times.index, fontsize=15, rotation=45)
    # plt.yticks(fontsize=15)
    # plt.tight_layout()
    # plt.savefig(f'{folder}/lines-clustering-times.png', bbox_inches='tight')
    # plt.close()


    # # figure of the total compression time
    # fig = plt.figure(figsize=(10, 7))
    # i = 0
    # for compress_opt in options:
    #     times = pd.read_csv(f'{folder}/{compress_opt}-compression_times.csv')
    #     plt.plot(times.sum(), color=col[i], marker="o", linestyle="-",
    #              linewidth=3, markersize=10, label=compress_opt)
    #     i = i + 1
    # plt.ylabel('Processing Time (s)', fontsize=15)
    # plt.xlabel('Factors', fontsize=15)
    # plt.legend(fontsize=15)
    # plt.xticks(range(len(times.sum())), times.sum().index, fontsize=15, rotation=45)
    # plt.yticks(fontsize=15)
    # plt.tight_layout()
    # plt.savefig(f'{folder}/lines-total-compression-times.png', bbox_inches='tight')
    # plt.close()


    # figure of the total time
    fig = plt.figure(figsize=(10, 7))
    i = 0
    for compress_opt in options:
        times_cl = pd.read_csv(f'{folder}/clustering_{compress_opt}_times.csv', index_col=0)
        times_cl = times_cl * 1e-9
        times = pd.read_csv(f'{folder}/{metric}_{compress_opt}_times.csv')
        times = (times.sum(axis=0) + times_cl.T).T
        times_compression = pd.read_csv(f'{folder}/{compress_opt}-compression_times.csv')
        times_compression = times_compression * 1e-9
        times[1:] = (times[1:].T + times_compression.sum()).T
        plt.plot(times, color=col[i], marker="o", linestyle="-",
                 linewidth=3, markersize=10, label=compress_opt)
        i = i + 1
    plt.ylabel('Processing Time (s)', fontsize=15)
    plt.xlabel('Factors', fontsize=15)
    plt.legend(fontsize=15)
    plt.xticks(range(len(times)), times.index, fontsize=15, rotation=45)
    plt.yticks(fontsize=15)
    plt.tight_layout()
    plt.savefig(f'{folder}/lines-total-times.png', bbox_inches='tight')
    plt.close()

    # figure of the clustering purity
    # lines_ca_score(folder, 'purity', options, col, eps=eps)
    # lines_ca_score(folder, 'coverage', options, col, eps=eps)
    lines_ca_score(folder, 'mh', options, col, eps=eps)
    lines_ca_score(folder, 'nmi', options, col, eps=eps)
    # lines_ca_score(folder, 'ri', options, col, eps=eps)
    # lines_ca_score(folder, 'mi', options, col, eps=eps)
    lines_ca_score(folder, 'ami', options, col, eps=eps)
    lines_ca_score(folder, 'f1', options, col, eps=eps)
    # lines_ca_score(folder, 'ari', options, col, eps=eps)

    # figure of the pearson
    fig = plt.figure(figsize=(10, 7))
    i = 0
    for compress_opt in options:
        measure = pd.read_csv(f'{folder}/measures_{metric}_{compress_opt}_times.csv', index_col=0)
        measure = measure.loc['mantel-corr']
        plt.plot(measure, color=col[i], marker="o", linestyle="-",
                 linewidth=3, markersize=10, label=compress_opt)
        i = i + 1
    plt.ylabel('Mantel Correlation - Pearson', fontsize=15)
    plt.xlabel('Factors', fontsize=15)
    plt.legend(fontsize=15)
    plt.xticks(range(len(measure)), measure.index, fontsize=15, rotation=45)
    plt.yticks(fontsize=15)
    # plt.tight_layout()
    plt.savefig(f'{folder}/lines-measure-mantel.png', bbox_inches='tight')
    plt.close()

    # # figure of the spearman
    fig = plt.figure(figsize=(10, 7))
    i = 0
    for compress_opt in options:
        measure = pd.read_csv(f'{folder}/measures_{metric}_{compress_opt}_times.csv', index_col=0)
        measure = measure.loc['mantel-pvalue']
        plt.plot(measure, color=col[i], marker="o", linestyle="-",
                 linewidth=3, markersize=10, label=compress_opt)
        i = i + 1
    plt.ylabel('Mantel Test p-value', fontsize=15)
    plt.xlabel('Factors', fontsize=15)
    plt.legend(fontsize=15)
    plt.xticks(range(len(measure)), measure.index, fontsize=15, rotation=45)
    plt.yticks(fontsize=15)
    plt.tight_layout()
    plt.savefig(f'{folder}/lines-measure-mantel-pvalue.png', bbox_inches='tight')
    plt.close()


def factor_analysis(dataset, compress_opt, folder):
    factors = [2, 1.5, 1, 1/2, 1/4, 1/8, 1/16, 1/32, 1/64, 1/128]
    rates = pd.DataFrame()
    times = pd.DataFrame()
    for i in factors:
        comp_dataset, comp_rate, comp_times = dataset.compress_trips(compress=compress_opt, alpha=i)
        comp_times = comp_times #* 1e-9
        rates = pd.concat([rates, pd.DataFrame(comp_rate)], axis=1)
        times = pd.concat([times, pd.DataFrame(comp_times)], axis=1)
    rates.columns = [str(i) for i in factors]
    times.columns = [str(i) for i in factors]
    rates.to_csv(f'{folder}/{compress_opt}-compression_rates.csv', index=False)
    times.to_csv(f'{folder}/{compress_opt}-compression_times.csv', index=False)

    # fig = plt.figure(figsize=(10, 5))
    # rates.boxplot()
    # plt.ylabel('Compression Rate', fontsize=15)
    # plt.xlabel('Factor', fontsize=15)
    # plt.xticks(range(rates.shape[1]), rates.columns, fontsize=15, rotation=45)
    # plt.yticks(fontsize=15)
    # plt.tight_layout()
    # plt.savefig(f'{folder}/boxplot-{compress_opt}.png', bbox_inches='tight')
    # plt.close()
    #
    # fig = plt.figure(figsize=(10, 5))
    # times.boxplot()
    # plt.ylabel('Processing Time (s)', fontsize=15)
    # plt.xlabel('Factor', fontsize=15)
    # plt.xticks(range(times.shape[1]), times.columns, fontsize=15, rotation=45)
    # plt.yticks(fontsize=15)
    # plt.tight_layout()
    # plt.savefig(f'{folder}/boxplot-{compress_opt}-times.png', bbox_inches='tight')
    # plt.close()

    return rates, times


def get_time_dtw(path):
    up = pickle.load(open(path, 'rb'))
    up = pd.DataFrame.from_dict(up)
    up[up.isna()] = 0
    up = up.to_numpy()
    up = up[np.triu_indices_from(up)]
    # up = up * 1e-9
    return up


def factor_dist_analysis(dataset, compress_opt, folder, ncores=15, metric='dtw'):
    factors = [2, 1.5, 1, 1/2, 1/4, 1/8, 1/16, 1/32, 1/64, 1/128]
    times = pd.DataFrame()
    # comparing distances
    measures = {}
    features_folder = f'{folder}/NO/'
    if not os.path.exists(features_folder):
        os.makedirs(features_folder)
    features_path, main_time = compute_distance_matrix(dataset.get_dataset(), features_folder, verbose=True, njobs=ncores, metric=metric)

    dtw_raw = pickle.load(open(features_path, 'rb'))
    dtw_raw[np.isinf(dtw_raw)] = dtw_raw[~np.isinf(dtw_raw)].max() + 1
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
                                                                   njobs=ncores, metric=metric)
        dtw_factor = pickle.load(open(features_path, 'rb'))
        measures[i] = {}
        dtw_factor[np.isinf(dtw_factor)] = dtw_factor[~np.isinf(dtw_factor)].max() + 1
        measures[i]['mantel-corr'], measures[i]['mantel-pvalue'], _ = mantel.test(dtw_raw, dtw_factor, method='pearson', tail='upper')
        print(f"mantel - factor {i}: {measures[i]['mantel-corr']} - {measures[i]['mantel-pvalue']}")

        dtw_factor_time = get_time_dtw(feature_time)
        times = pd.concat([times, pd.DataFrame(dtw_factor_time)], axis=1)

    measures = pd.DataFrame(measures)
    measures.columns = [str(i) for i in factors]
    measures.to_csv(f'{folder}/measures_{metric}_{compress_opt}_times.csv')

    times.columns = ['no'] + [str(i) for i in factors]
    times.to_csv(f'{folder}/{metric}_{compress_opt}_times.csv', index=False)

    # fig = plt.figure(figsize=(10, 5))
    # times.boxplot()
    # plt.ylabel('Processing Time (s)', fontsize=15)
    # plt.xlabel('Factor', fontsize=15)
    # plt.xticks(range(times.shape[1]), times.columns, fontsize=15, rotation=45)
    # plt.yticks(fontsize=15)
    # plt.tight_layout()
    # plt.savefig(f'{folder}/boxplot-{metric}-{compress_opt}-times.png', bbox_inches='tight')
    # plt.close()

    return measures


def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


def factor_cluster_analysis(dataset, compress_opt, folder, ncores=15, ca='dbscan', eps=0.02, metric='dtw'):
    factors = [2, 1.5, 1, 1/2, 1/4, 1/8, 1/16, 1/32, 1/64, 1/128]
    measures_purity = {}
    measures_coverage = {}
    measures_mh = {}
    measures_mni = {}
    # measures_ri = {}
    # measures_mi = {}
    measures_ami = {}
    measures_f1 = {}
    # measures_ari = {}
    times = pd.DataFrame()
    times_cl = {}
    # comparing distances
    features_folder = f'{folder}/NO/'
    if not os.path.exists(features_folder):
        os.makedirs(features_folder)
    features_path, main_time = compute_distance_matrix(dataset.get_dataset(), features_folder, verbose=True, njobs=ncores, metric=metric)
    main_time = get_time_dtw(main_time)
    times = pd.concat([times, pd.DataFrame(main_time)], axis=1)
    #clustering
    model = Clustering(ais_data_path=dataset.preprocessed_path, distance_matrix_path=features_path,
                                       cluster_algorithm=ca, folder=features_folder, norm_dist=True, eps=eps)
    times_cl['no'] = model.time_elapsed
    labels_raw = model.labels
    for i in factors:
        comp_dataset, comp_rate, comp_times = dataset.compress_trips(compress=compress_opt, alpha=i)
        features_folder = f'{folder}/{compress_opt}-{i}/'
        if not os.path.exists(features_folder):
            os.makedirs(features_folder)
        print({features_folder})
        # DTW distances
        features_path, feature_time = compute_distance_matrix(comp_dataset, features_folder, verbose=True,
                                                                   njobs=ncores, metric=metric)
        # clustering
        model = Clustering(ais_data_path=dataset.preprocessed_path, distance_matrix_path=features_path,
                                      cluster_algorithm=ca, folder=features_folder, norm_dist=True, eps=eps)
        times_cl[str(i)] = model.time_elapsed
        labels_factor = model.labels
        measures_f1[str(i)] = metrics.f1_score(labels_raw, labels_factor, average='macro')
        measures_purity[str(i)] = purity_score(labels_raw, labels_factor)
        measures_coverage[str(i)] = purity_score(labels_factor, labels_raw)
        measures_mh[str(i)] = 2/(1/measures_purity[str(i)] + 1/measures_coverage[str(i)])
        measures_mni[str(i)] = metrics.normalized_mutual_info_score(labels_raw, labels_factor)
        # measures_ri[str(i)] = metrics.rand_score(labels_raw, labels_factor)
        # measures_mi[str(i)] = metrics.mutual_info_score(labels_raw, labels_factor)
        measures_ami[str(i)] = metrics.adjusted_mutual_info_score(labels_raw, labels_factor)
        # measures_ari[str(i)] = metrics.adjusted_rand_score(labels_raw, labels_factor)
        # print(f'Purity with factor {i}: {measures_purity[str(i)]}')
        # print(f'Coverage with factor {i}: {measures_coverage[str(i)]}')
        # print(f'MH with factor {i}: {measures_mh[str(i)]}')
        # print(f'NMI with factor {i}: {measures_mni[str(i)]}')
        # print(f"F1 macro with factor {i}: {metrics.f1_score(labels_raw, labels_factor, average='macro')}")
        # print(f'Rand_index with factor {i}: {measures_ri[str(i)]}')
        # print(f'MI with factor {i}: {measures_mi[str(i)]}')
        # print(metrics.confusion_matrix(labels_raw, labels_factor))
        # print(f'AMI with factor {i}: {measures_ami[str(i)]}')
        # print(f'ARI with factor {i}: {measures_ari[str(i)]}')

        # dtw_factor_time = get_time_dtw(feature_time)
        # times = pd.concat([times, pd.DataFrame(dtw_factor_time)], axis=1)

    measures_f1 = pd.Series(measures_f1)
    measures_f1.to_csv(f'{folder}/clustering_{ca}_{eps}_{compress_opt}_f1.csv')
    measures_purity = pd.Series(measures_purity)
    measures_purity.to_csv(f'{folder}/clustering_{ca}_{eps}_{compress_opt}_purity.csv')
    measures_coverage = pd.Series(measures_coverage)
    measures_coverage.to_csv(f'{folder}/clustering_{ca}_{eps}_{compress_opt}_coverage.csv')
    measures_mh = pd.Series(measures_mh)
    measures_mh.to_csv(f'{folder}/clustering_{ca}_{eps}_{compress_opt}_mh.csv')
    measures_mni = pd.Series(measures_mni)
    measures_mni.to_csv(f'{folder}/clustering_{ca}_{eps}_{compress_opt}_nmi.csv')
    # measures_ri = pd.Series(measures_ri)
    # measures_ri.to_csv(f'{folder}/clustering_{ca}_{eps}_{compress_opt}_ri.csv')
    # measures_ari = pd.Series(measures_ari)
    # measures_ari.to_csv(f'{folder}/clustering_{ca}_{eps}_{compress_opt}_ari.csv')
    # measures_mi = pd.Series(measures_mi)
    # measures_mi.to_csv(f'{folder}/clustering_{ca}_{eps}_{compress_opt}_mi.csv')
    measures_ami = pd.Series(measures_ami)
    measures_ami.to_csv(f'{folder}/clustering_{ca}_{eps}_{compress_opt}_ami.csv')

    times_cl = pd.Series(times_cl) #* 1e-9
    # times_cl = times.sum(axis=0) + times_cl
    times_cl.columns = ['no'] + [str(i) for i in factors]
    times_cl.to_csv(f'{folder}/clustering_{compress_opt}_times.csv')

    return measures_purity

