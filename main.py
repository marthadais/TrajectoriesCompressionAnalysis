from preprocessing.clean_trajectories import Trajectories
from src.distances import compute_distance_matrix
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import pickle
import numpy as np
from src.clustering import Clustering
import os

def lines_compression(folder):
    options = ['TR', 'DP', 'SP', 'TR_SP', 'SP_TR']
    # options = ['DP']
    factors = [2, 1.5, 1, 1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32, 1 / 64, 1 / 128]
    avgs = pd.DataFrame()
    fig = plt.figure(figsize=(10, 7))
    col=['red', 'blue', 'green', 'orange']
    i=0
    for compress_opt in options:
        rates = pd.read_csv(f'{folder}/{compress_opt}-compression_rates.csv')
        avgs = pd.concat([avgs, rates.mean(axis=0)], axis=1)
        plt.plot(range(len(rates.mean(axis=0))), rates.mean(axis=0), color=col[i], marker="o", linestyle="-", linewidth=3,
        markersize=10)
        i = i+1
    plt.ylabel('Average of Compression rate',fontsize=15)
    plt.xlabel('Factors',fontsize=15)
    plt.legend(fontsize=15)
    plt.xticks(range(len(rates.mean(axis=0))), [str(i) for i in factors], fontsize=15, rotation=45)
    plt.yticks(fontsize=15)
    plt.tight_layout()
    plt.savefig(f'{folder}/lines-compression.png', bbox_inches='tight')
    plt.close()

    fig = plt.figure(figsize=(10, 7))
    col = ['red', 'blue', 'green', 'orange']
    i = 0
    for compress_opt in options:
        times = pd.read_csv(f'{folder}/{compress_opt}-compression_times.csv')
        avgs = pd.concat([avgs, times.mean(axis=0)], axis=1)
        plt.plot(range(len(times.mean(axis=0))), times.mean(axis=0), color=col[i], marker="o", linestyle="-",
                 linewidth=3,
                 markersize=10)
        i = i + 1
    plt.ylabel('Average of Processing Time', fontsize=15)
    plt.xlabel('Factors', fontsize=15)
    plt.legend(fontsize=15)
    plt.xticks(range(len(times.mean(axis=0))), [str(i) for i in factors], fontsize=15, rotation=45)
    plt.yticks(fontsize=15)
    plt.tight_layout()
    plt.savefig(f'{folder}/lines-compression-times.png', bbox_inches='tight')
    plt.close()


def factor_analysis(compress_opt, folder):
    factors = [2, 1.5, 1, 1/2, 1/4, 1/8, 1/16, 1/32, 1/64, 1/128]
    rates = pd.DataFrame()
    times = pd.DataFrame()
    for i in factors:
        comp_dataset, comp_rate, comp_times = dataset.compress_trips(compress=compress_opt, alpha=i)
        rates = pd.concat([rates, pd.DataFrame(comp_rate)], axis=1)
        times = pd.concat([times, pd.DataFrame(comp_times)], axis=1)
    rates.columns = [str(i) for i in factors]
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
    plt.ylabel('Processing Time', fontsize=15)
    plt.xlabel('Factor', fontsize=15)
    plt.xticks(range(times.shape[1]), times.columns, fontsize=15, rotation=45)
    plt.yticks(fontsize=15)
    plt.tight_layout()
    plt.savefig(f'{folder}/boxplot-{compress_opt}-times.png', bbox_inches='tight')
    plt.close()

    return rates, times


print('Starting')
### Reading and cleaning dataset
# Number of vessels
n_samples = 30
# Fishing type
vessel_type = [30, 1001, 1002, 70]
# Time period
start_day = datetime(2020, 4, 1)
end_day = datetime(2020, 4, 30)
# Attributes
dim_set = ['lat', 'lon']
# polygon region
region_limits = [48, 49, -124, -123]
#compression technique
compress_opt='PD'

# Creating dataset
dataset = Trajectories(n_samples=n_samples, vessel_type=vessel_type, time_period=(start_day, end_day),
                       region=region_limits)

#### Computing Distances
metric = 'dtw'
ncores = 4


folder = f'./results/DCAIS/type_{vessel_type}/period_{start_day.date()}_to_{end_day.date()}/{metric}/'
# folder = f'./results/{dataset_name}/type_{vessel_type}/period_{start_day.date()}_to_{end_day.date()}/{metric}_fast/'
if region_limits is not None:
    folder = f'./results/DCAIS/type_{vessel_type}/period_{start_day.date()}_to_{end_day.date()}/{n_samples}_{region_limits}/{metric}_fast/'

dim_set = ['lat', 'lon']
factor = 1
ca = 'hierarchical'
k = 10


compress_opt = 'NO'
curr_folder = f'{folder}/{compress_opt}'
dataset_dict = dataset.get_dataset()
features_path, _ = compute_distance_matrix(dataset_dict, curr_folder, verbose=True, njobs=ncores, metric='dtw')
# result = Clustering(ais_data_path=dataset.preprocessed_path, distance_matrix_path=features_path,
#                         cluster_algorithm=ca, folder=curr_folder, norm_dist=True, k=10)

rates1, times1 = factor_analysis('DP', folder)
rates2, times2 = factor_analysis('TR', folder)
rates3, times3 = factor_analysis('SP', folder)
rates4, times4 = factor_analysis('TR_SP', folder)
rates5, times5 = factor_analysis('SP_TR', folder)

lines_compression(folder)

print('TR')
compress_opt = 'TR'
tr_dataset, comp_rate_tr, comp_time_tr = dataset.compress_trips(compress=compress_opt, alpha=factor)
features_tr_folder = f'{folder}/{compress_opt}-{factor}/'
features_tr_path, _ = compute_distance_matrix(tr_dataset, features_tr_folder, verbose=True, njobs=ncores, metric='dtw')
# result_tr = Clustering(ais_data_path=dataset.compress_path, distance_matrix_path=features_tr_path,
#                         cluster_algorithm=ca, folder=features_tr_folder, norm_dist=True, k=10)

print('DP')
compress_opt = 'DP'
dp_dataset, comp_rate_dp, comp_time_dp = dataset.compress_trips(compress=compress_opt, alpha=factor)
features_dp_folder = f'{folder}/{compress_opt}-{factor}/'
features_dp_path, _ = compute_distance_matrix(dp_dataset, features_dp_folder, verbose=True, njobs=ncores, metric='dtw')
# result_dp = Clustering(ais_data_path=dataset.compress_path, distance_matrix_path=features_dp_path,
#                         cluster_algorithm=ca, folder=features_dp_folder, norm_dist=True, k=10)

print('SP')
compress_opt = 'SP'
sp_dataset, comp_rate_sp, comp_time_sp = dataset.compress_trips(compress=compress_opt, alpha=factor)
features_sp_folder = f'{folder}/{compress_opt}-{factor}/'
features_sp_path, _ = compute_distance_matrix(sp_dataset, features_sp_folder, verbose=True, njobs=3, metric='dtw')
# result_sp = Clustering(ais_data_path=dataset.compress_path, distance_matrix_path=features_sp_path,
#                     cluster_algorithm=ca, folder=features_sp_folder, norm_dist=True, k=10)

print('TR_SP')
compress_opt = 'TR_SP'
tr_sp_dataset, comp_rate_tr_sp, comp_time_tr_sp = dataset.compress_trips(compress=compress_opt, alpha=factor)
features_tr_sp_folder = f'{folder}/{compress_opt}-{factor}/'
features_tr_sp_path, _ = compute_distance_matrix(tr_sp_dataset, features_tr_sp_folder, verbose=True, njobs=3, metric='dtw')
# result_sp = Clustering(ais_data_path=dataset.compress_path, distance_matrix_path=features_sp_path,
#                     cluster_algorithm=ca, folder=features_sp_folder, norm_dist=True, k=10)

print('SP_TR')
compress_opt = 'SP_TR'
sp_tr_dataset, comp_rate_sp_tr, comp_time_sp_tr = dataset.compress_trips(compress=compress_opt, alpha=factor)
features_sp_tr_folder = f'{folder}/{compress_opt}-{factor}/'
features_sp_tr_path, _ = compute_distance_matrix(sp_tr_dataset, features_sp_tr_folder, verbose=True, njobs=3, metric='dtw')
# result_sp = Clustering(ais_data_path=dataset.compress_path, distance_matrix_path=features_sp_path,
#                     cluster_algorithm=ca, folder=features_sp_folder, norm_dist=True, k=10)

# distance matrix significant test


dtw_raw = pickle.load(open(features_path,'rb'))
# dtw_raw = np.triu(dtw_raw).ravel()
# dtw_raw = (dtw_raw-dtw_raw.mean())/dtw_raw.std()
dtw_raw = dtw_raw/dtw_raw.max()

dtw_tr = pickle.load(open(features_tr_path,'rb'))
# dtw_tr = np.triu(dtw_tr).ravel()
# dtw_tr = (dtw_tr-dtw_tr.mean())/dtw_tr.std()
dtw_tr = dtw_tr/dtw_tr.max()

dtw_dp = pickle.load(open(features_dp_path,'rb'))
# dtw_dp = np.triu(dtw_dp).ravel()
# dtw_dp = (dtw_dp-dtw_dp.mean())/dtw_dp.std()
# dtw_dp = dtw_dp/dtw_dp.max()

dtw_sp = pickle.load(open(features_sp_path,'rb'))
# dtw_sp = np.triu(dtw_sp).ravel()
# dtw_sp = (dtw_sp-dtw_sp.mean())/dtw_sp.std()
dtw_sp = dtw_sp/dtw_sp.max()

dtw_tr_sp = pickle.load(open(features_tr_sp_path,'rb'))
# dtw_tr_sp = np.triu(dtw_tr_sp).ravel()
# dtw_tr_sp = (dtw_tr_sp-dtw_tr_sp.mean())/dtw_tr_sp.std()
dtw_tr_sp = dtw_tr_sp/dtw_tr_sp.max()

dtw_sp_tr = pickle.load(open(features_sp_tr_path,'rb'))
# dtw_sp_tr = np.triu(dtw_sp_tr).ravel()
# dtw_sp_tr = (dtw_sp_tr-dtw_sp_tr.mean())/dtw_sp_tr.std()
dtw_sp_tr = dtw_sp_tr/dtw_sp_tr.max()

def permutation(m1, m2, measure='spearman', seed=0, n_iter=0):
    """Nonparametric permutation testing Monte Carlo"""
    # https://towardsdatascience.com/how-to-measure-similarity-between-two-correlation-matrices-ce2ea13d8231
    np.random.seed(seed)
    rhos = []
    if measure == 'spearman':
        true_rho, perm_p = stats.spearmanr(np.triu(m1).ravel(), np.triu(m2).ravel())
    elif measure == 'kendall':
        true_rho, perm_p = stats.kendalltau(np.triu(m1).ravel(), np.triu(m2).ravel())
    else:
        true_rho, perm_p = stats.ttest_ind(np.triu(m1).ravel(), np.triu(m2).ravel())
    # matrix permutation, shuffle the groups
    m_ids = list(range(m1.shape[1]))
    m2_v = np.triu(m2).ravel()
    if n_iter > len(m_ids):
        n_iter = len(m_ids)
    if n_iter > 0:
        for iter in range(n_iter):
            np.random.shuffle(m_ids)  # shuffle list
            if measure == 'spearman':
                r, _ = stats.spearmanr(np.triu(m1[m_ids, m_ids]).ravel(), m2_v)
            elif measure == 'kendall':
                r, _ = stats.kendalltau(np.triu(m1[m_ids, m_ids]).ravel(), m2_v)
            else:
                r, _ = stats.ttest_ind(np.triu(m1[m_ids, m_ids]).ravel(), m2_v)
            rhos.append(r)
        perm_p = ((np.sum(np.abs(true_rho) <= np.abs(rhos))) + 1) / (n_iter + 1)  # two-tailed test
    return perm_p


print(f"spearman: {permutation(1-dtw_raw, 1-dtw_raw)}")
print(f"kendall: {permutation(1-dtw_raw, 1-dtw_raw, measure='kendall')}")
print(f"Ttest_ind: {permutation(1-dtw_raw, 1-dtw_raw, measure='Ttest_ind')}")

print(f"spearman: {permutation(1-dtw_raw, 1-dtw_dp)}")
print(f"kendall: {permutation(1-dtw_raw, 1-dtw_dp, measure='kendall')}")
print(f"Ttest_ind: {permutation(1-dtw_raw, 1-dtw_dp, measure='Ttest_ind')}")

print(f"spearman: {permutation(1-dtw_raw, 1-dtw_tr)}")
print(f"kendall: {permutation(1-dtw_raw, 1-dtw_tr, measure='kendall')}")
print(f"Ttest_ind: {permutation(1-dtw_raw, 1-dtw_tr, measure='Ttest_ind')}")

print(f"spearman: {permutation(1-dtw_raw, 1-dtw_sp)}")
print(f"kendall: {permutation(1-dtw_raw, 1-dtw_sp, measure='kendall')}")
print(f"Ttest_ind: {permutation(1-dtw_raw, 1-dtw_sp, measure='Ttest_ind')}")



