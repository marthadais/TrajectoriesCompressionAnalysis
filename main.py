from preprocessing.clean_trajectories import Trajectories
from src.distances import compute_distance_matrix
from datetime import datetime
from src.clustering import Clustering
import os

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
features_path = compute_distance_matrix(dataset_dict, curr_folder, verbose=True, njobs=ncores, metric='dtw')
# result = Clustering(ais_data_path=dataset.preprocessed_path, distance_matrix_path=features_path,
#                         cluster_algorithm=ca, folder=curr_folder, norm_dist=True, k=10)

print('TR')
compress_opt = 'TR'
tr_dataset = dataset.get_dataset(compress=compress_opt, alpha=factor)
features_tr_folder = f'{folder}/{compress_opt}-{factor}/'
features_tr_path = compute_distance_matrix(tr_dataset, features_tr_folder, verbose=True, njobs=ncores, metric='dtw')
# result_tr = Clustering(ais_data_path=dataset.compress_path, distance_matrix_path=features_tr_path,
#                         cluster_algorithm=ca, folder=features_tr_folder, norm_dist=True, k=10)

print('PD')
compress_opt = 'PD'
dp_dataset = dataset.get_dataset(compress=compress_opt, alpha=factor)
features_dp_folder = f'{folder}/{compress_opt}-{factor}/'
features_dp_path = compute_distance_matrix(dp_dataset, features_dp_folder, verbose=True, njobs=ncores, metric='dtw')
# result_dp = Clustering(ais_data_path=dataset.compress_path, distance_matrix_path=features_dp_path,
#                         cluster_algorithm=ca, folder=features_dp_folder, norm_dist=True, k=10)

print('SP')
compress_opt = 'SP'
sp_dataset = dataset.get_dataset(compress=compress_opt, alpha=factor)
features_sp_folder = f'{folder}/{compress_opt}-{factor}/'
features_sp_path = compute_distance_matrix(sp_dataset, features_sp_folder, verbose=True, njobs=3, metric='dtw')
# result_sp = Clustering(ais_data_path=dataset.compress_path, distance_matrix_path=features_sp_path,
#                     cluster_algorithm=ca, folder=features_sp_folder, norm_dist=True, k=10)

# distance matrix significant test
from scipy.stats import ttest_ind
import pickle
import numpy as np

dtw_raw = pickle.load(open(features_path,'rb'))
dtw_raw = np.triu(dtw_raw).ravel()
# dtw_raw = (dtw_raw-dtw_raw.mean())/dtw_raw.std()
dtw_raw = dtw_raw/dtw_raw.max()
dtw_tr = pickle.load(open(features_tr_path,'rb'))
dtw_tr = np.triu(dtw_tr).ravel()
# dtw_tr = (dtw_tr-dtw_tr.mean())/dtw_tr.std()
dtw_tr = dtw_tr/dtw_tr.max()
dtw_dp = pickle.load(open(features_dp_path,'rb'))
dtw_dp = np.triu(dtw_dp).ravel()
# dtw_dp = (dtw_dp-dtw_dp.mean())/dtw_dp.std()
dtw_dp = dtw_dp/dtw_dp.max()
dtw_sp = pickle.load(open(features_sp_path,'rb'))
dtw_sp = np.triu(dtw_sp).ravel()
# dtw_sp = (dtw_sp-dtw_sp.mean())/dtw_sp.std()
dtw_sp = dtw_sp/dtw_sp.max()



res_tr = ttest_ind(dtw_raw, dtw_tr)
res_dp = ttest_ind(dtw_raw, dtw_dp)
res_sp = ttest_ind(dtw_raw, dtw_sp)

print(res_tr)
print(res_dp)
print(res_sp)


