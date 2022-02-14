from preprocessing.clean_trajectories import Trajectories
from src.distances import DistanceMatrix
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


folder = f'./results/DCAIS/type_{vessel_type}/period_{start_day.date()}_to_{end_day.date()}/{metric}/'
# folder = f'./results/{dataset_name}/type_{vessel_type}/period_{start_day.date()}_to_{end_day.date()}/{metric}_fast/'
if region_limits is not None:
    folder = f'./results/DCAIS/type_{vessel_type}/period_{start_day.date()}_to_{end_day.date()}/{n_samples}_{region_limits}/{metric}_fast/'

dim_set = ['lat', 'lon']
ca = 'spectral'

dataset_dict = dataset.get_dataset()
features_path = f'{folder}/features_distance.p'
if not os.path.exists(features_path):
    features = DistanceMatrix(dataset=dataset_dict, features_opt=metric, dim_set=dim_set, folder=folder)
result = Clustering(ais_data_path=dataset.preprocessed_path, distance_matrix_path=features_path,
                        cluster_algorithm=ca, folder=folder, norm_dist=True)

print('TR')
tr_dataset = dataset.get_dataset(compress='TR')
features_tr_folder = f'{folder}/TR/'
if not os.path.exists(features_tr_folder):
    features_tr = DistanceMatrix(dataset=tr_dataset, features_opt=metric, dim_set=dim_set, folder=features_tr_folder)
result = Clustering(ais_data_path=dataset.compress_path, distance_matrix_path=f'{features_tr_folder}/features_distance.p',
                        cluster_algorithm=ca, folder=features_tr_folder, norm_dist=True)

print('PD')
dp_dataset = dataset.get_dataset(compress='PD')
features_dp_folder = f'{folder}/PD/'
if not os.path.exists(features_dp_folder):
    features_dp = DistanceMatrix(dataset=dp_dataset, features_opt=metric, dim_set=dim_set, folder=features_dp_folder)
result = Clustering(ais_data_path=dataset.compress_path, distance_matrix_path=f'{features_dp_folder}/features_distance.p',
                        cluster_algorithm=ca, folder=features_dp_folder, norm_dist=True)

print('SP')
sp_dataset = dataset.get_dataset(compress='SP')
features_sp_folder = f'{folder}/SP/'
if not os.path.exists(features_sp_folder):
    features_sp = DistanceMatrix(dataset=sp_dataset, features_opt=metric, dim_set=dim_set, folder=features_sp_folder)
result = Clustering(ais_data_path=dataset.compress_path, distance_matrix_path=f'{features_sp_folder}/features_distance.p',
                    cluster_algorithm=ca, folder=features_sp_folder, norm_dist=True)

# distance matrix significant test
from scipy.stats import ttest_ind
import pickle
import numpy as np

dtw_raw = pickle.load(open(f'{folder}/features_distance.p','rb'))
dtw_raw = np.triu(dtw_raw).ravel()
dtw_tr = pickle.load(open(f'{features_tr_folder}/features_distance.p','rb'))
dtw_tr = np.triu(dtw_tr).ravel()
dtw_dp = pickle.load(open(f'{features_dp_folder}/features_distance.p','rb'))
dtw_dp = np.triu(dtw_dp).ravel()
dtw_sp = pickle.load(open(f'{features_sp_folder}/features_distance.p','rb'))
dtw_sp = np.triu(dtw_sp).ravel()

res_tr = ttest_ind(dtw_raw, dtw_tr)
res_dp = ttest_ind(dtw_raw, dtw_dp)
res_sp = ttest_ind(dtw_raw, dtw_sp)

print(res_tr)
print(res_dp)
print(res_sp)

